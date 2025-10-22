#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S0 (GROBID) → s0.json (тонкий формат с плоским списком предложений)

Формат ответа:
{
  "doc_id": "...",
  "source_pdf": ".../input.pdf",
  "page_count": 7,
  "metadata": {
    "title": "...",
    "author": "...",
    "producer": "",
    "creationDate": "...",
    "arxiv_id": ""
  },
  "sentences": [
    {
      "text": "...",
      "page": 0,
      "bbox": [x0,y0,x1,y1],
      "section_hint": "INTRO|METHODS|RESULTS|DISCUSSION|REFERENCES|OTHER",
      "is_caption": false,
      "caption_type": ""
    }
  ]
}

Подход к TEI:
- Однопроходный итератор по узлам TEI — обновляем текущий <head> → IMRAD,
  эмитим <s>, <figDesc>, <table><head>.
- Основан на идее tei_iter_sentences из imrad_grobid_pipeline.py.
"""

import os
import re
from typing import Iterable
import json
import datetime
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests
from lxml import etree


# ───────────────────────────────────────────────────────────
# GROBID HTTP
# ───────────────────────────────────────────────────────────

# GROBID HTTP: call processFulltextDocument and return TEI-XML.
def grobid_fulltext_tei(server: str, pdf_path: str, timeout: int = 120) -> str:
    """
    Вызов GROBID /api/processFulltextDocument и возврат TEI-XML.
    """
    url = server.rstrip("/") + "/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        files = {"input": f}
        data = [
            ("segmentSentences", "1"),
            ("teiCoordinates", "s"),
            ("teiCoordinates", "p"),
            ("teiCoordinates", "head"),
            ("teiCoordinates", "figure"),
            ("teiCoordinates", "table"),
            ("teiCoordinates", "biblStruct"),
        ]
        r = requests.post(url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    return r.text


# ——— Noisy <head> filter ———
IGNORE_NOISY_HEADERS = True  # можно вынести в аргументы CLI/ENV
NOISY_HEAD_MAX_HEIGHT = 12.0  # эмпирически: очень плоские «пробегающие» заголовки


def _should_ignore_head(el, imrad_label: str) -> bool:
    if not IGNORE_NOISY_HEADERS:
        return False
    # игнорируем только «безопасные» разделы, где часто бегут колонтитулы
    if imrad_label not in {"INTRO", "REFERENCES"}:
        return False
    boxes = parse_coords_attr(el.get("coords") or "")
    page, bbox = union_bbox(boxes)
    if not bbox or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    h = max(0.0, float(y1) - float(y0))
    return h < NOISY_HEAD_MAX_HEIGHT


# ───────────────────────────────────────────────────────────
# Координаты / нормализация
# ───────────────────────────────────────────────────────────

COORD_ITEM_RE = re.compile(
    r"^\s*(?:p)?(?P<page>\d+)\s*[: ,]\s*(?P<x>[-\d.]+)\s*,\s*(?P<y>[-\d.]+)\s*,\s*(?P<w>[-\d.]+)\s*,\s*(?P<h>[-\d.]+)"
)


# Parse TEI `coords` attribute into a list of page-local boxes.
def parse_coords_attr(coords: str) -> List[Dict]:
    if not coords:
        return []
    boxes = []
    for chunk in coords.split(";"):
        m = COORD_ITEM_RE.match(chunk)
        if not m:
            continue
        page = int(m.group("page"))
        x = float(m.group("x"));
        y = float(m.group("y"))
        w = float(m.group("w"));
        h = float(m.group("h"))
        boxes.append({"page": page, "x": x, "y": y, "w": w, "h": h})
    return boxes


# Compute a union bbox across same-page boxes; returns (page, [x0,y0,x1,y1]).
def union_bbox(boxes: List[Dict]) -> Tuple[Optional[int], List[float]]:
    if not boxes:
        return None, [0, 0, 0, 0]
    page = boxes[0]["page"]
    xs0 = [b["x"] for b in boxes if b["page"] == page]
    ys0 = [b["y"] for b in boxes if b["page"] == page]
    xs1 = [b["x"] + b["w"] for b in boxes if b["page"] == page]
    ys1 = [b["y"] + b["h"] for b in boxes if b["page"] == page]
    return page, [min(xs0), min(ys0), max(xs1), max(ys1)]


_WS_MULTI = re.compile(r"[ \t\u00A0]+")
_NO_SPACE_BEFORE = re.compile(r"\s+([),.;:\]\}])")
_NO_SPACE_AFTER = re.compile(r"([(\[\{])\s+")
_JOIN_FIGTAB = re.compile(r"(\S)(?=(Figure|Table)\b)")
_FIX_PERCENT = re.compile(r"(%)([A-Za-z])")  # "60%of" → "60% of"


# Normalize inline whitespace/punctuation artifacts in TEI text.
def normalize_inline(text: str) -> str:
    if not text:
        return text
    t = text.replace("\r", "")
    t = _WS_MULTI.sub(" ", t)
    t = _NO_SPACE_BEFORE.sub(r"\1", t)
    t = _NO_SPACE_AFTER.sub(r"\1", t)
    t = _JOIN_FIGTAB.sub(r"\1 ", t)
    t = _FIX_PERCENT.sub(r"% \2", t)
    return t.strip()


# ───────────────────────────────────────────────────────────
# Citation detection (TEI + текстовые паттерны с гвардами)
# ───────────────────────────────────────────────────────────

# (A) квадратные числовые ссылки: [1], [2–5], [3, 7, 9]
_RX_NUMERIC_BRACKETS = re.compile(r"""
    (?<![A-Za-z])     # не массив A[i]
    \[
      \s*\d{1,3}
      (?:\s*(?:[-–,;]\s*|\s*,\s*)\d{1,3}){0,6}
    \]                # <-- без \b
""", re.X)

# (B) автор–год: (Smith, 2019) ; (Smith et al., 2019; Wang, 2021)
# гварды исключают служебные скобки (Fig., Table, Eq., p<, n=)
_YEAR_MIN = 1800
_YEAR_MAX = datetime.datetime.now().year + 1
_RX_PAREN_AUTHOR_YEAR = re.compile(
    r"""
    \(
      \s*
      [A-Z][A-Za-z'’\-]+                           # фамилия 1
      (?:\s+et\s+al\.)?                            # опционально et al.
      (?:\s*,\s*(?P<y1>(?:18|19|20|21)\d{2}))      # , 2019
      (?:                                          # ; Smith, 2020 ; Wang, 2021
        \s*[,;]\s*
        [A-Z][A-Za-z'’\-]+(?:\s+et\s+al\.)?\s*,\s*(?P<yN>(?:18|19|20|21)\d{2})
      )*
      \s*
    \)
    """,
    re.X
)

# (C) негативные паттерны в круглых скобках, которые НЕ цитаты:
# (n = 20), (p < 0.05), (CI 95%), (Fig. 2), (Table 1), (Eq. 3)
_RX_PAREN_NON_CITATION = re.compile(
    r"""
    \(
      [^)]{0,6}                # короткий префикс
      (?:                      # набор «служебных» маркеров
        n\s*=\s*\d+ |
        p\s*[<≤=]\s*0?\.\d+ |
        CI\s*\d{1,3}\s*% |
        Fig\.?\s*\d+ |
        Table\s*\d+ |
        Eq\.?\s*\d+
      )
      [^)]{0,20}
    \)
    """, re.X | re.I
)


def caption_boxes_with_fallback(caption_el, parent_tag: str) -> List[Dict]:
    """
    Возвращает списки боксов для caption: сначала пробуем сам caption,
    затем — родитель (figure/table). Если есть оба — склеиваем.
    """
    boxes = parse_coords_attr(caption_el.get("coords") or "")
    parent = caption_el.getparent()
    if parent is not None and etree.QName(parent).localname.lower() == parent_tag:
        parent_boxes = parse_coords_attr(parent.get("coords") or "")
    else:
        parent_boxes = []
    if boxes and parent_boxes:
        return boxes + parent_boxes
    return boxes or parent_boxes


def _has_citation_struct(el) -> bool:
    """Структурный TEI-сигнал: <ref type="bibl">…</ref> или target="#b…"."""
    try:
        for ref in el.iterfind(".//ref"):
            ty = (ref.get("type") or "").lower()
            tgt = (ref.get("target") or "")
            if ty.startswith("bibl"):
                return True
            if tgt.startswith("#b"):
                return True
    except Exception:
        pass
    return False


def _text_has_numeric_brackets(text: str) -> bool:
    return bool(_RX_NUMERIC_BRACKETS.search(text or ""))


def _text_has_author_year(text: str) -> bool:
    t = text or ""
    if _RX_PAREN_NON_CITATION.search(t):  # <— этот гвард оставляем
        return False
    m = _RX_PAREN_AUTHOR_YEAR.search(t)
    if not m:
        return False
    # Валидация годов (1800..текущий+1) — оставляем как было
    for y in (m.groupdict().get("y1"), m.groupdict().get("yN")):
        if y:
            yi = int(y)
            if yi < _YEAR_MIN or yi > _YEAR_MAX:
                return False
    return True


def compute_citation_flags(el, text: str) -> tuple[bool, float]:
    """
    Возвращает (has_citation, citation_strength) c приоритетом:
      1. TEI-структура → 1.0
      2. Автор–год → 0.7
      3. Квадратные номера → 0.5
      4. Иначе → 0.0
    """
    if _has_citation_struct(el):
        return True, 1.0
    if _text_has_author_year(text):
        return True, 0.7
    if _text_has_numeric_brackets(text):
        return True, 0.5
    return False, 0.0


# ───────────────────────────────────────────────────────────
# IMRAD mapping
# ───────────────────────────────────────────────────────────

_WORD = r"(?:^|[^a-z])"
_EOW = r"(?:$|[^a-z])"


def _clean_head_text(txt: str) -> str:
    t = (txt or "").strip()
    t = re.sub(r"^\s*(?:\d+|[IVXLCM]+)[\.)]?\s+", "", t, flags=re.I)
    return t.replace("&", "and").lower()


## Map TEI <head> text to a coarse IMRAD section hint.
def map_head_to_hint(head_text: str) -> str:
    t = _clean_head_text(head_text)
    if not t: return "OTHER"
    if re.search(rf"{_WORD}(abstract|introduction|background|aims and scope){_EOW}", t): return "INTRO"
    if (re.search(rf"{_WORD}(materials? and methods?){_EOW}", t) or
        re.search(rf"{_WORD}(methods?|methodology){_EOW}", t) or
        re.search(rf"{_WORD}(experimental(?: section)?){_EOW}", t) or
        re.search(rf"{_WORD}(patients? and methods?|subjects? and methods?){_EOW}", t) or
        re.search(rf"{_WORD}(study design){_EOW}", t) or
        re.search(rf"{_WORD}(statistical (analysis|methods?)){_EOW}", t)):
        return "METHODS"
    if (re.search(rf"{_WORD}(results? and discussion){_EOW}", t) or
        re.search(rf"{_WORD}(general discussion|discussion|conclusions?|concluding remarks|implications|limitations){_EOW}", t)):
        return "DISCUSSION"
    if re.search(rf"{_WORD}(results?|findings|outcomes){_EOW}", t): return "RESULTS"
    if re.search(rf"{_WORD}(references|bibliography|works cited){_EOW}", t): return "REFERENCES"
    return "OTHER"


def is_in_abstract(node: etree._Element) -> bool:
    el = node
    while el is not None:
        tag = etree.QName(el).localname.lower()
        if tag == "abstract":
            return True
        if tag == "div" and (el.get("type") or "").lower() in {"abstract", "summary"}:
            return True
        el = el.getparent()
    return False


# ───────────────────────────────────────────────────────────
# TEI → плоский список предложений
# ───────────────────────────────────────────────────────────

## Iterate TEI, yielding flat sentence/caption dicts with bbox and hints.
def tei_iter_sentences(tei_xml: str):
    """
    Однопроходный итератор по TEI:
      - держит current_imrad_section, обновляя его на <head>;
      - эмитит записи для <s>, <figDesc> (Figure), <table><head> (Table).
    """
    root = etree.fromstring(tei_xml.encode("utf-8")) if isinstance(tei_xml, str) else tei_xml
    txt = lambda el: "".join(el.itertext()).strip()

    current_imrad_section = "OTHER"

    for el in root.iter():
        tag = etree.QName(el).localname

        if tag == "head":
            head_txt = txt(el)
            label = map_head_to_hint(head_txt)
            if label != "OTHER" and not _should_ignore_head(el, label):
                current_imrad_section = label

        elif tag == "s":
            text = normalize_inline(txt(el))
            if not text:
                continue
            boxes = parse_coords_attr(el.get("coords") or "")
            page, bbox = union_bbox(boxes)
            page0 = (page - 1) if page is not None else None
            has_cit, cit_strength = compute_citation_flags(el, text)
            section_hint = "ABSTRACT" if is_in_abstract(el) else current_imrad_section
            yield {
                "text": text,
                "page": (page0 if page0 is not None else 0),
                "bbox": bbox,
                "section_hint": section_hint,
                "is_caption": False,
                "caption_type": "",
                "has_citation": has_cit,
                "citation_strength": round(cit_strength, 2)
            }

        elif tag == "figDesc":
            text = normalize_inline(txt(el))
            if not text:
                continue
            boxes = caption_boxes_with_fallback(el, "figure")
            page, bbox = union_bbox(boxes)
            page0 = (page - 1) if page is not None else None
            has_cit, cit_strength = compute_citation_flags(el, text)
            yield {
                "text": text,
                "page": (page0 if page0 is not None else 0),
                "bbox": bbox,
                "section_hint": current_imrad_section,
                "is_caption": True,
                "caption_type": "Figure",
                "has_citation": has_cit,
                "citation_strength": round(cit_strength, 2)
            }

        elif tag == "table":
            thead = el.find("./{http://www.tei-c.org/ns/1.0}head")
            if thead is None:
                continue
            text = normalize_inline(txt(thead))
            if not text:
                continue
            # объединяем head.coords и table.coords (если оба есть)
            head_boxes = parse_coords_attr(thead.get("coords") or "")
            table_boxes = parse_coords_attr(el.get("coords") or "")
            boxes = head_boxes + table_boxes if (head_boxes and table_boxes) else (head_boxes or table_boxes)
            page, bbox = union_bbox(boxes)
            page0 = (page - 1) if page is not None else None
            has_cit, cit_strength = compute_citation_flags(thead, text)
            yield {
                "text": text,
                "page": (page0 if page0 is not None else 0),
                "bbox": bbox,
                "section_hint": current_imrad_section,
                "is_caption": True,
                "caption_type": "Table",
                "has_citation": has_cit,
                "citation_strength": round(cit_strength, 2)
            }


# ───────────────────────────────────────────────────────────
# Метаданные
# ───────────────────────────────────────────────────────────

# Extract basic metadata (title, authors, date, arXiv id) from TEI root.
def extract_metadata(root) -> Dict:
    NS = {"t": root.nsmap.get(None) or "http://www.tei-c.org/ns/1.0"}
    txt = lambda el: "".join(el.itertext()).strip()

    def first(xpath):
        el = root.find(xpath, namespaces=NS)
        return txt(el) if el is not None else ""

    title = first(".//t:teiHeader/t:fileDesc/t:titleStmt/t:title")

    # authors
    authors = []
    for p in root.findall(".//t:teiHeader/t:fileDesc/t:titleStmt//t:author", namespaces=NS):
        pers = p.find(".//t:persName", namespaces=NS)
        if pers is not None:
            a = txt(pers) or txt(p)
        else:
            a = txt(p)
        if a:
            authors.append(a)
    author = ", ".join(authors) if authors else ""

    # dates
    date = first(".//t:teiHeader/t:fileDesc/t:publicationStmt//t:date") or first(".//t:teiHeader//t:date")

    # arXiv id
    arxiv = ""
    for el in root.findall(".//t:idno", namespaces=NS):
        val = txt(el)
        if "arXiv" in (el.get("type", "") + val):
            arxiv = val.strip();
            break
    if not arxiv:
        for el in root.findall(".//t:ptr|.//t:ref", namespaces=NS):
            href = el.get("target") or el.get("{http://www.w3.org/1999/xlink}href")
            if href and "arxiv.org" in href:
                m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", href)
                if m:
                    arxiv = m.group(0);
                    break

    return {
        "title": title,
        "author": author,
        "producer": "",
        "creationDate": date,
        "arxiv_id": arxiv
    }


# ───────────────────────────────────────────────────────────
# TEI → s0 (тонкий)
# ───────────────────────────────────────────────────────────

# Convert TEI to the s0.json structure for downstream stages.
def tei_to_s0(tei_xml: str, pdf_path: str) -> Dict:
    root = etree.fromstring(tei_xml.encode("utf-8")) if isinstance(tei_xml, str) else tei_xml

    # 1) собрать плоский список предложений/капшенов с секционными подсказками
    items = list(tei_iter_sentences(root))  # потоковый генератор, см. выше  :contentReference[oaicite:2]{index=2}

    # 2) посчитать page_count (максимальный встретившийся page + 1)
    max_page = 0
    for it in items:
        if isinstance(it.get("page"), int):
            max_page = max(max_page, it["page"] + 1)

    # 3) метаданные и идентификаторы
    meta = extract_metadata(root)
    pdf = Path(pdf_path)
    doc_id = meta.get("arxiv_id") or re.sub(r"[^a-zA-Z0-9._-]+", "-", pdf.stem)

    return {
        "doc_id": doc_id,
        "source_pdf": str(pdf),
        "page_count": int(max_page),
        "metadata": meta,
        "sentences": items
    }


# ───────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────

# CLI entrypoint to run S0 for a single PDF and write s0.json.
def main():
    ap = argparse.ArgumentParser(description="S0 via GROBID → s0.json (flat sentences)")
    ap.add_argument("--pdf", required=True, help="path to PDF")
    ap.add_argument("--server", default=os.getenv("GROBID_URL", "http://localhost:8070"))
    ap.add_argument("--out", default="s0.json")
    args = ap.parse_args()

    tei = grobid_fulltext_tei(args.server, args.pdf)
    s0 = tei_to_s0(tei, args.pdf)  # формируем тонкий формат, как договорились

    out = Path(args.out)
    out.write_text(json.dumps(s0, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S0] wrote {out} (doc_id={s0['doc_id']}, sentences={len(s0['sentences'])}, pages~{s0['page_count']})")


if __name__ == "__main__":
    main()

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
import json
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


# Helper: infer 0-based page index from boxes.
def page0_from_boxes(boxes: List[Dict]) -> Optional[int]:
    p, _ = union_bbox(boxes)
    return (p - 1) if p is not None else None


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
    if not t:
        return "OTHER"
    if re.search(rf"{_WORD}(abstract|aims and scope|background|introduction){_EOW}", t):
        return "INTRO"
    if (re.search(rf"{_WORD}(materials? and methods?){_EOW}", t) or
            re.search(rf"{_WORD}(methods?|methodology){_EOW}", t) or
            re.search(rf"{_WORD}(experimental(?: section)?){_EOW}", t) or
            re.search(rf"{_WORD}(patients? and methods?|subjects? and methods?){_EOW}", t) or
            re.search(rf"{_WORD}(study design){_EOW}", t) or
            re.search(rf"{_WORD}(statistical (analysis|methods?)){_EOW}", t)):
        return "METHODS"
    if (re.search(rf"{_WORD}(results? and discussion){_EOW}", t) or
            re.search(
                rf"{_WORD}(general discussion|discussion|conclusions?|concluding remarks|implications|limitations){_EOW}",
                t)):
        return "DISCUSSION"
    if re.search(rf"{_WORD}(results?|findings|outcomes){_EOW}", t):
        return "RESULTS"
    if re.search(rf"{_WORD}(references|bibliography|works cited){_EOW}", t):
        return "REFERENCES"
    return "OTHER"


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
    NS = {"t": root.nsmap.get(None) or "http://www.tei-c.org/ns/1.0"}
    txt = lambda el: "".join(el.itertext()).strip()

    current_imrad_section = "OTHER"

    for el in root.iter():
        tag = etree.QName(el).localname

        if tag == "head":
            head_txt = txt(el)
            label = map_head_to_hint(head_txt)
            # Если подзаголовок даёт OTHER — не меняем основную IMRAD-секцию
            if label != "OTHER":
                current_imrad_section = label

        elif tag == "s":
            text = normalize_inline(txt(el))
            if not text:
                continue
            boxes = parse_coords_attr(el.get("coords") or "")
            page, bbox = union_bbox(boxes)
            page0 = (page - 1) if page is not None else None
            yield {
                "text": text,
                "page": (page0 if page0 is not None else 0),
                "bbox": bbox,
                "section_hint": current_imrad_section,
                "is_caption": False,
                "caption_type": ""
            }

        elif tag == "figDesc":
            text = normalize_inline(txt(el))
            if not text:
                continue
            boxes = parse_coords_attr(el.get("coords") or "")
            if not boxes:
                # иногда coords на <figure>
                fig = el.getparent() if el.getparent() is not None and etree.QName(
                    el.getparent()).localname == "figure" else None
                if fig is not None:
                    boxes = parse_coords_attr(fig.get("coords") or "")
            page, bbox = union_bbox(boxes)
            page0 = (page - 1) if page is not None else None
            yield {
                "text": text,
                "page": (page0 if page0 is not None else 0),
                "bbox": bbox,
                "section_hint": current_imrad_section,
                "is_caption": True,
                "caption_type": "Figure"
            }

        elif tag == "table":
            # caption таблицы обычно в <table><head>
            thead = el.find("./{http://www.tei-c.org/ns/1.0}head")
            if thead is None:
                continue
            text = normalize_inline(txt(thead))
            if not text:
                continue
            boxes = parse_coords_attr(thead.get("coords") or "") or parse_coords_attr(el.get("coords") or "")
            page, bbox = union_bbox(boxes)
            page0 = (page - 1) if page is not None else None
            yield {
                "text": text,
                "page": (page0 if page0 is not None else 0),
                "bbox": bbox,
                "section_hint": current_imrad_section,
                "is_caption": True,
                "caption_type": "Table"
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

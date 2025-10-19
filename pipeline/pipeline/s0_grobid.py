#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S0 (GROBID) → s0.json (единый контракт для S1/S2/фронта)

Функции:
- grobid_fulltext_tei(server, pdf_path) -> str (TEI XML)
- tei_to_s0(tei_xml, pdf_path) -> dict (s0 payload)

Особенности:
- Секции нормализуются к IMRaD-названиям: Abstract / Introduction /
  Materials and methods / Results and discussion / Conclusions / Body / References
- Дополнительно к прежнему контракту:
  sections[].blocks[].sentences[] с координатами, индексами и span'ами,
  sections[].head_raw и sections[].imrad_hint
- Капшены: Figure{n} (арабские), Table{Roman(n)} (римские), лёгкая нормализация пробелов
- page_count берётся как максимум номера страницы из coords (если есть)
- doc_id: arXiv id (если найден) иначе slug из имени файла
"""

import re
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests
from lxml import etree


# ───────────────────────────────────────────────────────────
# GROBID HTTP
# ───────────────────────────────────────────────────────────

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
# Вспомогательные парсеры координат и нормализация
# ───────────────────────────────────────────────────────────

COORD_ITEM_RE = re.compile(
    r"^\s*(?:p)?(?P<page>\d+)\s*[: ,]\s*(?P<x>[-\d.]+)\s*,\s*(?P<y>[-\d.]+)\s*,\s*(?P<w>[-\d.]+)\s*,\s*(?P<h>[-\d.]+)"
)


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


def union_bbox(boxes: List[Dict]) -> Tuple[Optional[int], List[float]]:
    if not boxes:
        return None, [0, 0, 0, 0]
    page = boxes[0]["page"]
    xs0 = [b["x"] for b in boxes if b["page"] == page]
    ys0 = [b["y"] for b in boxes if b["page"] == page]
    xs1 = [b["x"] + b["w"] for b in boxes if b["page"] == page]
    ys1 = [b["y"] + b["h"] for b in boxes if b["page"] == page]
    return page, [min(xs0), min(ys0), max(xs1), max(ys1)]


def page0_from_boxes(boxes: List[Dict]) -> Optional[int]:
    p, _ = union_bbox(boxes)
    return (p - 1) if p is not None else None


_WS_MULTI = re.compile(r"[ \t\u00A0]+")
_NO_SPACE_BEFORE = re.compile(r"\s+([),.;:\]\}])")
_NO_SPACE_AFTER = re.compile(r"([(\[\{])\s+")
_JOIN_FIGTAB = re.compile(r"(\S)(?=(Figure|Table)\b)")
_FIX_PERCENT = re.compile(r"(%)([A-Za-z])")  # "60%of" → "60% of"


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


IMRAD_TO_SECTION_NAME = {
    "INTRO": "Introduction",
    "METHODS": "Materials and methods",
    "RESULTS": "Results and discussion",
    "DISCUSSION": "Conclusions",
    "REFERENCES": "References",
    "OTHER": "Body",
}


def roman(n: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s);
            n -= v
    return "".join(out)


# ───────────────────────────────────────────────────────────
# TEI → s0.json
# ───────────────────────────────────────────────────────────

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


def tei_to_s0(tei_xml: str, pdf_path: str) -> Dict:
    """
    Конвертирует TEI (строка) → dict s0-пэйлоад с детальной структурой (blocks/sentences),
    сохраняя обратную совместимость (sections[].text, captions[].text).
    """
    root = etree.fromstring(tei_xml.encode("utf-8")) if isinstance(tei_xml, str) else tei_xml
    NS = {"t": root.nsmap.get(None) or "http://www.tei-c.org/ns/1.0"}
    txt = lambda el: "".join(el.itertext()).strip()

    meta = extract_metadata(root)

    # 1) Секции: идём по <div>, берём <head> → IMRAD, собираем блоки <p> и внутри предложения <s>
    sections_raw: List[Dict] = []
    max_page = 0

    for div in root.findall(".//t:div", namespaces=NS):
        head = div.find("./t:head", namespaces=NS)
        head_text_raw = txt(head) if head is not None else ""
        imrad = map_head_to_hint(head_text_raw)
        sec_name = IMRAD_TO_SECTION_NAME.get(imrad, "Body")

        blocks = []
        # Для подсчёта span_in_section нам понадобится общий текст секции и смещения
        section_text_chunks: List[str] = []
        offset_in_section = 0

        for p_idx, p in enumerate(div.findall(".//t:p", namespaces=NS)):
            # Абзац: собираем предложения (если TEI дал <s>)
            sents = p.findall(".//t:s", namespaces=NS)

            block_sentences = []
            block_text_chunks: List[str] = []
            # coords абзаца
            p_coords = parse_coords_attr(p.get("coords") or "")
            p_page0 = page0_from_boxes(p_coords)

            if sents:
                for s_idx, s in enumerate(sents):
                    s_text = normalize_inline(txt(s))
                    if not s_text:
                        continue
                    s_coords = parse_coords_attr(s.get("coords") or "")
                    s_page0 = page0_from_boxes(s_coords) if s_coords else p_page0

                    # span внутри блока
                    block_local_start = sum(len(t) + 1 for t in block_text_chunks)  # +1 за пробел между предложениями
                    span_in_block = [block_local_start, block_local_start + len(s_text)]

                    # span в секции: учитываем то, что между абзацами будет "\n\n"
                    section_local_start = offset_in_section + (sum(len(t) + 1 for t in block_text_chunks))
                    span_in_section = [section_local_start, section_local_start + len(s_text)]

                    block_text_chunks.append(s_text)
                    block_sentences.append({
                        "s_idx": s_idx,
                        "text": s_text,
                        "span_in_block": span_in_block,
                        "span_in_section": span_in_section,
                        "page": (s_page0 if s_page0 is not None else (p_page0 if p_page0 is not None else 0)),
                        "coords": s_coords or p_coords
                    })
                    # обновим максимум страниц
                    for b in (s_coords or p_coords or []):
                        if b["page"] > max_page:
                            max_page = b["page"]
            else:
                # TEI не дал <s> — трактуем весь <p> как один sentence
                p_text = normalize_inline(txt(p))
                if p_text:
                    s_coords = p_coords
                    s_page0 = p_page0
                    span_in_block = [0, len(p_text)]
                    span_in_section = [offset_in_section, offset_in_section + len(p_text)]
                    block_text_chunks.append(p_text)
                    block_sentences.append({
                        "s_idx": 0,
                        "text": p_text,
                        "span_in_block": span_in_block,
                        "span_in_section": span_in_section,
                        "page": (s_page0 if s_page0 is not None else 0),
                        "coords": s_coords
                    })
                    for b in (s_coords or []):
                        if b["page"] > max_page:
                            max_page = b["page"]

            # Собираем block.text и продвигаем offset_in_section
            block_text = " ".join(block_text_chunks).strip()
            if block_text:
                blocks.append({
                    "type": "paragraph",
                    "p_idx": p_idx,
                    "page": (p_page0 if p_page0 is not None else 0),
                    "coords": p_coords,
                    "text": block_text,
                    "sentences": block_sentences
                })
                # В секции между абзацами будет разделитель "\n\n"
                section_text_chunks.append(block_text)
                offset_in_section += len(block_text) + 2  # +2 за "\n\n"

        # Секция собрана
        section_text = "\n\n".join(section_text_chunks).strip()
        if section_text or blocks:
            sections_raw.append({
                "name": sec_name,
                "head_raw": head_text_raw,
                "imrad_hint": imrad,
                "text": section_text,  # бэкомпат + предпросмотр
                "blocks": blocks
            })

    # 2) Капшены: figure/table
    captions = []
    fig_idx, tbl_idx = 0, 0

    for fig in root.findall(".//t:figure", namespaces=NS):
        desc = fig.find("./t:figDesc", namespaces=NS)
        if desc is None:
            continue
        text = normalize_inline(txt(desc))
        if not text:
            continue
        fig_idx += 1
        boxes = parse_coords_attr(desc.get("coords") or "") or parse_coords_attr(fig.get("coords") or "")
        page0 = page0_from_boxes(boxes)
        if page0 is not None:
            max_page = max(max_page, page0 + 1)
        captions.append({
            "id": f"Figure{fig_idx}",
            "text": text,
            "page": (page0 if page0 is not None else 0),
            "coords": boxes
        })

    for tbl in root.findall(".//t:table", namespaces=NS):
        thead = tbl.find("./t:head", namespaces=NS)
        if thead is None:
            continue
        text = normalize_inline(txt(thead))
        if not text:
            continue
        tbl_idx += 1
        boxes = parse_coords_attr(thead.get("coords") or "") or parse_coords_attr(tbl.get("coords") or "")
        page0 = page0_from_boxes(boxes)
        if page0 is not None:
            max_page = max(max_page, page0 + 1)
        captions.append({
            "id": f"Table{roman(tbl_idx)}",
            "text": text,
            "page": (page0 if page0 is not None else 0),
            "coords": boxes
        })

    # 3) Финальный список секций в стабильном порядке (пустые не включаем)
    ORDERED = [
        "Abstract", "Introduction", "Materials and methods",
        "Results and discussion", "Conclusions", "Body", "References"
    ]
    sections: List[Dict] = []
    # сгруппируем по имени (на случай нескольких <div> одной IMRAD-группы)
    grouped: Dict[str, List[Dict]] = {}
    for sec in sections_raw:
        grouped.setdefault(sec["name"], []).append(sec)

    for name in ORDERED:
        if name not in grouped:
            continue
        # Сшиваем несколько кусков одной секции (если были) — blocks и text
        merged_blocks: List[Dict] = []
        merged_text_chunks: List[str] = []
        head_raw_all: List[str] = []
        imrad_hint = None
        # Пересчёт span_in_section: уже учтён при построении каждой части; при склейке добавим сдвиг.
        # Но чтобы не усложнять, мы уже склеивали внутри каждого <div>; обычно IMRAD секции идут одной кучей.
        # Здесь просто конкатенируем.
        for part in grouped[name]:
            head_raw_all.append(part.get("head_raw", ""))
            imrad_hint = imrad_hint or part.get("imrad_hint")
            # blocks
            base_blocks = part.get("blocks", [])
            # Никакого пересчёта индексов внутри блоков не требуется — они локальны внутри своей "части".
            merged_blocks.extend(base_blocks)
            # text
            t = part.get("text", "")
            if t:
                merged_text_chunks.append(t)
        full_text = "\n\n".join(merged_text_chunks).strip()
        sections.append({
            "name": name,
            "head_raw": "; ".join([h for h in head_raw_all if h]),
            "imrad_hint": imrad_hint or "OTHER",
            "text": full_text,
            "blocks": merged_blocks
        })

    # # 4) Доп. индекс предложений (плоский) — опционально, чтобы S1 мог итерироваться линейно
    # index_sentences = []
    # for sec_idx, sec in enumerate(sections):
    #     for blk_idx, blk in enumerate(sec.get("blocks", [])):
    #         for sent in blk.get("sentences", []):
    #             index_sentences.append({"sec": sec_idx, "p": blk_idx, "s": int(sent.get("s_idx", 0))})

    # 5) doc_id, page_count, source_pdf
    pdf = Path(pdf_path)
    doc_id = meta.get("arxiv_id") or re.sub(r"[^a-zA-Z0-9._-]+", "-", pdf.stem)
    page_count = int(max_page) if max_page else 0

    return {
        "doc_id": doc_id,
        "source_pdf": str(pdf),
        "page_count": page_count,
        "metadata": meta,
        "sections": sections,
        "captions": captions,
        # "index": {"sentences": index_sentences}
    }


# ───────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="S0 via GROBID → s0.json (fine-grained)")
    ap.add_argument("--pdf", required=True, help="path to PDF")
    ap.add_argument("--server", default=os.getenv("GROBID_URL", "http://localhost:8070"))
    ap.add_argument("--out", default="s0.json")
    args = ap.parse_args()

    tei = grobid_fulltext_tei(args.server, args.pdf)
    s0 = tei_to_s0(tei, args.pdf)

    out = Path(args.out)
    out.write_text(json.dumps(s0, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S0] wrote {out} "
          f"(doc_id={s0['doc_id']}, sections={len(s0['sections'])}, "
          f"captions={len(s0['captions'])}, pages~{s0['page_count']}, "
          f"sentences={len(s0.get('index', {}).get('sentences', []))})")


if __name__ == "__main__":
    main()

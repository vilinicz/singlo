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
- Капшены: Figure{n} (арабские), Table{Roman(n)} (римские)
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
# Вспомогательные парсеры координат
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
        x = float(m.group("x")); y = float(m.group("y"))
        w = float(m.group("w")); h = float(m.group("h"))
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
                if m: arxiv = m.group(0); break

    return {
        "title": title,
        "author": author,
        "producer": "",
        "creationDate": date,
        "arxiv_id": arxiv
    }


def tei_to_s0(tei_xml: str, pdf_path: str) -> Dict:
    """
    Конвертирует TEI (строка) → dict s0-пэйлоад по нашему контракту.
    """
    root = etree.fromstring(tei_xml.encode("utf-8")) if isinstance(tei_xml, str) else tei_xml
    NS = {"t": root.nsmap.get(None) or "http://www.tei-c.org/ns/1.0"}
    txt = lambda el: "".join(el.itertext()).strip()

    meta = extract_metadata(root)

    # 1) Секции: идём по <div>, берём <head> → IMRAD, собираем текст из <p>/<s>
    sections_map: Dict[str, List[str]] = {}
    max_page = 0

    for div in root.findall(".//t:div", namespaces=NS):
        head = div.find("./t:head", namespaces=NS)
        head_text = txt(head) if head is not None else ""
        imrad = map_head_to_hint(head_text)
        sec_name = IMRAD_TO_SECTION_NAME.get(imrad, "Body")

        buf = []
        for p in div.findall(".//t:p", namespaces=NS):
            sents = p.findall(".//t:s", namespaces=NS)
            chunk = " ".join(txt(s) for s in sents if txt(s)) if sents else txt(p)
            if chunk:
                buf.append(chunk)

            coords = p.get("coords") or ""
            boxes = parse_coords_attr(coords)
            for b in boxes:
                if b["page"] > max_page:
                    max_page = b["page"]

        body = "\n\n".join(buf).strip()
        if body:
            sections_map.setdefault(sec_name, []).append(body)

    # 2) Капшены: figure/table
    captions = []
    fig_idx, tbl_idx = 0, 0

    for fig in root.findall(".//t:figure", namespaces=NS):
        desc = fig.find("./t:figDesc", namespaces=NS)
        if desc is None:
            continue
        text = txt(desc)
        if not text:
            continue
        fig_idx += 1
        boxes = parse_coords_attr(desc.get("coords") or "") or parse_coords_attr(fig.get("coords") or "")
        page0 = page0_from_boxes(boxes)
        max_page = max(max_page, (page0 + 1) if page0 is not None else 0)
        captions.append({"id": f"Figure{fig_idx}", "text": text, "page": (page0 if page0 is not None else 0)})

    for tbl in root.findall(".//t:table", namespaces=NS):
        thead = tbl.find("./t:head", namespaces=NS)
        if thead is None:
            continue
        text = txt(thead)
        if not text:
            continue
        tbl_idx += 1
        boxes = parse_coords_attr(thead.get("coords") or "") or parse_coords_attr(tbl.get("coords") or "")
        page0 = page0_from_boxes(boxes)
        max_page = max(max_page, (page0 + 1) if page0 is not None else 0)
        captions.append({"id": f"Table{roman(tbl_idx)}", "text": text, "page": (page0 if page0 is not None else 0)})

    # 3) Финальные секции в стабильном порядке, пустые не включаем
    ordered = [
        "Abstract", "Introduction", "Materials and methods",
        "Results and discussion", "Conclusions", "Body", "References"
    ]
    sections = []
    for name in ordered:
        chunks = sections_map.get(name, [])
        if chunks:
            sections.append({"name": name, "text": "\n\n".join(chunks)})

    # 4) doc_id, page_count, source_pdf
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
    }


# ───────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="S0 via GROBID → s0.json")
    ap.add_argument("--pdf", required=True, help="path to PDF")
    ap.add_argument("--server", default=os.getenv("GROBID_URL", "http://localhost:8070"))
    ap.add_argument("--out", default="s0.json")
    args = ap.parse_args()

    tei = grobid_fulltext_tei(args.server, args.pdf)
    s0 = tei_to_s0(tei, args.pdf)

    out = Path(args.out)
    out.write_text(json.dumps(s0, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[S0] wrote {out} (doc_id={s0['doc_id']}, sections={len(s0['sections'])}, captions={len(s0['captions'])}, pages~{s0['page_count']})")


if __name__ == "__main__":
    main()

"""Lightweight PDF parser helpers for building S0-like context from PDFs.

Uses PyMuPDF to extract text blocks, infer section headings, and detect
caption candidates. Produces structures that can be merged with TEI-based
data when GROBID is unavailable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pymupdf

from .utils import normalize_text

SECTION_PAT = re.compile(
    r"""(?ix) ^
        (?: \d+\.?\s+ | [ivxl]+\.\s+ )?
        (abstract|introduction|background|
         materials\s+and\s+methods|materials\s*&\s*methods|methods|method|
         results?\s+and\s+discussion|results?|discussion|
         conclusions?|conclusion|
         references|acknowledg(e)?ments|related\s+work|limitations)
        \s* $
    """
)
ALLCAPS_SECTION_PAT = re.compile(r"^[A-Z][A-Z\s\-]{3,}$")

CAPTION_PAT = re.compile(
    r"""
    ^\s*
    (?P<label>fig(?:\.|ure)?|table|tab\.)
    \s*
    (?P<num>[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*)?
    \s*
    (?P<punct>[:.\-–—])
    \s*
    (?P<body>.*)
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

INLINE_HEADING_RE = re.compile(
    r'^\s*(abstract|introduction|conclusion|conclusions|discussion|results|'
    r'materials\s+and\s+methods|materials\s*&\s*methods|results?\s+and\s+discussion)'
    r'\s*[-:—–]\s*(.+)$',
    re.I
)


## Heuristic: treat short, capitalized lines without trailing punctuation as headings.
def _looks_like_heading(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if len(stripped) > 80:
        return False
    if re.search(r'[.!?;:,]$', stripped):
        return False
    if re.match(r'^[A-Z]\.$', stripped):
        return True
    words = stripped.split()
    if not words:
        return False
    if len(words) <= 6 and all(w.isupper() or (w[0].isupper() and w[1:].islower()) for w in words):
        return True
    return False


@dataclass
class CaptionData:
    id: str
    kind: str
    page: int
    text: str
    bbox: Optional[List[float]] = None
    image_bbox: Optional[List[float]] = None


def extract_title_and_authors(doc: pymupdf.Document) -> tuple[Optional[str], Optional[str]]:
    """Heuristic detection of title/authors from the first page."""
    if len(doc) == 0:
        return None, None
    page = doc[0]
    data = page.get_text("dict") or {}
    spans = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                spans.append({
                    "text": text,
                    "size": float(span.get("size", 0.0)),
                    "y": float(line.get("bbox", [0, 0, 0, 0])[1]),
                })
    if not spans:
        return None, None

    max_size = max(span["size"] for span in spans)
    title_lines = [s for s in spans if s["size"] >= max_size - 0.5 and s["y"] < spans[0]["y"] + 300]
    title = " ".join(s["text"] for s in title_lines).strip()

    below_title = [s for s in spans if s["y"] > (title_lines[0]["y"] if title_lines else 0)]
    sizes = sorted({s["size"] for s in below_title}, reverse=True)
    second = sizes[0] if sizes else 0
    author_lines = [s for s in below_title if abs(s["size"] - second) < 0.6]
    authors = " ".join(s["text"] for s in author_lines).strip()
    return (title or None), (authors or None)


## Detect "Figure/Table" caption markers and return (kind, number, tail).
def _match_caption(line: str) -> Optional[tuple[str, str, str]]:
    m = CAPTION_PAT.match(line or "")
    if not m:
        return None
    label = m.group("label").lower()
    kind = "Figure" if label.startswith("fig") else "Table"
    num = (m.group("num") or "").strip()
    tail = (m.group("body") or "").strip()
    return kind, num, tail


## Return a small preview of text blocks for debugging.
def _extract_blocks_sample(page: pymupdf.Page, limit: int = 5) -> List[Dict[str, Any]]:
    blocks = []
    for b in page.get_text("blocks")[:limit]:
        x0, y0, x1, y1, text, *_ = b
        if not (text or "").strip():
            continue
        blocks.append({
            "bbox": [round(float(x0), 1), round(float(y0), 1), round(float(x1), 1), round(float(y1), 1)],
            "text_preview": text.strip().replace("\n", " ")[:160]
        })
    return blocks


## Split inline heading patterns like "Results - text..." into (heading, remainder).
def _split_inline_heading(text: str) -> Optional[tuple[str, str]]:
    m = INLINE_HEADING_RE.match(text or "")
    if not m:
        return None
    title = (m.group(1) or "").strip().title()
    remainder = (m.group(2) or "").strip()
    return title, remainder


## Pick nearest image block bbox to a caption bbox, within a distance threshold.
def _assign_nearest_image_bbox(caption_bbox: List[float],
                               image_blocks: List[Dict[str, Any]]) -> Optional[List[float]]:
    if not caption_bbox or not image_blocks:
        return None
    best_block: Optional[Dict[str, Any]] = None
    best_distance = float("inf")
    cap_top = caption_bbox[1]
    cap_left = caption_bbox[0]
    cap_right = caption_bbox[2]

    for block in image_blocks:
        if block.get("used"):
            continue
        bbox = block.get("bbox")
        if not bbox:
            continue
        bottom = bbox[3]
        if bottom > cap_top + 10:
            continue
        horiz_overlap = not (bbox[2] < cap_left or bbox[0] > cap_right)
        vertical_distance = max(0.0, cap_top - bottom)
        if not horiz_overlap and vertical_distance > 200:
            continue
        if vertical_distance < best_distance:
            best_distance = vertical_distance
            best_block = block

    if best_block:
        best_block["used"] = True
        bbox = best_block.get("bbox")
        if bbox:
            return [round(float(x), 2) for x in bbox]
    return None


## Extract basic per-page structure: lines and image blocks.
def _extract_pdf_page_struct(page: pymupdf.Page, page_no: int) -> Dict[str, Any]:
    raw_blocks = page.get_text("blocks") or []
    lines: List[Dict[str, Any]] = []
    image_blocks: List[Dict[str, Any]] = []

    for block_idx, block in enumerate(raw_blocks):
        if not block or len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        block_type = int(block[6]) if len(block) > 6 else 0
        bbox = [round(float(x0), 2), round(float(y0), 2), round(float(x1), 2), round(float(y1), 2)]

        if block_type == 1:
            image_blocks.append({
                "block_index": block_idx,
                "bbox": bbox,
                "used": False,
                "page": page_no
            })
            continue

        if not (text or "").strip():
            continue

        for line in (text or "").splitlines():
            line_text = line.strip()
            if not line_text:
                continue
            lines.append({
                "text": line_text,
                "bbox": bbox,
                "block_index": block_idx,
                "page": page_no,
                "is_heading": _looks_like_heading(line_text)
            })

    return {
        "text_blocks": raw_blocks,
        "lines": lines,
        "image_blocks": image_blocks
    }


## Recognize common section headings (incl. ALLCAPS variants).
def _is_section_heading(line: str) -> Optional[str]:
    raw = (line or "").strip()
    if not raw:
        return None
    m = SECTION_PAT.match(raw)
    if m:
        return m.group(1).capitalize()
    if ALLCAPS_SECTION_PAT.match(raw) and len(raw.split()) <= 5:
        word = raw.lower()
        for candidate in [
            "abstract", "introduction", "methods", "materials and methods",
            "results", "discussion", "conclusions", "conclusion",
            "supplementary", "appendix", "acknowledgements", "acknowledgments"
        ]:
            if candidate.replace(" ", "") in word.replace(" ", ""):
                return candidate.title()
    return None


## Group lines into sections by headings; join adjacent text into chunks.
def _build_pdf_sections(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []

    def finalize(sec: Optional[Dict[str, Any]]):
        if not sec or not sec.get("chunks"):
            return
        chunks = [ch for ch in sec["chunks"] if ch.get("text")]
        if not chunks:
            return

        text_parts: List[str] = []
        page_numbers: List[int] = []

        for chunk in chunks:
            chunk_text = chunk.get("text", "").strip()
            if not chunk_text:
                continue
            if text_parts:
                text_parts.append(" ")
            text_parts.append(chunk_text)
            page = chunk.get("page")
            if page is not None:
                page_numbers.append(page)

        text = "".join(text_parts).strip()
        if not text:
            return

        sections.append({
            "name": sec.get("name") or "Body",
            "text": text,
            "page_start": min(page_numbers) if page_numbers else None,
            "page_end": max(page_numbers) if page_numbers else None,
        })

    current: Optional[Dict[str, Any]] = None

    for entry in lines:
        text = entry.get("text", "")
        if not text:
            continue
        heading = _is_section_heading(text)
        if heading:
            finalize(current)
            current = {"name": heading, "chunks": []}
            continue

        inline = _split_inline_heading(text)
        if inline:
            heading_name, remainder = inline
            finalize(current)
            current = {"name": heading_name, "chunks": []}
            if remainder:
                current["chunks"].append({**entry, "text": remainder})
            continue

        if entry.get("is_heading"):
            continue

        if current is None:
            current = {"name": "FrontMatter", "chunks": []}
        current.setdefault("chunks", []).append(entry)

    finalize(current)

    if not sections:
        text_parts: List[str] = []
        page_numbers: List[int] = []
        for entry in lines:
            chunk_text = entry.get("text", "").strip()
            if not chunk_text:
                continue
            if text_parts:
                text_parts.append(" ")
            text_parts.append(chunk_text)
            page = entry.get("page")
            if page is not None:
                page_numbers.append(page)
        combined_text = "".join(text_parts).strip()
        if combined_text:
            sections.append({
                "name": "Body",
                "text": combined_text,
                "page_start": min(page_numbers) if page_numbers else None,
                "page_end": max(page_numbers) if page_numbers else None,
            })
    return sections


## Scan text/image blocks to collect figure/table captions with approximate bboxes.
def _collect_captions_from_blocks(blocks: List[Any],
                                  page_no: int,
                                  image_blocks: Optional[List[Dict[str, Any]]] = None) -> List[CaptionData]:
    caps: List[CaptionData] = []
    image_blocks = image_blocks or []

    for block in blocks:
        if not block or len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        block_type = int(block[6]) if len(block) > 6 else 0
        if block_type != 0:
            continue
        if not (text or "").strip():
            continue
        bbox = [round(float(x0), 2), round(float(y0), 2), round(float(x1), 2), round(float(y1), 2)]
        raw_lines = []
        for line in (text or "").splitlines():
            line_text = line.strip()
            if not line_text:
                continue
            raw_lines.append({
                "text": line_text,
                "bbox": bbox
            })
        if not raw_lines:
            continue

        lines = [row["text"] for row in raw_lines]
        i = 0
        while i < len(lines):
            line = lines[i]
            cap_head = _match_caption(line)
            if not cap_head:
                i += 1
                continue

            kind, num, tail = cap_head
            cap_id = f"{kind}{num}" if num else kind
            collected_lines: List[str] = []
            used_indices: List[int] = []
            if tail:
                collected_lines.append(tail)
                used_indices.append(i)

            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt:
                    break
                if _match_caption(nxt) or _is_section_heading(nxt):
                    break
                collected_lines.append(nxt)
                used_indices.append(j)
                j += 1

            caption_text = " ".join(collected_lines).strip()
            if not caption_text:
                i = j
                continue

            indices = used_indices or [i]
            xs0 = min(raw_lines[idx]["bbox"][0] for idx in indices)
            ys0 = min(raw_lines[idx]["bbox"][1] for idx in indices)
            xs1 = max(raw_lines[idx]["bbox"][2] for idx in indices)
            ys1 = max(raw_lines[idx]["bbox"][3] for idx in indices)
            caption_bbox = [xs0, ys0, xs1, ys1]
            image_bbox = _assign_nearest_image_bbox(caption_bbox, image_blocks)

            caps.append(CaptionData(
                kind=kind,
                id=cap_id,
                page=page_no,
                text=caption_text,
                bbox=caption_bbox,
                image_bbox=image_bbox
            ))
            i = j
    return caps


## Parse a PyMuPDF Document into sections/captions/figures/tables/pages sample.
def parse_pdf_document(doc: pymupdf.Document) -> Dict[str, Any]:
    page_count = len(doc)
    pages_sample: List[Dict[str, Any]] = []
    all_lines: List[Dict[str, Any]] = []
    all_captions: List[CaptionData] = []

    for idx in range(page_count):
        page = doc[idx]
        page_no = idx + 1
        page_struct = _extract_pdf_page_struct(page, page_no)
        pages_sample.append({
            "page": page_no,
            "blocks_sample": _extract_blocks_sample(page)
        })
        all_lines.extend(page_struct["lines"])
        page_caps = _collect_captions_from_blocks(
            page_struct["text_blocks"],
            page_no,
            page_struct["image_blocks"]
        )
        all_captions.extend(page_caps)

    sections = _build_pdf_sections(all_lines)
    for idx, sec in enumerate(sections):
        sec["id"] = f"sec-{idx + 1:02d}"

    captions_list: List[Dict[str, Any]] = []
    seen_caps: set[tuple] = set()
    for cap in all_captions:
        if not cap.text:
            continue
        norm_text = normalize_text(cap.text)
        key = (cap.id, norm_text, cap.page)
        if key in seen_caps:
            continue
        seen_caps.add(key)
        entry: Dict[str, Any] = {
            "id": cap.id,
            "kind": cap.kind,
            "text": norm_text,
            "provenance": [{
                "source": "pdf",
                "page": cap.page,
                "bbox": cap.bbox
            }]
        }
        if cap.image_bbox:
            entry["content_bbox"] = cap.image_bbox
        captions_list.append(entry)

    figures = []
    tables = []
    for cap in captions_list:
        prov = cap.get("provenance") or []
        page_ref = prov[0]["page"] if prov else None
        base_entry = {
            "id": cap.get("id"),
            "page": page_ref,
            "caption": cap.get("text"),
            "caption_provenance": prov
        }
        if cap.get("content_bbox"):
            base_entry["content_bbox"] = cap["content_bbox"]
        if cap.get("kind") == "Figure":
            figures.append(base_entry)
        elif cap.get("kind") == "Table":
            tables.append(base_entry)

    return {
        "sections": sections,
        "captions": captions_list,
        "figures": figures,
        "tables": tables,
        "pages_sample": pages_sample
    }

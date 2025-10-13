# pipeline/s0.py
from __future__ import annotations
import re
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

# --------- Регексы для детектора секций/капшенов ---------

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

FIG_PAT = re.compile(r"^\s*(Figure|Fig\.)\s*([A-Za-z]?\d+[A-Za-z]?)\s*[:\-]?\s*(.*)$", re.I)
TAB_PAT = re.compile(r"^\s*(Table|Tab\.|TABLE)\s*([A-Za-z]?\d+[A-Za-z]?)?\s*[:\-]?\s*(.*)$", re.I)
TAB_ROMAN_PAT = re.compile(r"^\s*TABLE\s+([IVXLC]+)\s*[:\-]?\s*(.*)$", re.I)

# --------- DocID helpers ---------

JUNK_TITLE_PAT = re.compile(r'^(title|untitled|document)\b', re.I)
JUNK_PRODUCER_PAT = re.compile(r'(microsoft\s+word|adobe\s+pdf|libreoffice|pages)\b', re.I)
ARXIV_RE = re.compile(r'\b(\d{4}\.\d{5})(v\d+)?\b', re.I)


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    s = re.sub(r'-{2,}', '-', s).strip('-')
    return s or 'doc'


def _looks_junky(s: str) -> bool:
    if not s:
        return True
    s = s.strip()
    return bool(JUNK_TITLE_PAT.match(s) or JUNK_PRODUCER_PAT.search(s))


def _extract_arxiv_id_from_text(text: str) -> Optional[str]:
    m = ARXIV_RE.search(text or "")
    return (m.group(1) + (m.group(2) or "")) if m else None


def _guess_arxiv_id(source_pdf: str, first_page_text: str, metadata: dict) -> Optional[str]:
    arx = (metadata or {}).get("arxiv_id")
    if arx:
        return arx
    arx = _extract_arxiv_id_from_text(first_page_text or "")
    if arx:
        return arx
    arx = _extract_arxiv_id_from_text(Path(source_pdf or "").stem)
    if arx:
        return arx
    return None


def _safe_doc_id(s0: dict, s0_path: str, first_page_text: str = "") -> str:
    meta = s0.get("metadata") or {}
    src = s0.get("source_pdf") or ""
    raw_id = (s0.get("doc_id") or "").strip()

    # Если уже есть адекватный doc_id — только «причесать»
    if raw_id and not _looks_junky(raw_id):
        return _slug(raw_id)

    # 1) arXiv
    arx = _guess_arxiv_id(src, first_page_text, meta)
    if arx:
        return _slug(arx)
    # 2) имя PDF
    if src:
        return _slug(Path(src).stem)
    # 3) имя рабочей директории
    stem = Path(s0_path).parent.name
    if stem:
        return _slug(stem)
    # 4) короткий хэш
    return 'doc-' + hashlib.md5(s0_path.encode('utf-8')).hexdigest()[:8]


# --------- Модель подписи ---------

@dataclass
class Caption:
    kind: str  # Figure / Table
    id: str  # "Figure2" / "Table I"
    page: int
    text: str


# --------- Нормализация текста ---------

RE_SOFT_HYPHEN_BREAK = re.compile(r'(\w)[\-­]\n(\w)')  # advec-\ntion → advection
RE_LINE_BREAK_IN_NUMBER = re.compile(r'(\d)\s*\n\s*(\d)')  # 40.\n0 → 40.0
RE_LINE_BREAK_AFTER_PAREN = re.compile(r'\)\s*\n\s*(\d)')  # ") \n60" → ") 60"
RE_MULTI_SPACES = re.compile(r'[ \t]{2,}')


def _normalize_text(text: str) -> str:
    t = (text or "").replace('\r\n', '\n').replace('\r', '\n')
    t = RE_SOFT_HYPHEN_BREAK.sub(r'\1\2', t)
    t = RE_LINE_BREAK_IN_NUMBER.sub(r'\1.\2', t)
    t = RE_LINE_BREAK_AFTER_PAREN.sub(r') \1', t)
    # одиночные переносы внутри абзаца → пробел
    t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)
    t = RE_MULTI_SPACES.sub(' ', t)
    return t.strip()


# --------- Вытаскиваем строки/капшены со страницы ---------

def _extract_page_lines(page: fitz.Page) -> List[str]:
    return page.get_text("text").splitlines()


def _extract_blocks_sample(page: fitz.Page, limit: int = 5) -> List[Dict[str, Any]]:
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


def _is_section_heading(line: str) -> Optional[str]:
    raw = (line or "").strip()
    if not raw:
        return None
    m = SECTION_PAT.match(raw)
    if m:
        return m.group(1).capitalize()
    if ALLCAPS_SECTION_PAT.match(raw) and len(raw.split()) <= 5:
        word = raw.lower()
        for c in ["abstract", "introduction", "methods", "materials and methods",
                  "results", "discussion", "conclusions", "conclusion",
                  "supplementary", "appendix", "acknowledgements", "acknowledgments"]:
            if c.replace(" ", "") in word.replace(" ", ""):
                return c.title()
    return None


def _collect_captions(lines: List[str], page_no: int) -> List[Caption]:
    caps: List[Caption] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m_fig = FIG_PAT.match(line)
        m_tab = TAB_PAT.match(line)
        m_tab_roman = TAB_ROMAN_PAT.match(line)

        if m_tab_roman:
            num = m_tab_roman.group(1)
            tail = m_tab_roman.group(2).strip()
            buf = [tail] if tail else []
            j = i + 1
            while j < len(lines):
                nxt = (lines[j] or "").strip()
                if not nxt:
                    break
                if FIG_PAT.match(nxt) or TAB_PAT.match(nxt) or TAB_ROMAN_PAT.match(nxt) or _is_section_heading(nxt):
                    break
                buf.append(nxt);
                j += 1
            caps.append(Caption(kind="Table", id=f"Table{num}", page=page_no, text=" ".join(buf).strip()))
            i = j;
            continue

        if m_fig or m_tab:
            kind = "Figure" if m_fig else "Table"
            m = m_fig or m_tab
            num = (m.group(2) or "").strip()
            tail = (m.group(3) or "").strip()
            buf = [tail] if tail else []
            j = i + 1
            while j < len(lines):
                nxt = (lines[j] or "").strip()
                if not nxt:
                    break
                if FIG_PAT.match(nxt) or TAB_PAT.match(nxt) or _is_section_heading(nxt):
                    break
                buf.append(nxt);
                j += 1
            caps.append(Caption(kind=kind, id=f"{kind}{num}", page=page_no, text=" ".join(buf).strip()))
            i = j;
            continue

        i += 1
    return caps


# --------- Главная функция S0 ---------

def build_s0(pdf_path: str, out_dir: str) -> Dict[str, Any]:
    """
    Читает PDF и сохраняет s0.json в out_dir.
    Возвращает словарь S0.
    """
    pdf_path = str(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    # 0) эвристика заголовка/авторов с 1-й страницы
    def _extract_title_and_authors(doc_obj):
        page = doc_obj[0]
        d = page.get_text("dict")
        spans = []
        for b in d.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    spans.append({
                        "text": s.get("text", "").strip(),
                        "size": float(s.get("size", 0.0)),
                        "y": float(l.get("bbox", [0, 0, 0, 0])[1]),
                    })
        spans = [s for s in spans if s["text"]]
        if not spans:
            return None, None
        max_size = max(s["size"] for s in spans)
        title_lines = [s for s in spans if s["size"] >= max_size - 0.5 and s["y"] < spans[0]["y"] + 300]
        title = " ".join(s["text"] for s in title_lines).strip()
        # авторы — следующий по размеру кегль
        below_title = [s for s in spans if s["y"] > (title_lines[0]["y"] if title_lines else 0)]
        sizes = sorted({s["size"] for s in below_title}, reverse=True)
        second = sizes[0] if sizes else 0
        author_lines = [s for s in below_title if abs(s["size"] - second) < 0.6]
        authors = " ".join(s["text"] for s in author_lines).strip()
        return (title or None), (authors or None)

    # 1) постранично собираем строки/капшены/сэмплы
    pages = []
    all_captions: List[Caption] = []
    for pno in range(len(doc)):
        page = doc[pno]
        lines = _extract_page_lines(page)
        caps = _collect_captions(lines, pno + 1)
        all_captions.extend(caps)
        pages.append({
            "page": pno + 1,
            "lines": lines,
            "blocks_sample": _extract_blocks_sample(page)
        })

    # первая страница текстом — для doc_id эвристик
    first_page_text = (doc[0].get_text("text") if len(doc) > 0 else "") or ""

    # 2) собираем секции
    flat: List[tuple] = []
    for p in pages:
        for i, ln in enumerate(p["lines"]):
            flat.append((p["page"], i, ln))

    heads: List[tuple] = []
    for idx, (pg, li, ln) in enumerate(flat):
        name = _is_section_heading(ln)
        if name:
            heads.append((idx, pg, name))

    sections: List[Dict[str, Any]] = []
    if heads:
        for i, (start_idx, start_pg, name) in enumerate(heads):
            end_idx = heads[i + 1][0] if i + 1 < len(heads) else len(flat)
            chunk_lines = [flat[k][2] for k in range(start_idx + 1, end_idx)]
            text = "\n".join(chunk_lines).strip()
            if not text:
                continue
            sec_pages = {flat[k][0] for k in range(start_idx, end_idx)} or {start_pg}
            sections.append({
                "name": name,
                "page_start": min(sec_pages),
                "page_end": max(sec_pages),
                "text": _normalize_text(text)
            })
    else:
        body = "\n".join([ln for _, _, ln in flat]).strip()
        sections.append({
            "name": "Body",
            "page_start": 1,
            "page_end": len(doc),
            "text": _normalize_text(body)
        })

    # 3) приводим капшены
    captions = []
    for c in all_captions:
        if not c.text:
            continue
        captions.append({
            "id": c.id,
            "kind": c.kind,
            "page": c.page,
            "text": _normalize_text(c.text)
        })
    tables = [{"id": cap["id"], "page": cap["page"], "caption": cap["text"]}
              for cap in captions if cap["kind"] == "Table"]

    # 4) метаданные из PDF
    meta_pdf = doc.metadata or {}

    # 5) формируем S0 (черновой doc_id поменяем ниже)
    s0 = {
        "doc_id": Path(out_dir).name,  # временно
        "source_pdf": str(pdf_path),
        "page_count": len(doc),
        "metadata": {
            "title": meta_pdf.get("title") or "",
            "author": meta_pdf.get("author") or "",
            "producer": meta_pdf.get("producer") or "",
            "creationDate": meta_pdf.get("creationDate") or ""
        },
        "sections": sections,
        "captions": [{"id": c["id"], "text": c["text"]} for c in captions],
        "figures": [c for c in captions if c["kind"] == "Figure"],
        "tables": tables,
        "pages_sample": [
            {"page": p["page"], "blocks_sample": p["blocks_sample"]}
            for p in pages[:2]
        ]
    }

    # 6) корректный doc_id
    s0["doc_id"] = _safe_doc_id(s0, pdf_path, first_page_text)

    # 7) эвристика заголовка/авторов с 1-й страницы (для UI), плюс arXiv в метаданные
    title_guess, authors_guess = _extract_title_and_authors(doc)
    meta = s0.get("metadata", {})
    if not meta.get("title") and title_guess:
        meta["title"] = title_guess
    if (not meta.get("author") or len(meta["author"]) < 3) and authors_guess:
        meta["author"] = authors_guess
    # arXiv из имени файла (если есть)
    m_arxiv = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', Path(pdf_path).name)
    if m_arxiv:
        meta["arxiv_id"] = m_arxiv.group(0)
    s0["metadata"] = meta

    # 8) запись и возврат
    (out_dir / "s0.json").write_text(json.dumps(s0, ensure_ascii=False, indent=2))
    doc.close()
    return s0


# CLI
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="path to PDF")
    ap.add_argument("--out", required=True, help="output directory for s0.json")
    args = ap.parse_args()
    s0 = build_s0(args.pdf, args.out)
    print(f"✅ S0 saved to {Path(args.out) / 's0.json'} | sections={len(s0['sections'])} captions={len(s0['captions'])}")

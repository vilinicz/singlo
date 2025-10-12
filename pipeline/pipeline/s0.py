# pipeline/pipeline/s0.py
from __future__ import annotations
import re, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF

# --------- Хелперы распознавания ---------

SECTION_PAT = re.compile(
    r"""(?ix) ^
        (?: \d+\.?\s+ | [ivxl]+\.\s+ )?         # опциональная нумерация "1." или "I."
        (abstract|introduction|background|
         materials\s+and\s+methods|methods|method|
         results?|discussion|conclusions?|conclusion|
         supplementary|appendix|acknowledg(e)?ments)
        \s* $
    """
)

# Заголовки в UPPERCASE без точек — тоже считаем секциями
ALLCAPS_SECTION_PAT = re.compile(r"^[A-Z][A-Z\s\-]{3,}$")

FIG_PAT = re.compile(r"^\s*(Figure|Fig\.)\s*([A-Za-z]?\d+[A-Za-z]?)\s*[:\-]?\s*(.*)$", re.I)
TAB_PAT = re.compile(r"^\s*(Table|Tab\.)\s*([A-Za-z]?\d+[A-Za-z]?)\s*[:\-]?\s*(.*)$", re.I)

@dataclass
class Caption:
    kind: str     # Figure / Table
    id: str       # "Figure2" / "Table 1"
    page: int
    text: str

# --------- Основной парсер ---------

def _extract_page_lines(page: fitz.Page) -> List[str]:
    """Строки страницы, как видит читатель (layouted text)."""
    # 'text' даёт разбивку по строкам с учётом layout
    txt = page.get_text("text")
    lines = txt.splitlines()
    return lines

def _extract_blocks_sample(page: fitz.Page, limit: int = 5) -> List[Dict[str, Any]]:
    """Небольшой сэмпл текстовых блоков с bbox — чтобы видеть провенанс."""
    blocks = []
    for i, b in enumerate(page.get_text("blocks")[:limit]):
        x0, y0, x1, y1, text, *_ = b
        if not text.strip():
            continue
        blocks.append({
            "bbox": [round(float(x0),1), round(float(y0),1), round(float(x1),1), round(float(y1),1)],
            "text_preview": text.strip().replace("\n", " ")[:160]
        })
    return blocks

def _is_section_heading(line: str) -> Optional[str]:
    raw = line.strip()
    if not raw:
        return None
    if SECTION_PAT.match(raw):
        return SECTION_PAT.match(raw).group(1).capitalize()
    # UPPERCASE вариант
    if ALLCAPS_SECTION_PAT.match(raw) and len(raw.split()) <= 5:
        # не все верхние регистры — секции; фильтруем типичные слова
        word = raw.lower()
        candidates = [
            "abstract","introduction","methods","materials and methods",
            "results","discussion","conclusions","conclusion",
            "supplementary","appendix","acknowledgements","acknowledgments"
        ]
        for c in candidates:
            if c.replace(" ","") in word.replace(" ",""):
                return c.title()
    return None

def _collect_captions(lines: List[str], page_no: int) -> List[Caption]:
    caps: List[Caption] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m_fig = FIG_PAT.match(line)
        m_tab = TAB_PAT.match(line)
        if m_fig or m_tab:
            kind = "Figure" if m_fig else "Table"
            m = m_fig or m_tab
            num = m.group(2)
            tail = m.group(3).strip()
            buf = [tail] if tail else []
            # тянем следующие строки, пока не пусто и не начался новый caption/секция
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt:
                    break
                if FIG_PAT.match(nxt) or TAB_PAT.match(nxt) or _is_section_heading(nxt):
                    break
                buf.append(nxt)
                j += 1
            cap = Caption(kind=kind, id=f"{kind}{num}", page=page_no, text=" ".join(buf).strip())
            caps.append(cap)
            i = j
            continue
        i += 1
    return caps

def build_s0(pdf_path: str, out_dir: str) -> Dict[str, Any]:
    """
    Главная функция: читает PDF и сохраняет s0.json в out_dir.
    Возвращает словарь S0.
    """
    pdf_path = str(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}

    # 1) Постранично собираем строки и сэмплы блоков
    pages = []
    all_captions: List[Caption] = []
    for pno in range(len(doc)):
        page = doc[pno]
        lines = _extract_page_lines(page)
        caps = _collect_captions(lines, pno+1)
        all_captions.extend(caps)
        pages.append({
            "page": pno + 1,
            "lines": lines,
            "blocks_sample": _extract_blocks_sample(page)
        })

    # 2) Секции: ищем заголовки по всем строкам, затем сшиваем текст интервалами
    # Соберём глобальный список (page, idx, line)
    flat: List[tuple] = []
    for p in pages:
        for i, ln in enumerate(p["lines"]):
            flat.append((p["page"], i, ln))

    # Индексы заголовков
    heads: List[tuple] = []  # (flat_idx, page, norm_name)
    for idx, (pg, li, ln) in enumerate(flat):
        name = _is_section_heading(ln)
        if name:
            heads.append((idx, pg, name))

    # Группируем участки между заголовками
    sections: List[Dict[str, Any]] = []
    if heads:
        for i, (start_idx, start_pg, name) in enumerate(heads):
            end_idx = heads[i+1][0] if i+1 < len(heads) else len(flat)
            chunk_lines = [flat[k][2] for k in range(start_idx+1, end_idx)]  # после заголовка до след. заголовка
            text = "\n".join(chunk_lines).strip()
            if not text:
                continue
            sec_pages = {flat[k][0] for k in range(start_idx, end_idx)} or {start_pg}
            sections.append({
                "name": name,
                "page_start": min(sec_pages),
                "page_end": max(sec_pages),
                "text": text
            })
    else:
        # fallback: одна большая секция Body
        body = "\n".join([ln for _,_,ln in flat]).strip()
        sections.append({
            "name": "Body",
            "page_start": 1,
            "page_end": len(doc),
            "text": body
        })

    # 3) Капшены в сводку
    captions = []
    for c in all_captions:
        if not c.text:  # короткие/пустые пропустим
            continue
        captions.append({
            "id": c.id,
            "kind": c.kind,
            "page": c.page,
            "text": c.text
        })

    # 4) Простейшие таблицы (только по обнаруженным caption’ам Table)
    tables = [{"id": cap["id"], "page": cap["page"], "caption": cap["text"]} for cap in captions if cap["kind"]=="Table"]

    # 5) Метаданные
    s0 = {
        "doc_id": Path(pdf_path).stem,
        "source_pdf": str(pdf_path),
        "page_count": len(doc),
        "metadata": {
            "title": meta.get("title") or "",
            "author": meta.get("author") or "",
            "producer": meta.get("producer") or "",
            "creationDate": meta.get("creationDate") or ""
        },
        "sections": sections,
        "captions": [{"id": c["id"], "text": c["text"]} for c in captions],  # компактный список для S1
        "figures": [c for c in captions if c["kind"]=="Figure"],
        "tables": tables,
        "pages_sample": [
            {"page": p["page"], "blocks_sample": p["blocks_sample"]} for p in pages[:2]  # только первые 2 для наглядности
        ]
    }

    out_path = out_dir / "s0.json"
    out_path.write_text(json.dumps(s0, ensure_ascii=False, indent=2))
    return s0

# CLI для локального запуска:
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="path to PDF")
    ap.add_argument("--out", required=True, help="output directory for s0.json")
    args = ap.parse_args()
    s0 = build_s0(args.pdf, args.out)
    print(f"✅ S0 saved to {Path(args.out)/'s0.json'} | sections={len(s0['sections'])} captions={len(s0['captions'])}")

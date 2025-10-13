# pipeline/pipeline/s0.py
from __future__ import annotations
import re, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF

# --------- Хелперы распознавания (детектор секций) ---------

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

# Заголовки в UPPERCASE без точек — тоже считаем секциями
ALLCAPS_SECTION_PAT = re.compile(r"^[A-Z][A-Z\s\-]{3,}$")

FIG_PAT = re.compile(r"^\s*(Figure|Fig\.)\s*([A-Za-z]?\d+[A-Za-z]?)\s*[:\-]?\s*(.*)$", re.I)
TAB_PAT = re.compile(r"^\s*(Table|Tab\.|TABLE)\s*([A-Za-z]?\d+[A-Za-z]?)?\s*[:\-]?\s*(.*)$", re.I)


@dataclass
class Caption:
    kind: str  # Figure / Table
    id: str  # "Figure2" / "Table 1"
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
            "bbox": [round(float(x0), 1), round(float(y0), 1), round(float(x1), 1), round(float(y1), 1)],
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
            "abstract", "introduction", "methods", "materials and methods",
            "results", "discussion", "conclusions", "conclusion",
            "supplementary", "appendix", "acknowledgements", "acknowledgments"
        ]
        for c in candidates:
            if c.replace(" ", "") in word.replace(" ", ""):
                return c.title()
    return None


RE_SOFT_HYPHEN_BREAK = re.compile(r'(\w)[\-­]\n(\w)')  # перенос со сливанием слов
RE_LINE_BREAK_IN_NUMBER = re.compile(r'(\d)\s*\n\s*(\d)')  # разрыв числа переносом строки
RE_LINE_BREAK_AFTER_PAREN = re.compile(r'\)\s*\n\s*(\d)')  # ") \n60" → ") 60"
RE_MULTI_SPACES = re.compile(r'[ \t]{2,}')


def _normalize_text(text: str) -> str:
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    # 1) "advec-\ntion" → "advection"
    t = RE_SOFT_HYPHEN_BREAK.sub(r'\1\2', t)
    # 2) "40.\n0%" → "40.0%"
    t = RE_LINE_BREAK_IN_NUMBER.sub(r'\1.\2', t)
    # 3) ") \n60" → ") 60"
    t = RE_LINE_BREAK_AFTER_PAREN.sub(r') \1', t)
    # 4) схлопываем одиночные переносы внутри абзаца в пробел
    t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)
    # 5) лишние пробелы
    t = RE_MULTI_SPACES.sub(' ', t)
    return t


def _collect_captions(lines: List[str], page_no: int) -> List[Caption]:
    caps: List[Caption] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m_fig = FIG_PAT.match(line)
        m_tab = TAB_PAT.match(line)
        m_tab_roman = re.match(r"^\s*TABLE\s+([IVXLC]+)\s*[:\-]?\s*(.*)$", line, re.I)
        if m_tab_roman:
            num = m_tab_roman.group(1)
            tail = m_tab_roman.group(2).strip()
            buf = [tail] if tail else []
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt:
                    break
                if FIG_PAT.match(nxt) or TAB_PAT.match(nxt) or _is_section_heading(nxt) \
                        or re.match(r"^\s*TABLE\s+([IVXLC]+)\s*[:\-]?", nxt, re.I):
                    break
                buf.append(nxt)
                j += 1
            caps.append(Caption(kind="Table", id=f"Table{num}", page=page_no, text=" ".join(buf).strip()))
            i = j
            continue
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

    def _extract_title_and_authors(doc):
        page = doc[0]
        d = page.get_text("dict")  # blocks/spans с размерами шрифтов
        spans = []
        for b in d.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    spans.append({
                        "text": s.get("text", "").strip(),
                        "size": s.get("size", 0.0),
                        "font": s.get("font", ""),
                        "y": l.get("bbox", [0, 0, 0, 0])[1],
                    })
        spans = [s for s in spans if s["text"]]
        if not spans:
            return None, None
        # заголовок = 1–3 верхних строки с самым крупным кеглем
        max_size = max(s["size"] for s in spans)
        title_lines = [s for s in spans if s["size"] >= max_size - 0.5 and s["y"] < spans[0]["y"] + 300]
        title = " ".join(s["text"] for s in title_lines).strip()
        # авторы = следующий “ярус” шрифта
        second_size = max([s["size"] for s in spans if s["size"] < max_size], default=0)
        author_lines = [s for s in spans if second_size and abs(s["size"] - second_size) < 0.5 and s["y"] > (
            title_lines[0]["y"] if title_lines else 0)]
        authors = " ".join(s["text"] for s in author_lines).strip()
        return (title or None), (authors or None)

    # 1) Постранично собираем строки и сэмплы блоков
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
            end_idx = heads[i + 1][0] if i + 1 < len(heads) else len(flat)
            chunk_lines = [flat[k][2] for k in range(start_idx + 1, end_idx)]  # после заголовка до след. заголовка
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
        # fallback: одна большая секция Body
        body = "\n".join([ln for _, _, ln in flat]).strip()
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
            "text": _normalize_text(c.text)
        })

        # 4) Простейшие таблицы ...
        tables = [{"id": cap["id"], "page": cap["page"], "caption": cap["text"]}
                  for cap in captions if cap["kind"] == "Table"]

        # 5) Метаданные ИЗ PDF — ВАЖНО: взять до использования
        meta_pdf = doc.metadata or {}
        doc_id_dir = Path(out_dir).name

        # базовый S0 с «сырыми» pdf-метаданными
        s0 = {
            "doc_id": doc_id_dir,
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

        # эвристика: заголовок/авторы с 1-й страницы (если в PDF-мета пусто)
        title_guess, authors_guess = _extract_title_and_authors(doc)
        meta = s0.get("metadata", {})
        if not meta.get("title") and title_guess:
            meta["title"] = title_guess
        if (not meta.get("author") or len(meta["author"]) < 3) and authors_guess:
            meta["author"] = authors_guess

        # бонус: вытащим arXiv id из имени файла, если есть
        m_arxiv = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', Path(pdf_path).name)
        if m_arxiv:
            meta["arxiv_id"] = m_arxiv.group(0)

        s0["metadata"] = meta

        out_path = out_dir / "s0.json"
        out_path.write_text(json.dumps(s0, ensure_ascii=False, indent=2))
        doc.close()
        return s0


# CLI для локального запуска:
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="path to PDF")
    ap.add_argument("--out", required=True, help="output directory for s0.json")
    args = ap.parse_args()
    s0 = build_s0(args.pdf, args.out)
    print(f"✅ S0 saved to {Path(args.out) / 's0.json'} | sections={len(s0['sections'])} captions={len(s0['captions'])}")

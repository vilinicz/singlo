"""LaTeX source parser to enrich S0 when arXiv sources are available.

Expands \input/\include, strips comments/math, extracts sections and captions,
and returns a structure convertible to S0-like fields.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import normalize_text, line_number, slugify


## Normalize a heading title (capitalize ALLCAPS, strip noise).
def _normalize_heading_title(title: str) -> str:
    cleaned = normalize_text(_clean_latex_text(title))
    if not cleaned:
        return ""
    if cleaned.isupper():
        return cleaned.title()
    return cleaned

INPUT_INCLUDE_RE = re.compile(r'\\(input|include)\s*\{([^}]*)\}', re.I)
LATEX_SECTION_CMD_RE = re.compile(r'\\(chapter|section|subsection|subsubsection)\*?\s*\{([^{}]*)\}', re.I)
LATEX_ABSTRACT_RE = re.compile(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', re.S | re.I)
LATEX_FLOAT_ENV_RE = re.compile(
    r'\\begin\{(?P<env>figure\*?|table\*?|longtable|sidewaystable|sidewaysfigure)\}'
    r'(?P<opts>\[[^\]]*\])?(?P<body>.*?)\\end\{(?P=env)\}',
    re.S | re.I
)
LATEX_CAPTION_CMD_RE = re.compile(r'\\caption(?P<star>\*?)(?P<opt>\[[^\]]*\])?\s*\{', re.I)
LATEX_CAPTION_OF_CMD_RE = re.compile(r'\\captionof\{(?P<kind>figure|table)\}(?P<opt>\[[^\]]*\])?\s*\{', re.I)
LATEX_LABEL_RE = re.compile(r'\\label\{([^{}]+)\}', re.I)
LATEX_COMMAND_WITH_ARG_RE = re.compile(r'\\[a-zA-Z]+\*?\s*\{([^{}]*)\}')
LATEX_COMMAND_RE = re.compile(r'\\[a-zA-Z]+(\*?)(\[[^\]]*\])?')
LATEX_INCLUDEGRAPHICS_RE = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\s*\{([^{}]+)\}', re.I)
DISPLAY_BLOCK_MATH_RE = re.compile(r'\\begin\{(align|equation|eqnarray|multline|gather)\}.*?\\end{\1}', re.S | re.I)
DISPLAY_DOLLAR_MATH_RE = re.compile(r'\$\$.*?\$\$', re.S)
DISPLAY_BRACKET_MATH_RE = re.compile(r'\\\[.*?\\\]', re.S)
INLINE_MATH_RE = re.compile(r'\$(?:\\.|[^$])+\$', re.S)


## Remove LaTeX comments (%) from text while preserving content.
def _strip_latex_comments(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        lines.append(re.sub(r'(?<!\\)%.*', '', line))
    return "\n".join(lines)


## Recursively expand \input/\include statements relative to base_dir.
def _expand_latex_inputs(text: str, base_dir: Path, visited: Optional[set[str]] = None) -> str:
    visited = visited or set()

    def repl(match: re.Match) -> str:
        raw = (match.group(2) or "").strip()
        if not raw:
            return ""
        candidates = []
        if not raw.lower().endswith(".tex"):
            candidates.append(base_dir / f"{raw}.tex")
        candidates.append(base_dir / raw)
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in visited:
                return ""
            visited.add(key)
            try:
                content = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return ""
            return "\n" + _expand_latex_inputs(content, candidate.parent, visited) + "\n"
        return ""

    return INPUT_INCLUDE_RE.sub(repl, text)


## Pick a main .tex file under source_root (by priority and presence of \begin{document}).
def _load_main_tex(source_root: Path) -> tuple[Optional[Path], Optional[str]]:
    tex_files = sorted(source_root.rglob("*.tex"))
    if not tex_files:
        return None, None

    priority = ("main.tex", "paper.tex", "ms.tex", "manuscript.tex")
    for name in priority:
        for path in tex_files:
            if path.name.lower() == name:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if "\\begin{document}" in text:
                    return path, text

    best_path: Optional[Path] = None
    best_text: Optional[str] = None
    best_score: Optional[tuple[int, str]] = None
    for path in tex_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "\\begin{document}" not in text:
            continue
        rel_depth = len(path.relative_to(source_root).parts)
        score = (rel_depth, path.name.lower())
        if best_score is None or score < best_score:
            best_path = path
            best_text = text
            best_score = score
    if best_path and best_text is not None:
        return best_path, best_text

    try:
        fallback_text = tex_files[0].read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, None
    return tex_files[0], fallback_text


## Strip math, refs, citations, footnotes and commands from LaTeX to plain text.
def _clean_latex_text(text: Optional[str]) -> str:
    if not text:
        return ""
    t = text
    t = DISPLAY_BLOCK_MATH_RE.sub(" ", t)
    t = DISPLAY_DOLLAR_MATH_RE.sub(" ", t)
    t = DISPLAY_BRACKET_MATH_RE.sub(" ", t)
    t = INLINE_MATH_RE.sub(" ", t)
    t = re.sub(r'\\footnote\{.*?\}', ' ', t, flags=re.S)
    t = re.sub(r'\\cite[t|p]?\*?(?:\[[^\]]*\])?\{[^{}]*\}', ' ', t, flags=re.I)
    t = re.sub(r'\\ref\*?(?:\[[^\]]*\])?\{[^{}]*\}', ' ', t, flags=re.I)
    for _ in range(3):
        t = LATEX_COMMAND_WITH_ARG_RE.sub(r' \1 ', t)
    t = LATEX_COMMAND_RE.sub(' ', t)
    t = t.replace('~', ' ')
    t = t.replace('\\', ' ')
    t = re.sub(r'[{}]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


## Extract {...} content starting at '{' index; returns (content, end_idx).
def _extract_braced_argument(text: str, start_idx: int) -> tuple[str, int]:
    n = len(text)
    if start_idx >= n or text[start_idx] != '{':
        return "", start_idx
    depth = 0
    i = start_idx
    content_start = start_idx + 1
    while i < n:
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[content_start:i], i + 1
        elif ch == '\\' and i + 1 < n:
            i += 1
        i += 1
    return "", start_idx


## Extract simple metadata fields from LaTeX preamble (title, author, date).
def _extract_latex_metadata(tex_source: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    mt = re.search(r'\\title\{(.*?)\}', tex_source, re.S | re.I)
    if mt:
        meta["title"] = normalize_text(_clean_latex_text(mt.group(1)))
    ma = re.search(r'\\author\{(.*?)\}', tex_source, re.S | re.I)
    if ma:
        raw = ma.group(1).replace("\\and", ", ")
        meta["author"] = normalize_text(_clean_latex_text(raw))
    md = re.search(r'\\date\{(.*?)\}', tex_source, re.S | re.I)
    if md:
        meta["date"] = normalize_text(_clean_latex_text(md.group(1)))
    return meta


## Parse LaTeX sectioning commands into a list of sections with text excerpts.
def _parse_latex_sections(doc_body: str,
                           doc_offset: int,
                           full_source: str,
                           source_label: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    body = doc_body or ""

    def _section_prov(start_idx: int, end_idx: int) -> Dict[str, Any]:
        abs_start = doc_offset + start_idx
        abs_end = doc_offset + end_idx
        return {
            "source": "latex",
            "file": source_label,
            "offset": [abs_start, abs_end],
            "line_start": line_number(full_source, abs_start),
            "line_end": line_number(full_source, abs_end)
        }

    abstract_match = LATEX_ABSTRACT_RE.search(body)
    if abstract_match:
        abs_text = normalize_text(_clean_latex_text(abstract_match.group(1)))
        if abs_text:
            sections.append({
                "name": "Abstract",
                "title_path": "Abstract",
                "heading": "Abstract",
                "level": "abstract",
                "page_start": None,
                "page_end": None,
                "text": abs_text,
                "provenance": [_section_prov(abstract_match.start(1), abstract_match.end(1))]
            })
        body = body[:abstract_match.start()] + " " + body[abstract_match.end():]

    matches = list(LATEX_SECTION_CMD_RE.finditer(body))
    if matches:
        hierarchy: Dict[str, Optional[str]] = {
            "chapter": None,
            "section": None,
            "subsection": None,
            "subsubsection": None,
        }
        level_order = ["chapter", "section", "subsection", "subsubsection"]

        for idx, match in enumerate(matches):
            level = (match.group(1) or "").lower()
            title_raw = match.group(2) or ""
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
            chunk_raw = body[start:end]
            chunk = normalize_text(_clean_latex_text(chunk_raw))
            if not chunk:
                # even если текст пуст, обновляем иерархию
                hierarchy[level] = _normalize_heading_title(title_raw) or level.title()
                for deeper in level_order[level_order.index(level) + 1:]:
                    hierarchy[deeper] = None
                continue

            cleaned_title = _normalize_heading_title(title_raw)
            if not cleaned_title:
                cleaned_title = level.title()

            hierarchy[level] = cleaned_title
            # сбрасываем дочерние уровни
            for deeper in level_order[level_order.index(level) + 1:]:
                hierarchy[deeper] = None

            full_path: List[str] = []
            if level == "section":
                full_path.append(cleaned_title)
            elif level == "subsection":
                if hierarchy.get("section"):
                    full_path.append(hierarchy.get("section"))
                full_path.append(cleaned_title)
            elif level == "subsubsection":
                if hierarchy.get("section"):
                    full_path.append(hierarchy.get("section"))
                if hierarchy.get("subsection"):
                    full_path.append(hierarchy.get("subsection"))
                full_path.append(cleaned_title)
            else:
                full_path.append(cleaned_title)

            if level == "section":
                display_name = cleaned_title
            elif level == "subsection":
                display_name = hierarchy.get("section") or cleaned_title
            elif level == "subsubsection":
                display_name = hierarchy.get("subsection") or hierarchy.get("section") or cleaned_title
            else:
                display_name = cleaned_title

            sections.append({
                "name": display_name,
                "title_path": " > ".join(full_path),
                "heading": cleaned_title,
                "level": level,
                "page_start": None,
                "page_end": None,
                "text": chunk,
                "provenance": [_section_prov(start, end)]
            })
    else:
        content = normalize_text(_clean_latex_text(body))
        if content:
            prov = _section_prov(0, len(body))
            sections.append({
                "name": "Body",
                "title_path": "Body",
                "heading": "Body",
                "level": "body",
                "page_start": None,
                "page_end": None,
                "text": content,
                "provenance": [prov]
            })

    return sections


## Extract captions from figure/table environments and caption(of) commands.
def _extract_latex_captions(doc_body: str,
                             doc_offset: int,
                             full_source: str,
                             source_label: str) -> List[Dict[str, Any]]:
    captions: List[Dict[str, Any]] = []
    counters = {"Figure": 0, "Table": 0}
    seen_ids: set[str] = set()

    def _canonical_kind(env_name: str) -> str:
        env = (env_name or "").lower()
        if env.endswith("*"):
            env = env[:-1]
        if env in ("figure", "sidewaysfigure"):
            return "Figure"
        return "Table"

    def _assign_id(kind: str, label: Optional[str]) -> str:
        if label and label not in seen_ids:
            seen_ids.add(label)
            return label
        counters[kind] = counters.get(kind, 0) + 1
        generated = f"{kind}{counters[kind]}"
        seen_ids.add(generated)
        return generated

    body = doc_body or ""
    for match in LATEX_FLOAT_ENV_RE.finditer(body):
        env_name = match.group("env") or ""
        block = match.group("body") or ""
        caption_text = ""
        caption_end = -1
        caption_content_start = None

        for cap_match in LATEX_CAPTION_CMD_RE.finditer(block):
            start_brace_idx = cap_match.end() - 1
            if start_brace_idx < 0 or start_brace_idx >= len(block):
                continue
            raw_content, end_idx = _extract_braced_argument(block, start_brace_idx)
            caption_text = normalize_text(_clean_latex_text(raw_content))
            caption_end = end_idx
            caption_content_start = start_brace_idx + 1
            if caption_text:
                break
        if not caption_text or caption_content_start is None:
            continue

        body_offset = match.start("body") if match.start("body") is not None else 0
        abs_start = doc_offset + body_offset + caption_content_start
        abs_end = abs_start + len(raw_content)

        label_value: Optional[str] = None
        if caption_end != -1:
            for lbl in LATEX_LABEL_RE.finditer(block[caption_end:]):
                label_value = lbl.group(1)
                if label_value:
                    break
        if not label_value:
            for lbl in LATEX_LABEL_RE.finditer(block):
                label_value = lbl.group(1)
                if label_value:
                    break

        kind = _canonical_kind(env_name)
        cid = _assign_id(kind, label_value)
        graphics = LATEX_INCLUDEGRAPHICS_RE.findall(block)
        captions.append({
            "id": cid,
            "kind": kind,
            "page": None,
            "text": caption_text,
            "provenance": [{
                "source": "latex",
                "file": source_label,
                "offset": [abs_start, abs_end],
                "line_start": line_number(full_source, abs_start),
                "line_end": line_number(full_source, abs_end)
            }],
            "graphics": graphics
        })

    for capof in LATEX_CAPTION_OF_CMD_RE.finditer(body):
        start_brace_idx = capof.end() - 1
        raw_content, end_idx = _extract_braced_argument(body, start_brace_idx)
        caption_text = normalize_text(_clean_latex_text(raw_content))
        if not caption_text:
            continue
        abs_start = doc_offset + start_brace_idx + 1
        abs_end = abs_start + len(raw_content)
        label_value: Optional[str] = None
        for lbl in LATEX_LABEL_RE.finditer(body[end_idx:]):
            label_value = lbl.group(1)
            if label_value:
                break
        kind = "Figure" if capof.group("kind").lower().startswith("figure") else "Table"
        cid = _assign_id(kind, label_value)
        captions.append({
            "id": cid,
            "kind": kind,
            "page": None,
            "text": caption_text,
            "provenance": [{
                "source": "latex",
                "file": source_label,
                "offset": [abs_start, abs_end],
                "line_start": line_number(full_source, abs_start),
                "line_end": line_number(full_source, abs_end)
            }],
            "graphics": []
        })

    return captions


def parse_latex_sources(arxiv_id: str,
                        source_root: Path,
                        pdf_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    main_tex_path, tex_source = _load_main_tex(source_root)
    if not tex_source:
        return None

    tex_source = _strip_latex_comments(tex_source)
    if main_tex_path:
        try:
            visited = {str(main_tex_path.resolve())}
        except OSError:
            visited = set()
        tex_source = _expand_latex_inputs(tex_source, main_tex_path.parent, visited=visited)
        tex_source = _strip_latex_comments(tex_source)

    body_start_match = re.search(r'\\begin\{document\}', tex_source, re.I)
    doc_offset = 0
    doc_body = tex_source
    if body_start_match:
        doc_offset = body_start_match.end()
        remainder = tex_source[doc_offset:]
        body_end_match = re.search(r'\\end\{document\}', remainder, re.I)
        if body_end_match:
            doc_body = remainder[:body_end_match.start()]
        else:
            doc_body = remainder

    source_label = str(main_tex_path.relative_to(source_root)) if main_tex_path else "main.tex"
    latex_meta = _extract_latex_metadata(tex_source)
    sections = _parse_latex_sections(doc_body, doc_offset, tex_source, source_label)
    for idx, sec in enumerate(sections):
        sec["id"] = f"sec-{idx + 1:02d}"

    captions = _extract_latex_captions(doc_body, doc_offset, tex_source, source_label)
    figures = []
    tables = []
    for cap in captions:
        entry = {
            "id": cap.get("id"),
            "page": None,
            "caption": cap.get("text"),
            "caption_provenance": cap.get("provenance"),
            "graphics": cap.get("graphics")
        }
        if cap.get("kind") == "Figure":
            figures.append(entry)
        elif cap.get("kind") == "Table":
            tables.append(entry)

    captions_for_s0 = [
        {
            "id": cap.get("id"),
            "kind": cap.get("kind"),
            "text": cap.get("text"),
            "provenance": cap.get("provenance"),
            "graphics": cap.get("graphics")
        }
        for cap in captions
    ]

    metadata_updates = {
        "title": latex_meta.get("title") or pdf_metadata.get("title") or "",
        "author": latex_meta.get("author") or pdf_metadata.get("author") or "",
        "producer": pdf_metadata.get("producer") or "",
        "creationDate": pdf_metadata.get("creationDate") or "",
        "arxiv_id": arxiv_id,
        "source_format": "latex"
    }
    if latex_meta.get("date"):
        metadata_updates["date"] = latex_meta["date"]

    return {
        "sections": sections,
        "captions": captions_for_s0,
        "figures": figures,
        "tables": tables,
        "pages_sample": [],
        "metadata": metadata_updates,
        "suggested_doc_id": slugify(arxiv_id) if arxiv_id else None
    }

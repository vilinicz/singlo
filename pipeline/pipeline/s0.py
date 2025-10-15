from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import pymupdf

from .parsers import (
    extract_title_and_authors,
    parse_latex_sources,
    parse_pdf_document,
)
from .parsers.utils import slugify

JUNK_TITLE_PAT = re.compile(r'^(title|untitled|document)\b', re.I)
JUNK_PRODUCER_PAT = re.compile(r'(microsoft\s+word|adobe\s+pdf|libreoffice|pages)\b', re.I)
ARXIV_CANONICAL_RE = re.compile(r'arxiv\s*[:/ ]\s*(\d{4}\.\d{4,5})(v\d+)?', re.I)
ARXIV_URL_RE = re.compile(r'arxiv\.org/(?:abs|pdf|format)/(\d{4}\.\d{4,5})(v\d+)?', re.I)
ARXIV_RE = re.compile(r'\b(\d{4}\.\d{4,5})(v\d+)?\b', re.I)

def _looks_junky(value: str) -> bool:
    if not value:
        return True
    value = value.strip()
    return bool(JUNK_TITLE_PAT.match(value) or JUNK_PRODUCER_PAT.search(value))


def _is_valid_arxiv_id(candidate: str) -> bool:
    if not candidate or "." not in candidate:
        return False
    prefix, _ = candidate.split(".", 1)
    if len(prefix) != 4 or not prefix.isdigit():
        return False
    month = int(prefix[2:])
    return 1 <= month <= 12


def _extract_arxiv_id_from_text(text: str) -> Optional[str]:
    txt = text or ""
    for patt in (ARXIV_CANONICAL_RE, ARXIV_URL_RE):
        match = patt.search(txt)
        if match:
            candidate = match.group(1)
            if _is_valid_arxiv_id(candidate):
                return candidate + (match.group(2) or "")
    for match in ARXIV_RE.finditer(txt):
        candidate = match.group(1)
        if _is_valid_arxiv_id(candidate):
            return candidate + (match.group(2) or "")
    return None


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


def _safe_doc_id(s0: Dict[str, Any], out_dir: Path, first_page_text: str = "") -> str:
    meta = s0.get("metadata") or {}
    src = s0.get("source_pdf") or ""
    raw_id = (s0.get("doc_id") or "").strip()

    if raw_id and not _looks_junky(raw_id):
        return slugify(raw_id)

    arx = _guess_arxiv_id(src, first_page_text, meta)
    if arx:
        return slugify(arx)
    if src:
        return slugify(Path(src).stem)
    if out_dir.name:
        return slugify(out_dir.name)
    return 'doc-' + hashlib.md5(str(out_dir).encode('utf-8')).hexdigest()[:8]


def _extract_arxiv_id_from_metadata(meta: Dict[str, Any]) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    for key in ("arxiv_id", "identifier", "subject", "keywords", "title"):
        value = meta.get(key)
        if isinstance(value, str):
            found = _extract_arxiv_id_from_text(value)
            if found:
                return found
    return None


def _find_arxiv_id(pdf_path: str, doc: pymupdf.Document, first_page_text: str) -> Optional[str]:
    meta = doc.metadata or {}
    arx = _extract_arxiv_id_from_metadata(meta)
    if arx:
        return arx

    candidates: list[str] = []
    for idx in range(min(len(doc), 5)):
        try:
            page = doc[idx]
        except Exception:
            continue
        candidates.append(page.get_text("text") or "")
        try:
            for block in page.get_text("blocks"):
                if not block or len(block) < 5:
                    continue
                candidates.append(block[4] or "")
        except Exception:
            pass
    candidates.append(first_page_text or "")
    candidates.append(Path(pdf_path).stem)

    for text in candidates:
        arx = _extract_arxiv_id_from_text(text or "")
        if arx:
            return arx
    return None


def _safe_tar_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            raise ValueError(f"Blocked unsafe tar member: {member.name}")
    tar.extractall(dest)


def _download_arxiv_source(arxiv_id: str, cache_dir: Path) -> Optional[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_dir = cache_dir / slugify(arxiv_id)
    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir

    def _url_candidates(arxiv: str):
        if "v" in arxiv and arxiv.rsplit("v", 1)[-1].isdigit():
            bare = arxiv.split("v", 1)[0]
            yield f"https://arxiv.org/e-print/{arxiv}"
            yield f"https://arxiv.org/src/{arxiv}"
            yield f"https://arxiv.org/e-print/{bare}"
            yield f"https://arxiv.org/src/{bare}"
        else:
            yield f"https://arxiv.org/e-print/{arxiv}"
            yield f"https://arxiv.org/src/{arxiv}"

    for url in _url_candidates(arxiv_id):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "singlo-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
        except urllib.error.HTTPError as err:
            if err.code in (403, 404):
                continue
            return None
        except urllib.error.URLError:
            return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_tar = Path(tmp_dir) / "src.tar"
            tmp_tar.write_bytes(data)
            extract_root = Path(tmp_dir) / "src"
            extract_root.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(tmp_tar, mode="r:*") as tarf:
                    _safe_tar_extract(tarf, extract_root)
            except (tarfile.TarError, ValueError):
                continue

            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            shutil.move(str(extract_root), target_dir)
            return target_dir
    return None


def build_s0(pdf_path: str, out_dir: str, *, use_latex: bool = True) -> Dict[str, Any]:
    pdf_path = str(pdf_path)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    try:
        page_count = len(doc)
        first_page_text = doc[0].get_text("text") if page_count else ""
        pdf_metadata = doc.metadata or {}

        metadata = {
            "title": pdf_metadata.get("title") or "",
            "author": pdf_metadata.get("author") or "",
            "producer": pdf_metadata.get("producer") or "",
            "creationDate": pdf_metadata.get("creationDate") or "",
            "source_format": "pdf"
        }

        arxiv_id = _find_arxiv_id(pdf_path, doc, first_page_text)
        core_data: Optional[Dict[str, Any]] = None

        if use_latex and arxiv_id:
            source_cache = out_dir_path / "_arxiv_cache"
            source_dir = _download_arxiv_source(arxiv_id, source_cache)
            if source_dir:
                latex_result = parse_latex_sources(arxiv_id, source_dir, pdf_metadata)
                if latex_result:
                    core_data = latex_result
                    metadata.update(latex_result.get("metadata", {}))

        if core_data is None:
            pdf_result = parse_pdf_document(doc)
            core_data = pdf_result
            metadata["source_format"] = "pdf"
            if arxiv_id:
                metadata.setdefault("arxiv_id", arxiv_id)
        else:
            metadata.setdefault("source_format", "latex")

        title_guess, authors_guess = extract_title_and_authors(doc)
        if not metadata.get("title") and title_guess:
            metadata["title"] = title_guess
        if (not metadata.get("author") or len(metadata.get("author", "")) < 3) and authors_guess:
            metadata["author"] = authors_guess

        sections_raw = core_data.get("sections", []) or []
        sections: list[Any] = []
        for section in sections_raw:
            if isinstance(section, dict):
                filtered = {k: v for k, v in section.items() if k != "sentences"}
                sections.append(filtered)
            else:
                sections.append(section)

        s0_data = {
            "doc_id": core_data.get("suggested_doc_id") or slugify(out_dir_path.name),
            "source_pdf": str(pdf_path),
            "page_count": page_count,
            "metadata": metadata,
            "sections": sections,
            "captions": core_data.get("captions", []),
            "figures": core_data.get("figures", []),
            "tables": core_data.get("tables", [])
        }

        s0_data["doc_id"] = _safe_doc_id(s0_data, out_dir_path, first_page_text)
        (out_dir_path / "s0.json").write_text(json.dumps(s0_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return s0_data
    finally:
        doc.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="path to PDF")
    parser.add_argument("--out", required=True, help="output directory for s0.json")
    parser.add_argument("--no-latex", action="store_true", help="disable LaTeX source parsing")
    args = parser.parse_args()

    s0 = build_s0(args.pdf, args.out, use_latex=False)
    print(f"✅ S0 saved to {Path(args.out) / 's0.json'} | sections={len(s0['sections'])} captions={len(s0['captions'])}")

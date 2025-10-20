from __future__ import annotations

import re
from typing import Optional

# Text normalization helpers for PDF/LaTeX extraction

RE_SOFT_HYPHEN_BREAK = re.compile(r'(\w)[\-­]\n(\w)')
RE_LINE_BREAK_IN_NUMBER = re.compile(r'(\d)\s*\n\s*(\d)')
RE_LINE_BREAK_AFTER_PAREN = re.compile(r'\)\s*\n\s*(\d)')
RE_MULTI_SPACES = re.compile(r'[ \t]{2,}')


def normalize_text(text: Optional[str]) -> str:
    """Normalize whitespace and soft hyphen breaks in extracted text."""
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = RE_SOFT_HYPHEN_BREAK.sub(r"\1\2", t)
    t = RE_LINE_BREAK_IN_NUMBER.sub(r"\1.\2", t)
    t = RE_LINE_BREAK_AFTER_PAREN.sub(r") \1", t)
    t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)
    t = RE_MULTI_SPACES.sub(' ', t)
    return t.strip()


def slugify(value: str) -> str:
    """Create filesystem-friendly slug."""
    value = (value or "").strip().lower()
    value = re.sub(r'[^a-z0-9]+', '-', value)
    value = re.sub(r'-{2,}', '-', value).strip('-')
    return value or 'doc'


def line_number(source: str, offset: int) -> int:
    """Return 1-based line number for character offset."""
    if offset <= 0:
        return 1
    return source.count("\n", 0, min(len(source), offset)) + 1

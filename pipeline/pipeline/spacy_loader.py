# -*- coding: utf-8 -*-
"""
spacy_loader.py — загрузка spaCy-модели и data-driven паттернов (matcher/depmatcher + lexicon).

Публичные функции:
- load_spacy_model() -> (nlp, dep_enabled, model_name)
- load_spacy_patterns(nlp, themes_root, themes)
    -> (matcher, depmatcher|None, type_boosts, loaded_paths)
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.matcher import Matcher, DependencyMatcher

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

def load_spacy_model() -> Tuple["spacy.language.Language", bool, str]:
    """
    Пытаемся загрузить модель из ENV (SPACY_MODEL) или en_core_web_sm.
    Если не получилось — fallback на blank('en') с sentencizer.
    Возвращаем (nlp, dep_enabled, model_name).
    """
    name = os.getenv("SPACY_MODEL", "en_core_web_sm")
    try:
        nlp = spacy.load(name, disable=["ner", "textcat", "lemmatizer"])
        dep_enabled = "parser" in nlp.pipe_names
        if not dep_enabled and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp, dep_enabled, name
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp, False, f"blank_en (fallback; missing {name})"

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

_NODE_TYPES = [
    "Input Fact", "Hypothesis", "Experiment", "Technique",
    "Result", "Dataset", "Analysis", "Conclusion"
]
_TYPE_CANON = {t.lower().replace(" ", ""): t for t in _NODE_TYPES}

def _canon_type(label: str) -> Optional[str]:
    k = (label or "").lower().replace(" ", "")
    return _TYPE_CANON.get(k)

def _read_json_array(path: Path) -> List[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(data, list):
        raise RuntimeError(f"{path}: root must be a JSON array")
    return data

def _read_lexicon_dict(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: root must be a JSON object")
    return data

def _collect_pattern_files(themes_root: Path, themes: Optional[List[str]]) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Возвращает (matcher_paths, depmatcher_paths, lexicon_paths)
    Всегда включает common; затем — перечисленные темы (если есть).
    """
    themes = [t for t in (themes or []) if t]
    if "common" not in themes:
        themes = ["common"] + themes

    matcher_paths: List[Path] = []
    depmatcher_paths: List[Path] = []
    lexicon_paths: List[Path] = []

    def add_topic(t: str):
        tdir = themes_root / t
        pdir = tdir / "patterns"
        if (pdir / "matcher.json").exists():
            matcher_paths.append(pdir / "matcher.json")
        if (pdir / "depmatcher.json").exists():
            depmatcher_paths.append(pdir / "depmatcher.json")
        if (tdir / "lexicon.json").exists():
            lexicon_paths.append(tdir / "lexicon.json")

    for t in themes:
        add_topic(t)

    return matcher_paths, depmatcher_paths, lexicon_paths

# ─────────────────────────────────────────────────────────────
# OPTIONAL NODE EXPANSION (handles {"OP":"?"} in dep patterns)
# ─────────────────────────────────────────────────────────────

def _expand_optional_nodes(seq: List[dict]) -> List[List[dict]]:
    """
    DependencyMatcher sequence format НЕ поддерживает 'OP' на узлах.
    Мы раскрываем опциональные узлы ("OP":"?") в 2 ветки: с узлом и без.
    Возвращает список валидных последовательностей без ключа OP.
    """
    out: List[List[dict]] = []

    def rec(i: int, acc: List[dict]):
        if i >= len(seq):
            if acc:  # отбрасываем пустые
                out.append(acc)
            return
        node = dict(seq[i])  # копия
        op = node.pop("OP", None)
        if op == "?":
            # 1) пропустить узел
            rec(i + 1, acc)
            # 2) включить узел (без OP)
            rec(i + 1, acc + [node])
        else:
            # обязательный узел
            rec(i + 1, acc + [node])

    rec(0, [])
    # на всякий – фильтруем явные мусорные варианты
    cleaned = []
    for pat in out:
        # требуем хотя бы один RIGHT_ID (иначе spaCy падает)
        if any("RIGHT_ID" in d for d in pat):
            cleaned.append(pat)
    return cleaned

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def load_spacy_patterns(
        nlp: "spacy.language.Language",
        themes_root: str | Path,
        themes: Optional[List[str]]
) -> Tuple[Matcher, Optional[DependencyMatcher], Dict[str, float], Dict[str, List[str]]]:
    """
    Собирает matcher/depmatcher/lexicon из набора тем.
    Возвращает:
      - matcher (всегда, иначе бросает RuntimeError),
      - depmatcher (или None, если парсера нет или не нашлось dep-паттернов),
      - type_boosts (объединённый),
      - loaded_paths: {"matcher":[...], "depmatcher":[...], "lexicons":[...]} — строки путей.
    """
    themes_root = Path(themes_root or "/app/rules/themes")
    mpaths, dpaths, lpaths = _collect_pattern_files(themes_root, themes)

    # TOKEN MATCHER
    matcher = Matcher(nlp.vocab)
    loaded_m: List[str] = []
    m_count = 0

    if not mpaths:
        raise RuntimeError(f"No matcher.json files found (themes={themes or ['(none)']}) under {themes_root}")

    for p in mpaths:
        arr = _read_json_array(p)
        added_here = 0
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            label = _canon_type(str(obj.get("label", "")))
            pattern = obj.get("pattern")
            if not label or not isinstance(pattern, list):
                continue
            try:
                matcher.add(label, [pattern])
                m_count += 1
                added_here += 1
            except Exception as e:
                raise RuntimeError(f"Bad token pattern in {p} for label={label}: {e}") from e
        if added_here > 0:
            try:
                loaded_m.append(str(p.relative_to(themes_root.parent)))
            except Exception:
                loaded_m.append(str(p))

    if m_count == 0:
        raise RuntimeError("No token patterns loaded from: " + ", ".join(str(x) for x in mpaths))

    # DEPENDENCY MATCHER (sequence format only: pattern = List[dict])
    depmatcher: Optional[DependencyMatcher] = None
    loaded_d: List[str] = []
    if ("parser" in nlp.pipe_names) and dpaths:
        depmatcher = DependencyMatcher(nlp.vocab)
        d_count = 0
        for p in dpaths:
            arr = _read_json_array(p)
            added_here = 0
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                label = _canon_type(str(obj.get("label", "")))
                pat = obj.get("pattern")
                if not label or not isinstance(pat, list) or not all(isinstance(x, dict) for x in pat):
                    raise RuntimeError(f"Dep pattern in {p} for label={label} must be a List[dict] (no 'nodes/edges').")
                # раскрываем опциональные узлы OP:"?"
                expanded = _expand_optional_nodes(pat)
                if not expanded:
                    continue
                try:
                    for seq in expanded:
                        depmatcher.add(label, [seq])  # spaCy ждёт List[List[dict]]
                    d_count += 1
                    added_here += 1
                except Exception as e:
                    raise RuntimeError(f"Bad dep pattern in {p} for label={label}: {e}") from e
            if added_here > 0:
                try:
                    loaded_d.append(str(p.relative_to(themes_root.parent)))
                except Exception:
                    loaded_d.append(str(p))
        if d_count == 0:
            depmatcher = None

    # LEXICONS
    boosts: Dict[str, float] = {}
    for p in lpaths:
        try:
            data = _read_lexicon_dict(p)
            tb = data.get("type_boosts") or {}
            if isinstance(tb, dict):
                for k, v in tb.items():
                    lt = _canon_type(str(k))
                    if lt is None:
                        continue
                    try:
                        boosts[lt] = float(v)
                    except Exception:
                        continue
        except Exception:
            continue

    loaded_paths = {"matcher": loaded_m, "depmatcher": loaded_d, "lexicons": [str(p) for p in lpaths]}
    return matcher, depmatcher, boosts, loaded_paths

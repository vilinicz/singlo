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

## Load spaCy model from env SPACY_MODEL; fallback to blank('en')+sentencizer.
def load_spacy_model() -> Tuple["spacy.language.Language", bool, str]:
    """
    Пытаемся загрузить модель из ENV (SPACY_MODEL) или en_core_web_sm.
    Если не получилось — fallback на blank('en') с sentencizer + lemmatizer.
    Возвращаем (nlp, dep_enabled, model_name).
    """
    name = os.getenv("SPACY_MODEL", "en_core_web_sm")
    try:
        # ВАЖНО: не отключаем lemmatizer
        nlp = spacy.load(name, disable=["ner", "textcat"])
        # Гарантируем наличие лемматизатора (на случай кастомной модели)
        if "lemmatizer" not in nlp.pipe_names:
            if "attribute_ruler" not in nlp.pipe_names:
                nlp.add_pipe("attribute_ruler", before="lemmatizer")
            nlp.add_pipe("lemmatizer", config={"mode": "rule"})
            try:
                nlp.initialize()
            except Exception:
                # Если модель уже инициализирована — игнорируем
                pass

        dep_enabled = "parser" in nlp.pipe_names
        if not dep_enabled and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        return nlp, dep_enabled, name

    except Exception:
        # Fallback: blank('en') + sentencizer + lemmatizer(rule)
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        if "attribute_ruler" not in nlp.pipe_names:
            nlp.add_pipe("attribute_ruler")
        if "lemmatizer" not in nlp.pipe_names:
            nlp.add_pipe("lemmatizer", config={"mode": "rule"})
        try:
            nlp.initialize()
        except Exception:
            pass

        # parser в blank-пайплайне отсутствует → dep-мэтчи отключим
        return nlp, False, f"blank_en (fallback; missing {name})"


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

_NODE_TYPES = [
    "Input Fact", "Hypothesis", "Experiment", "Technique",
    "Result", "Dataset", "Analysis", "Conclusion"
]
_TYPE_CANON = {t.lower().replace(" ", ""): t for t in _NODE_TYPES}

## Canonicalize a free-form label into one of the configured node types.
def _canon_type(label: str) -> Optional[str]:
    k = (label or "").lower().replace(" ", "")
    return _TYPE_CANON.get(k)

## Read a JSON array from path; raise if invalid or wrong root type.
def _read_json_array(path: Path) -> List[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(data, list):
        raise RuntimeError(f"{path}: root must be a JSON array")
    return data

## Read a JSON object lexicon from path; raise if invalid or wrong root type.
def _read_lexicon_dict(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: root must be a JSON object")
    return data

## Collect matcher/depmatcher/lexicon files for selected themes (always includes 'common').
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

## Expand DependencyMatcher sequence nodes with {"OP":"?"} into variants.
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

## Load token/dependency patterns and lexicons for the given themes.
def load_spacy_patterns(
        nlp: "spacy.language.Language",
        themes_root: str | Path,
        themes: Optional[List[str]]
) -> Tuple[Matcher, Optional[DependencyMatcher], Dict[str, float], Dict[str, str], Dict[str, List[str]]]:
    """
    Собирает matcher/depmatcher/lexicon из набора тем.
    Возвращает:
      - matcher (всегда, иначе бросает RuntimeError),
      - depmatcher (или None, если парсера нет или не нашлось dep-паттернов),
      - type_boosts (объединённый),
      - rule_labels: имя правила → канонический тип,
      - loaded_paths: {"matcher":[...], "depmatcher":[...], "lexicons":[...]} — строки путей.
    """
    themes_root = Path(themes_root or "/app/rules/themes")
    mpaths, dpaths, lpaths = _collect_pattern_files(themes_root, themes)

    # TOKEN MATCHER
    matcher = Matcher(nlp.vocab)
    loaded_m: List[str] = []
    m_count = 0
    rule_labels: Dict[str, str] = {}
    token_auto_idx = 0

    if not mpaths:
        raise RuntimeError(f"No matcher.json files found (themes={themes or ['(none)']}) under {themes_root}")

    for p in mpaths:
        arr = _read_json_array(p)
        added_here = 0
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            raw_label = str(obj.get("label", "")).strip()
            label = _canon_type(raw_label) or raw_label
            patterns_raw = obj.get("patterns")
            if patterns_raw and isinstance(patterns_raw, list):
                pattern_list = []
                for pat in patterns_raw:
                    if isinstance(pat, list):
                        pattern_list.append(pat)
                if not pattern_list:
                    continue
            else:
                pattern = obj.get("pattern")
                if not isinstance(pattern, list):
                    continue
                pattern_list = [pattern]
            if not label:
                continue
            rule_name = str(obj.get("name", "")).strip()
            if not rule_name:
                token_auto_idx += 1
                safe_label = label.lower().replace(" ", "_")
                rule_name = f"{safe_label}_rule_{token_auto_idx}"
            try:
                matcher.add(rule_name, pattern_list)
                m_count += 1
                added_here += 1
                rule_labels[rule_name] = label
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
        dep_auto_idx = 0
        for p in dpaths:
            arr = _read_json_array(p)
            added_here = 0
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                raw_label = str(obj.get("label", "")).strip()
                label = _canon_type(raw_label) or raw_label
                patterns_raw = obj.get("patterns")
                if patterns_raw and isinstance(patterns_raw, list):
                    raw_patterns = [pat for pat in patterns_raw if isinstance(pat, list) and all(isinstance(x, dict) for x in pat)]
                else:
                    pat = obj.get("pattern")
                    if not isinstance(pat, list) or not all(isinstance(x, dict) for x in pat):
                        raise RuntimeError(f"Dep pattern in {p} for label={label} must be a List[dict] (no 'nodes/edges').")
                    raw_patterns = [pat]
                if not raw_patterns or not label:
                    continue

                expanded_batches: List[List[dict]] = []
                for pat in raw_patterns:
                    expanded = _expand_optional_nodes(pat)
                    expanded_batches.extend(expanded)
                if not expanded_batches:
                    continue
                rule_name = str(obj.get("name", "")).strip()
                if not rule_name:
                    dep_auto_idx += 1
                    safe_label = label.lower().replace(" ", "_")
                    rule_name = f"{safe_label}_dep_rule_{dep_auto_idx}"
                try:
                    depmatcher.add(rule_name, expanded_batches)  # spaCy ждёт List[List[dict]]
                    d_count += 1
                    added_here += 1
                    rule_labels[rule_name] = label
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
    return matcher, depmatcher, boosts, rule_labels, loaded_paths

# -*- coding: utf-8 -*-
"""
spacy_loader.py — единая точка загрузки spaCy и JSON-паттернов.

Функции:
- load_spacy_model() -> (nlp, dep_enabled, model_name)
- load_spacy_patterns(nlp, themes_root, theme_override)
    -> (Matcher, DependencyMatcher, type_boosts, registry)

registry — это список словарей вида:
  { "type": "Result", "engine": "token"|"dep",
    "source": "rules/themes/biomed/patterns/matcher.json",
    "index": 12,
    "key": "Result||src=rules/themes/biomed/patterns/matcher.json#12||engine=token" }

S1 сможет:
- класть registry в шапку s1_graph.json как pattern_sources (уникальные source),
- писать matched_rules в каждый узел (по возврату match_id → key).
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.matcher import Matcher, DependencyMatcher


# ---------- helpers ----------

def _read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _canon_type(label: str) -> Optional[str]:
    NODE_TYPES = [
        "Input Fact", "Hypothesis", "Experiment", "Technique",
        "Result", "Dataset", "Analysis", "Conclusion"
    ]
    TYPE_CANON = {t.lower().replace(" ", ""): t for t in NODE_TYPES}
    k = (label or "").lower().replace(" ", "")
    return TYPE_CANON.get(k)


def _mk_rule_key(label_type: str, source_rel: str, idx: int, engine: str) -> str:
    # Пример: "Result||src=rules/themes/biomed/patterns/matcher.json#12||engine=token"
    return f"{label_type}||src={source_rel}#{idx}||engine={engine}"


def _relpath_for_registry(themes_root: Path, p: Path) -> str:
    # Красивый относительный путь для registry/ключей
    try:
        return str(p.relative_to(themes_root.parent))
    except Exception:
        return str(p)


# ---------- public API ----------

def load_spacy_model() -> Tuple[spacy.language.Language, bool, str]:
    """
    Пытаемся загрузить модель из ENV (SPACY_MODEL) или en_core_web_sm.
    Если не получилось — fallback на spacy.blank('en').
    Возвращаем (nlp, dep_enabled, model_name).
    """
    name = os.getenv("SPACY_MODEL", "en_core_web_sm")
    try:
        nlp = spacy.load(name, disable=["ner", "textcat"])
        dep_enabled = nlp.has_pipe("parser")
        return nlp, dep_enabled, name
    except Exception:
        nlp = spacy.blank("en")
        if not nlp.has_pipe("sentencizer"):
            nlp.add_pipe("sentencizer")
        return nlp, False, f"blank:en (fallback; missing {name})"


def load_spacy_patterns(
        nlp: spacy.language.Language,
        themes_root: str,
        theme_override: Optional[List[str]] = None
) -> Tuple[Matcher, DependencyMatcher, Dict[str, float], List[Dict[str, Any]]]:
    """
    Загружает паттерны для spaCy Matcher/DependencyMatcher из themes/<topic>/patterns.
    Всегда подключает themes/common/patterns. Для выбранных тем добавляет их паттерны.
    Также собирает type_boosts из lexicon.json (common + темы).

    Форматы:
      - matcher.json:    [ { "label":"Result",   "pattern":[{"LOWER":"increase"}, ...] }, ... ]
      - depmatcher.json: [ { "label":"Technique","pattern":{"nodes":[...], "edges":[...] } }, ... ]
      - lexicon.json:    { "type_boosts": {"Result":1.03, "Technique":1.02} }

    Возвращает:
      (matcher, depmatcher, type_boosts, registry)
    """
    matcher = Matcher(nlp.vocab)
    depmatcher = DependencyMatcher(nlp.vocab)
    registry: List[Dict[str, Any]] = []

    root = Path(themes_root or "/app/rules/themes")
    pattern_roots: List[Path] = []

    topics = list(theme_override or [])
    for t in topics:
        pattern_roots.append(root / t / "patterns")
    # common — всегда
    pattern_roots.append(root / "common" / "patterns")

    # Загрузка matcher.json / depmatcher.json с регистрацией ключей
    for pr in pattern_roots:
        if not pr.exists():
            continue

        # token matcher
        mfile = pr / "matcher.json"
        if mfile.exists():
            try:
                items = _read_json(mfile)
                if isinstance(items, list):
                    for idx, it in enumerate(items):
                        label = _canon_type(str(it.get("label", "")))
                        pattern = it.get("pattern")
                        if not label or not pattern:
                            continue
                        key = _mk_rule_key(
                            label_type=label,
                            source_rel=_relpath_for_registry(root, mfile),
                            idx=idx,
                            engine="token",
                        )
                        matcher.add(key, [pattern])
                        registry.append({
                            "type": label,
                            "engine": "token",
                            "source": _relpath_for_registry(root, mfile),
                            "index": idx,
                            "key": key
                        })
            except Exception:
                # не валим пайплайн из-за одного файла
                pass

        # dependency matcher
        dfile = pr / "depmatcher.json"
        if dfile.exists():
            try:
                items = _read_json(dfile)
                if isinstance(items, list):
                    for idx, it in enumerate(items):
                        label = _canon_type(str(it.get("label", "")))
                        pat = it.get("pattern")
                        if not label or not pat:
                            continue
                        key = _mk_rule_key(
                            label_type=label,
                            source_rel=_relpath_for_registry(root, dfile),
                            idx=idx,
                            engine="dep",
                        )
                        depmatcher.add(key, [pat])
                        registry.append({
                            "type": label,
                            "engine": "dep",
                            "source": _relpath_for_registry(root, dfile),
                            "index": idx,
                            "key": key
                        })
            except Exception:
                pass

    # Загрузка бустов типов из lexicon.json
    type_boosts: Dict[str, float] = {}
    # shared/common
    for p in [root / "shared-lexicon.json", root / "_shared" / "lexicon.json", root / "common" / "lexicon.json"]:
        if p.exists():
            try:
                lx = _read_json(p)
                for k, v in (lx.get("type_boosts") or {}).items():
                    lab = _canon_type(str(k))
                    if lab:
                        type_boosts[lab] = float(v)
            except Exception:
                pass
    # per-topic
    for t in topics:
        p = root / t / "lexicon.json"
        if p.exists():
            try:
                lx = _read_json(p)
                for k, v in (lx.get("type_boosts") or {}).items():
                    lab = _canon_type(str(k))
                    if lab:
                        type_boosts[lab] = float(v)
            except Exception:
                pass

    return matcher, depmatcher, type_boosts, registry

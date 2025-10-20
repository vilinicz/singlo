# -*- coding: utf-8 -*-
"""
themes_router.py — preload реестр тем и авто-детекция тем по s0 (новый формат: s0["sentences"])

Структура на диске:
  /app/rules/themes/<topic>/
      patterns/matcher.json
      patterns/depmatcher.json (опц.)
      lexicon.json (опц.)
      triggers.json  <-- простые триггеры для detect_topics

Пример triggers.json (biomed):
{
  "any": [
    "\\bp\\s*(?:<|=)\\s*0?\\.\\d+",
    "\\bfisher(?:'s)?\\b",
    "\\bspearman(?:'s)?\\b",
    "\\bregression\\b",
    "\\bcohort\\b|\\bparticipants?\\b|\\bsubjects?\\b",
    "\\b(l\\.\\s*[a-z]+|lactobacillus|streptococcus)\\b"
  ],
  "intro_bias": ["\\baims?\\b|\\bobjective[s]?\\b"],
  "methods_bias": ["\\bmaterials? and methods?\\b|\\bstatistical analysis\\b"]
}

Логика detect_topics:
- Склеиваем видимое содержимое из s0["sentences"] (text, с лимитами).
- Для каждой темы проверяем её triggers.json:
    * если совпало что-то из any → +2 очка
    * если попадания в INTRO / METHODS (по section_hint) → дополнительные +1 за каждое
- Возвращаем темы с суммой >= threshold (по умолчанию 2). Если пусто — [].
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# -------- preload --------

## Preload available themes and their triggers.json into a registry.
def preload(themes_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """Сканируем themes/* и собираем реестр: {topic: {"path": Path, "triggers": {...}}}"""
    root = Path(themes_dir)
    registry: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return registry
    for td in root.iterdir():
        if not td.is_dir():
            continue
        topic = td.name.strip()
        triggers_path = td / "triggers.json"
        triggers = None
        if triggers_path.exists():
            try:
                triggers = json.loads(triggers_path.read_text(encoding="utf-8"))
            except Exception:
                triggers = None
        registry[topic] = {
            "path": td,
            "triggers": triggers or {}
        }
    return registry


# -------- helpers: compile patterns --------

## Compile a list of regex strings safely; ignore invalid entries.
def _compile_list(lst: Optional[List[str]]) -> List[re.Pattern]:
    out: List[re.Pattern] = []
    for s in (lst or []):
        try:
            out.append(re.compile(s, flags=re.I))
        except Exception:
            # игнорируем кривые регексы
            continue
    return out


## Heuristic score of a theme against S0 sentences using triggers.
def _topic_score_for_sentences(sents: List[dict], trig: Dict[str, Any]) -> int:
    """
    Считаем баллы по триггерам темы.
    any → +2 за любое совпадение
    intro_bias/methods_bias → +1 за совпадение в соответствующем section_hint
    """
    any_re = _compile_list(trig.get("any"))
    intro_re = _compile_list(trig.get("intro_bias"))
    meth_re = _compile_list(trig.get("methods_bias"))

    score = 0
    # ограничим объём для скорости
    # берём до 400 предложений, короткая выжимка
    for i, s in enumerate(sents[:400]):
        txt = (s.get("text") or "")
        if not txt:
            continue
        shint = (s.get("section_hint") or "OTHER").upper()

        # any
        if any_re and any(r.search(txt) for r in any_re):
            score += 2

        # biases
        if intro_re and shint == "INTRO":
            if any(r.search(txt) for r in intro_re):
                score += 1
        if meth_re and shint in {"METHODS", "OTHER", "RESULTS"}:
            # часто статистику пишут в METHODS/OTHER/RESULTS
            if any(r.search(txt) for r in meth_re):
                score += 1

        # лёгкий ранний выход
        if score >= 6:
            break
    return score


## Pick likely topics (themes) using triggers and a simple threshold.
def detect_topics(s0: Dict[str, Any], registry: Optional[Dict[str, Dict[str, Any]]],
                  *, threshold: int = 2) -> List[str]:
    """
    Определяет подходящие темы. Возвращает список тем без 'common'.
    Работает с новым S0: s0["sentences"].
    """
    if not registry:
        return []

    sents = [s for s in (s0.get("sentences") or []) if isinstance(s, dict) and s.get("text")]
    if not sents:
        return []

    scores: List[Tuple[str, int]] = []

    for topic, meta in registry.items():
        if topic == "common":
            continue
        trig = meta.get("triggers") or {}
        sc = _topic_score_for_sentences(sents, trig)
        if sc >= threshold:
            scores.append((topic, sc))

    # ранжируем и возвращаем имена тем (без common)
    scores.sort(key=lambda kv: -kv[1])
    return [t for t, _ in scores]

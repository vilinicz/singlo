# -*- coding: utf-8 -*-
"""
themes_router.py — быстрый роутер тем (top-k) с инвертированным индексом и ранним стопом.

Ключевые идеи:
- При preload() читаем themes/*/triggers.yaml и строим реестр:
  - plain_index: token(str) -> set(theme)
  - registry[theme]: {must[], should_pairs[(tok,wt)], negative_pairs[(tok,wt)], threshold}
  - regex buckets: для should/negative с regex-токенами
- route_themes() сначала собирает кандидатов по пересечению plain-токенов (top-M),
  затем детально скорит только их; ранний стоп: как только есть k тем со score>=stop_threshold — выходим.

API:
  preload(themes_root) -> ThemeRegistry
  route_themes(s0, registry, global_topk=2, stop_threshold=1.8, override=None) -> [ThemeScore]

Ручной override: список имён тем -> возвращаем сразу их.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import yaml


# -------- data structures --------

@dataclass
class Triggers:
    name: str
    must: List[str] = field(default_factory=list)
    should_plain: List[Tuple[str, float]] = field(default_factory=list)
    should_regex: List[Tuple[re.Pattern, float]] = field(default_factory=list)
    neg_plain: List[Tuple[str, float]] = field(default_factory=list)
    neg_regex: List[Tuple[re.Pattern, float]] = field(default_factory=list)
    threshold: float = 0.0


@dataclass
class ThemeRegistry:
    themes_root: Path
    themes: Dict[str, Triggers]  # name -> triggers
    plain_index: Dict[str, Set[str]]  # token -> {theme,...}


@dataclass
class ThemeScore:
    name: str
    score: float
    threshold: float
    chosen: bool


# -------- public API --------

def preload(themes_root: str | Path) -> ThemeRegistry:
    root = Path(themes_root)
    themes: Dict[str, Triggers] = {}
    plain_index: Dict[str, Set[str]] = {}

    for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
        tname = tdir.name
        tfile = tdir / "triggers.yaml"
        if not tfile.exists():
            continue
        cfg = yaml.safe_load(tfile.read_text(encoding="utf-8")) or {}
        name = cfg.get("name") or tname
        threshold = float(cfg.get("threshold", 0.0))
        must = [str(x) for x in (cfg.get("must") or [])]

        def split_pairs(lst):
            plain, regex = [], []
            for it in (lst or []):
                tok, wt = it[0], float(it[1])
                if _looks_regex(tok):
                    try:
                        regex.append((re.compile(tok, re.IGNORECASE | re.MULTILINE), wt))
                    except re.error:
                        # пропускаем битые шаблоны
                        continue
                else:
                    plain.append((tok.lower(), wt))
            return plain, regex

        sh_plain, sh_regex = split_pairs(cfg.get("should"))
        ng_plain, ng_regex = split_pairs(cfg.get("negative"))

        trig = Triggers(
            name=name,
            must=must,
            should_plain=sh_plain,
            should_regex=sh_regex,
            neg_plain=ng_plain,
            neg_regex=ng_regex,
            threshold=threshold,
        )
        themes[name] = trig

        # наполняем инвертированный индекс plain-токенов
        for tok, _ in sh_plain:
            plain_index.setdefault(tok, set()).add(name)
        for tok, _ in ng_plain:
            plain_index.setdefault(tok, set()).add(name)  # negative тоже как сигнал присутствия

    return ThemeRegistry(root, themes, plain_index)


def route_themes(
        s0: Dict,
        registry: ThemeRegistry,
        global_topk: int = 2,
        stop_threshold: float = 1.8,  # ранний стоп: «достаточно высокий» скор
        max_candidates: int = 20,  # ограничиваем детальную проверку до top-M
        override: Optional[List[str]] = None,
) -> List[ThemeScore]:
    """
    Возвращает top-k тем. Не перебирает все темы: сначала собирает кандидатов из plain-индекса.
    """
    # 0) override
    if override:
        chosen = []
        for nm in override:
            if nm in registry.themes:
                chosen.append(ThemeScore(nm, 999.0, 0.0, True))
        return chosen

    # 1) витрина
    text = build_showcase_text(s0)
    text_lower = text.lower()

    # 2) кандидаты: bag of tokens (plain)
    hits: Dict[str, int] = {}  # theme -> count of intersected tokens
    # простая токенизация по словам/биграммам (минимум): для plain-токенов достаточно подстроки
    # пройдёмся по всем токенам индекса, но отфильтруем дешёво — только те, что реально встречаются
    # Чтобы не перебирать все ключи, извлечём слова из текста и проверим их в индексе:
    vocab = set(_quick_tokens(text_lower))
    for tok in vocab:
        if tok in registry.plain_index:
            for th in registry.plain_index[tok]:
                hits[th] = hits.get(th, 0) + 1

    # 3) грубая приоритизация кандидатов и отсечение
    candidates = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)[:max_candidates]
    if not candidates:
        # fallback: если ни одного plain-хита (редко), проверим хотя бы первые N тем детально
        candidates = [(name, 0) for name in list(registry.themes.keys())[:max_candidates]]

    # 4) детальный скоринг только по кандидатам, с ранним стопом
    picked: List[ThemeScore] = []
    best_so_far: List[ThemeScore] = []

    for name, _cnt in candidates:
        trg = registry.themes.get(name)
        if not trg:
            continue
        score, passed = _score_theme_detail(text, trg)
        chosen = passed and (score >= trg.threshold)
        ts = ThemeScore(name=name, score=score, threshold=trg.threshold, chosen=chosen)
        if chosen:
            best_so_far.append(ts)
            # ранний стоп: как только у нас есть k тем со скором >= stop_threshold — хватит
            if len([x for x in best_so_far if x.score >= stop_threshold]) >= global_topk:
                break

    if not best_so_far:
        # если ничего не прошло пороги — возьмём лучшую по детальному скору среди кандидатов
        fallback_best = None
        for name, _ in candidates:
            trg = registry.themes.get(name)
            if not trg: continue
            score, passed = _score_theme_detail(text, trg)
            if (fallback_best is None) or (score > fallback_best.score):
                fallback_best = ThemeScore(name, score, trg.threshold, True)
        if fallback_best:
            picked = [fallback_best]
    else:
        picked = sorted(best_so_far, key=lambda x: x.score, reverse=True)[:global_topk]

    return picked


def select_rule_files(
        themes_root: str | Path,
        chosen: List[ThemeScore],
        common_path: str | Path | None = None,
) -> Dict[str, List[Path]]:
    root = Path(themes_root)
    files: Dict[str, List[Path]] = {"rules": [], "lexicons": []}

    # common.yaml: либо явно указали, либо ищем у родителя themes_root
    if common_path is not None:
        cp = Path(common_path)
        if cp.exists():
            files["rules"].append(cp)
    else:
        parent = root.parent
        cp = parent / "common.yaml"
        if cp.exists():
            files["rules"].append(cp)

    # shared-lexicon рядом с themes_root
    shared_lex = root / "shared-lexicon.yaml"
    if shared_lex.exists():
        files["lexicons"].append(shared_lex)

    for ts in chosen:
        tdir = root / ts.name
        r = tdir / "rules.yaml"
        if r.exists():
            files["rules"].append(r)
        lx = tdir / "lexicon.yaml"
        if lx.exists():
            files["lexicons"].append(lx)

    return files


# -------- internals --------

def build_showcase_text(s0: Dict) -> str:
    parts: List[str] = []
    for sec in s0.get("sections", []):
        nm = (sec.get("name") or "").lower()
        if nm.startswith("abstract") or nm.startswith("introduction"):
            parts.append(sec.get("text") or "")
    for c in (s0.get("captions") or [])[:10]:
        parts.append(c.get("text") or "")
    return _normalize(" ".join(parts))


def _normalize(t: str) -> str:
    t = (t or "").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = t.replace("–", "-").replace("—", "-").lower()
    return t


def _quick_tokens(t: str) -> List[str]:
    # примитивная токенизация слов ≥3 символов
    return re.findall(r"[a-z][a-z0-9\-]{2,}", t)


def _looks_regex(s: str) -> bool:
    return bool(re.search(r"[\\\[\]\(\)\.\+\*\?\|\^\$\{\}]", s))


def _contains_regex(rx: re.Pattern, text: str) -> bool:
    try:
        return rx.search(text) is not None
    except re.error:
        return False


def _score_theme_detail(text: str, trg: Triggers) -> Tuple[float, bool]:
    # must
    for tok in trg.must:
        if _looks_regex(tok):
            try:
                if re.search(tok, text, re.IGNORECASE | re.MULTILINE) is None:
                    return (0.0, False)
            except re.error:
                return (0.0, False)
        else:
            if tok.lower() not in text:
                return (0.0, False)

    score = 0.0
    # should plain
    for tok, wt in trg.should_plain:
        if tok in text:
            score += wt
    # negative plain
    for tok, wt in trg.neg_plain:
        if tok in text:
            score -= wt
    # should regex
    for rx, wt in trg.should_regex:
        if _contains_regex(rx, text):
            score += wt
    # negative regex
    for rx, wt in trg.neg_regex:
        if _contains_regex(rx, text):
            score -= wt

    return (score, True)

# -------- verbose scoring for debugging (exported) --------

def score_theme_detail_verbose(text: str, trg: Triggers) -> Dict[str, Any]:
    """
    Подробно считает скор темы и возвращает matched-триггеры.
    Возвращает:
      {
        "score": float,
        "passed_must": bool,
        "matched": [
           {"kind":"should","mode":"plain","token":"cohort","weight":1.1},
           {"kind":"negative","mode":"regex","token":"eigen(value|vector)","weight":1.0},
           ...
        ],
        "unmet_must": ["..."],                # если must не прошли
        "top_triggers": [ ... ]               # отсортированные matched по |weight|, сначала should, затем negative
      }
    """
    out = {"score": 0.0, "passed_must": True, "matched": [], "unmet_must": [], "top_triggers": []}

    # must
    for tok in (trg.must or []):
        if _looks_regex(tok):
            try:
                ok = re.search(tok, text, re.IGNORECASE | re.MULTILINE) is not None
            except re.error:
                ok = False
        else:
            ok = (tok.lower() in text)
        if not ok:
            out["passed_must"] = False
            out["unmet_must"].append(tok)

    # если must завалены, дальше всё равно посчитаем — полезно для отладки
    score = 0.0

    # should plain
    for tok, wt in (trg.should_plain or []):
        if tok in text:
            score += wt
            out["matched"].append({"kind": "should", "mode": "plain", "token": tok, "weight": wt})

    # negative plain
    for tok, wt in (trg.neg_plain or []):
        if tok in text:
            score -= wt
            out["matched"].append({"kind": "negative", "mode": "plain", "token": tok, "weight": wt})

    # should regex
    for rx, wt in (trg.should_regex or []):
        if _contains_regex(rx, text):
            score += wt
            out["matched"].append({"kind": "should", "mode": "regex", "token": rx.pattern, "weight": wt})

    # negative regex
    for rx, wt in (trg.neg_regex or []):
        if _contains_regex(rx, text):
            score -= wt
            out["matched"].append({"kind": "negative", "mode": "regex", "token": rx.pattern, "weight": wt})

    # топ-триггеры: сначала should по весу, потом negative по весу
    shoulds = [m for m in out["matched"] if m["kind"] == "should"]
    negs   = [m for m in out["matched"] if m["kind"] == "negative"]
    shoulds.sort(key=lambda m: float(m["weight"]), reverse=True)
    negs.sort(key=lambda m: float(m["weight"]), reverse=True)
    out["top_triggers"] = shoulds[:8] + negs[:8]

    out["score"] = float(score)
    return out


def explain_themes_for_debug(s0: Dict, registry: ThemeRegistry, chosen: List[ThemeScore]) -> List[Dict[str, Any]]:
    """
    Возвращает подробный блок для s1_debug:
    [
      {
        "name": "biomed",
        "score": 2.7,
        "threshold": 1.6,
        "chosen": true,
        "top_triggers": [...],
        "unmet_must": [...],
        "matched_count": 5
      },
      ...
    ]
    """
    text = build_showcase_text(s0)
    out = []
    for t in (chosen or []):
        trg = registry.themes.get(t.name)
        if not trg:
            out.append({"name": t.name, "score": t.score, "threshold": t.threshold, "chosen": t.chosen})
            continue
        det = score_theme_detail_verbose(text, trg)
        out.append({
            "name": t.name,
            "score": t.score,
            "threshold": t.threshold,
            "chosen": t.chosen,
            "top_triggers": det.get("top_triggers", []),
            "unmet_must": det.get("unmet_must", []),
            "matched_count": len(det.get("matched", [])),
        })
    return out

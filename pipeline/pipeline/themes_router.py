# -*- coding: utf-8 -*-
"""
themes_router.py — быстрый роутер тем (top-k) с инвертированным индексом, ранним стопом и расширенной отладкой.

Публичный API (совместим с прежним):
  preload(themes_root) -> ThemeRegistry
  route_themes(s0, registry, global_topk=2, stop_threshold=1.8, override=None) -> List[ThemeScore]
  select_rule_files(themes_root, chosen, common_path=None) -> Dict[str, List[Path]]
  explain_themes_for_debug(s0, registry, chosen) -> List[Dict[str, Any]]

Новое (для отладки и s1_debug):
  route_themes_with_candidates(s0, registry, ...) -> (chosen: List[ThemeScore], candidates: List[ThemeScore])
  explain_routing_full(s0, registry, chosen, candidates, top_n=5) -> Dict[str, Any]
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

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
    hits: int = 0        # грубое количество plain-хитов (для приоритизации кандидатов)
    passed_must: bool = True


# -------- public API --------

def preload(themes_root: str | Path) -> ThemeRegistry:
    root = Path(themes_root)
    themes: Dict[str, Triggers] = {}
    plain_index: Dict[str, Set[str]] = {}

    for tfile in sorted(root.rglob("triggers.yaml")):
        tdir = tfile.parent
        if not tfile.exists():
            continue
        cfg = yaml.safe_load(tfile.read_text(encoding="utf-8")) or {}
        name = (cfg.get("name") or str(tdir.relative_to(root)).replace("\\", "/"))
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
                        continue
                else:
                    plain.append((_normalize_token(tok), wt))
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

        # инвертированный индекс по plain-токенам should/negative
        for tok, _ in sh_plain:
            plain_index.setdefault(tok, set()).add(name)
        for tok, _ in ng_plain:
            plain_index.setdefault(tok, set()).add(name)

    return ThemeRegistry(root, themes, plain_index)


def route_themes(
        s0: Dict,
        registry: ThemeRegistry,
        global_topk: int = 2,
        stop_threshold: float = 1.8,
        max_candidates: int = 20,
        override: Optional[List[str]] = None,
) -> List[ThemeScore]:
    """
    Быстрая маршрутизация: возвращает только выбранные темы (совместимо со старым кодом).
    """
    chosen, _cands = route_themes_with_candidates(
        s0, registry, global_topk=global_topk, stop_threshold=stop_threshold, max_candidates=max_candidates, override=override
    )
    return chosen


def select_rule_files(
        themes_root: str | Path,
        chosen: List[ThemeScore],
        common_path: str | Path | None = None,
) -> Dict[str, List[Path]]:
    root = Path(themes_root)
    files: Dict[str, List[Path]] = {"rules": [], "lexicons": []}

    # common.yaml
    if common_path is not None:
        cp = Path(common_path)
        if cp.exists():
            files["rules"].append(cp)
    else:
        parent = root.parent
        cp = parent / "common.yaml"
        if cp.exists():
            files["rules"].append(cp)

    # shared-lexicon (две возможные локации)
    shared_candidates = [
        root / "shared-lexicon.yaml",
        root / "_shared" / "lexicon.yaml"
    ]
    seen = set()
    for p in shared_candidates:
        if p.exists():
            if p.resolve() not in seen:
                files["lexicons"].append(p)
                seen.add(p.resolve())

    # тематические пакеты
    for ts in (t for t in chosen if getattr(t, "chosen", False)):
        tdir = root / ts.name
        r = tdir / "rules.yaml"
        if r.exists():
            files["rules"].append(r)
        lx = tdir / "lexicon.yaml"
        if lx.exists():
            files["lexicons"].append(lx)

    return files


# -------- new: routing with candidates (for debug) --------

def route_themes_with_candidates(
        s0: Dict,
        registry: ThemeRegistry,
        global_topk: int = 2,
        stop_threshold: float = 1.8,
        max_candidates: int = 20,
        override: Optional[List[str]] = None,
) -> Tuple[List[ThemeScore], List[ThemeScore]]:
    """
    Возвращает (chosen, candidates). Candidates — просмотренные темы (с hits/score/passed_must).
    """
    # 0) override
    if override:
        chosen = []
        for nm in override:
            if nm in registry.themes:
                chosen.append(ThemeScore(nm, 999.0, 0.0, True, hits=999))
        return chosen, chosen

    # 1) витрина
    text = build_showcase_text(s0)
    text_lower = text.lower()

    # 2) кандидаты: пересечение с plain-индексом
    hits: Dict[str, int] = {}
    vocab = set(_quick_tokens(text_lower))
    for tok in vocab:
        if tok in registry.plain_index:
            for th in registry.plain_index[tok]:
                hits[th] = hits.get(th, 0) + 1

    candidates = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)[:max_candidates]
    if not candidates:
        candidates = [(name, 0) for name in list(registry.themes.keys())[:max_candidates]]

    # 3) детальный скоринг
    chosen: List[ThemeScore] = []
    detailed: List[ThemeScore] = []
    early_passed = 0

    for name, cnt in candidates:
        trg = registry.themes.get(name)
        if not trg:
            continue
        score, passed, _ = _score_theme_detail_verbose(text, trg)
        ts = ThemeScore(name=name, score=score, threshold=trg.threshold, chosen=False, hits=int(cnt), passed_must=passed)
        if passed and (score >= trg.threshold):
            ts.chosen = True
            chosen.append(ts)
            if score >= stop_threshold:
                early_passed += 1
        detailed.append(ts)
        if early_passed >= global_topk:
            break

    if not chosen and detailed:
        # для диагностики оставим «лучшего» как невыбранного
        pass

    # итог: top-k из выбранных
    chosen = sorted([t for t in chosen if t.chosen], key=lambda x: x.score, reverse=True)[:global_topk]
    return chosen, detailed


# -------- internals --------

def build_showcase_text(s0: Dict) -> str:
    parts: List[str] = []
    for sec in s0.get("sections", []):
        nm = (sec.get("name") or "").lower()
        if nm.startswith("abstract") or nm.startswith("introduction"):
            parts.append(sec.get("text") or "")
        # добавим ещё вероятно-важные хвосты, если есть
        if ("conclusion" in nm) or ("discussion" in nm):
            parts.append(sec.get("text") or "")
    for c in (s0.get("captions") or [])[:12]:
        parts.append(c.get("text") or "")
    return _normalize(" ".join(parts))


def _normalize(t: str) -> str:
    t = (t or "").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = t.replace("–", "-").replace("—", "-").lower()
    return t


def _normalize_token(tok: str) -> str:
    tok = (tok or "").strip().lower()
    tok = re.sub(r"[^a-z0-9\-]+", "", tok)
    return tok


def _quick_tokens(t: str) -> List[str]:
    # простые токены: слова ≥3 символов и термины с дефисом/цифрой
    return re.findall(r"[a-z][a-z0-9\-]{2,}", t)


def _looks_regex(s: str) -> bool:
    return bool(re.search(r"[\\\[\]\(\)\.\+\*\?\|\^\$\{\}]", s))


def _contains_regex(rx: re.Pattern, text: str) -> bool:
    try:
        return rx.search(text) is not None
    except re.error:
        return False


def _score_theme_detail(text: str, trg: Triggers) -> Tuple[float, bool]:
    for tok in (trg.must or []):
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
    for tok, wt in (trg.should_plain or []):
        if tok in text:
            score += wt
    for tok, wt in (trg.neg_plain or []):
        if tok in text:
            score -= wt
    for rx, wt in (trg.should_regex or []):
        if _contains_regex(rx, text):
            score += wt
    for rx, wt in (trg.neg_regex or []):
        if _contains_regex(rx, text):
            score -= wt
    return (score, True)


# -------- verbose scoring for debugging (exported) --------

def score_theme_detail_verbose(text: str, trg: Triggers) -> Dict[str, Any]:
    """
    Совместимая версия: подробно считает скор и matched-триггеры.
    """
    det = _score_theme_detail_verbose(text, trg)[2]
    return det


def _score_theme_detail_verbose(text: str, trg: Triggers) -> Tuple[float, bool, Dict[str, Any]]:
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

    score = 0.0

    for tok, wt in (trg.should_plain or []):
        if tok in text:
            score += wt
            out["matched"].append({"kind": "should", "mode": "plain", "token": tok, "weight": wt})

    for tok, wt in (trg.neg_plain or []):
        if tok in text:
            score -= wt
            out["matched"].append({"kind": "negative", "mode": "plain", "token": tok, "weight": wt})

    for rx, wt in (trg.should_regex or []):
        if _contains_regex(rx, text):
            score += wt
            out["matched"].append({"kind": "should", "mode": "regex", "token": rx.pattern, "weight": wt})

    for rx, wt in (trg.neg_regex or []):
        if _contains_regex(rx, text):
            score -= wt
            out["matched"].append({"kind": "negative", "mode": "regex", "token": rx.pattern, "weight": wt})

    shoulds = [m for m in out["matched"] if m["kind"] == "should"]
    negs = [m for m in out["matched"] if m["kind"] == "negative"]
    shoulds.sort(key=lambda m: float(m["weight"]), reverse=True)
    negs.sort(key=lambda m: float(m["weight"]), reverse=True)
    out["top_triggers"] = shoulds[:8] + negs[:8]

    out["score"] = float(score)
    return score, out["passed_must"], out


def explain_themes_for_debug(s0: Dict, registry: ThemeRegistry, chosen: List[ThemeScore]) -> List[Dict[str, Any]]:
    """
    Совместимая функция: подробности только по выбранным темам.
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


# -------- new: full debug block (chosen + rejected top candidates) --------

def explain_routing_full(
    s0: Dict,
    registry: ThemeRegistry,
    chosen: List[ThemeScore],
    candidates: List[ThemeScore],
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Возвращает полный блок для s1_debug:
      {
        "chosen": [... как в explain_themes_for_debug ...],
        "rejected_top": [
           {"name":..., "score":..., "threshold":..., "hits":..., "passed_must":..., "top_triggers":[...], "unmet_must":[...]},
           ...
        ]
      }
    """
    text = build_showcase_text(s0)
    chosen_block = explain_themes_for_debug(s0, registry, chosen)

    # невыбранные топ-кандидаты
    rejected = [c for c in (candidates or []) if not c.chosen]
    rejected = sorted(rejected, key=lambda x: (x.score, x.hits), reverse=True)[:top_n]

    rej_block = []
    for r in rejected:
        trg = registry.themes.get(r.name)
        if not trg:
            rej_block.append({
                "name": r.name, "score": r.score, "threshold": r.threshold,
                "hits": r.hits, "passed_must": r.passed_must
            })
            continue
        det = score_theme_detail_verbose(text, trg)
        rej_block.append({
            "name": r.name,
            "score": r.score,
            "threshold": r.threshold,
            "hits": r.hits,
            "passed_must": r.passed_must,
            "top_triggers": det.get("top_triggers", []),
            "unmet_must": det.get("unmet_must", []),
            "matched_count": len(det.get("matched", [])),
        })

    return {"chosen": chosen_block, "rejected_top": rej_block}

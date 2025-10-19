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
    hits: int = 0  # грубое количество plain-хитов (для приоритизации кандидатов)
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
        s0, registry, global_topk=global_topk, stop_threshold=stop_threshold, max_candidates=max_candidates,
        override=override
    )
    return chosen


# legacy
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


def select_pattern_files(
        themes_root: str | Path,
        chosen: List[ThemeScore],
) -> Dict[str, List[Path]]:
    """
    Возвращает json-файлы с паттернами spaCy для выбранных тем + common:
      {
        "matcher": [Path(.../common/patterns/matcher.json), Path(.../<topic>/patterns/matcher.json), ...],
        "depmatcher": [...],
        "lexicons": [Path(.../common/lexicon.json?), Path(.../<topic>/lexicon.json?) ...]
      }
    """
    root = Path(themes_root)
    out: Dict[str, List[Path]] = {"matcher": [], "depmatcher": [], "lexicons": []}

    # common
    cpat = root / "common" / "patterns"
    if (cpat / "matcher.json").exists():
        out["matcher"].append(cpat / "matcher.json")
    if (cpat / "depmatcher.json").exists():
        out["depmatcher"].append(cpat / "depmatcher.json")
    # общий лексикон (опционально)
    for p in [root / "shared-lexicon.json", root / "_shared" / "lexicon.json", root / "common" / "lexicon.json"]:
        if p.exists(): out["lexicons"].append(p)

    # per-theme
    for ts in (t for t in chosen if getattr(t, "chosen", False)):
        tdir = root / ts.name
        pdir = tdir / "patterns"
        if (pdir / "matcher.json").exists():
            out["matcher"].append(pdir / "matcher.json")
        if (pdir / "depmatcher.json").exists():
            out["depmatcher"].append(pdir / "depmatcher.json")
        lx = tdir / "lexicon.json"
        if lx.exists(): out["lexicons"].append(lx)

    return out


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

        # 1) витрина (buckets)
    texts = build_showcase_buckets(s0)  # {"intro","results","rest","captions","all"}
    text_all = texts["all"]  # единая витрина для токенизации и индекса
    text_lower = text_all.lower()

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

    # 3) детальный скоринг по buckets
    chosen: List[ThemeScore] = []
    detailed: List[ThemeScore] = []
    early_passed = 0

    for name, cnt in candidates:
        trg = registry.themes.get(name)
        if not trg:
            continue
        score, passed, _ = _score_theme_detail_verbose_buckets(texts, trg)
        ts = ThemeScore(name=name, score=score, threshold=trg.threshold, chosen=False, hits=int(cnt),
                        passed_must=passed)
        if passed and (score >= trg.threshold):
            ts.chosen = True
            chosen.append(ts)
            if score >= stop_threshold:
                early_passed += 1
        detailed.append(ts)
        if early_passed >= global_topk:
            break

    # итог: top-k из выбранных
    chosen = sorted([t for t in chosen if t.chosen], key=lambda x: x.score, reverse=True)[:global_topk]
    return chosen, detailed


# -------- legacy: internals --------
def build_showcase_text(s0: Dict) -> str:
    # 1) Abstract/Intro + Discussion/Conclusions
    parts: List[str] = []
    for sec in s0.get("sections", []):
        nm = (sec.get("name") or "").lower()
        if nm.startswith("abstract") or nm.startswith("introduction"):
            parts.append(sec.get("text") or "")
        if ("conclusion" in nm) or ("discussion" in nm):
            parts.append(sec.get("text") or "")

    # 2) немного Methods/Results — первые 2 предложения (если есть разметка)
    for sec in s0.get("sections", []):
        nm = (sec.get("name") or "").lower()
        if ("materials and methods" in nm) or nm.startswith("materials") or nm == "methods":
            sents = []
            for blk in sec.get("blocks", []):
                for s in blk.get("sentences", [])[:2]:
                    sents.append(s.get("text") or "")
                    if len(sents) >= 2: break
                if len(sents) >= 2: break
            if sents: parts.append(" ".join(sents))
        if ("results" in nm):
            sents = []
            for blk in sec.get("blocks", []):
                for s in blk.get("sentences", [])[:2]:
                    sents.append(s.get("text") or "")
                    if len(sents) >= 2: break
                if len(sents) >= 2: break
            if sents: parts.append(" ".join(sents))

    # 3) captions
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


def _has_plain_token(tok: str, text: str) -> bool:
    # \b по ASCII-словам. В _normalize мы уже свели регистр/дефисы, так что этого достаточно.
    return re.search(rf"\b{re.escape(tok)}\b", text) is not None


def _score_theme_detail(text: str, trg: Triggers) -> Tuple[float, bool]:
    for tok in (trg.must or []):
        if _looks_regex(tok):
            try:
                if re.search(tok, text, re.IGNORECASE | re.MULTILINE) is None:
                    return (0.0, False)
            except re.error:
                return (0.0, False)
        else:
            if not _has_plain_token(tok.lower(), text):
                return (0.0, False)

    score = 0.0
    for tok, wt in (trg.should_plain or []):
        if _has_plain_token(tok, text):
            score += wt
    for tok, wt in (trg.neg_plain or []):
        if _has_plain_token(tok, text):
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
            ok = _has_plain_token(tok.lower(), text)
        if not ok:
            out["passed_must"] = False
            out["unmet_must"].append(tok)

    score = 0.0

    for tok, wt in (trg.should_plain or []):
        if _has_plain_token(tok, text):
            score += wt
            out["matched"].append({"kind": "should", "mode": "plain", "token": tok, "weight": wt})

    for tok, wt in (trg.neg_plain or []):
        if _has_plain_token(tok, text):
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


def _score_theme_detail_verbose_buckets(texts: Dict[str, str], trg: Triggers) -> Tuple[float, bool, Dict[str, Any]]:
    # как твой _score_theme_detail_verbose, но вместо одного текста — словарь витрин
    # scheme: all (база), intro (x1.10), results (x1.08), captions (x1.08)
    M_IN, M_RS, M_CP = 1.10, 1.08, 1.08

    def hit_plain(tok, t):
        return _has_plain_token(tok, t)

    def hit_rx(rx, t):
        return _contains_regex(rx, t)

    # must на общей витрине
    passed = True
    for tok in (trg.must or []):
        if _looks_regex(tok):
            ok = re.search(tok, texts["all"], re.IGNORECASE | re.MULTILINE) is not None
        else:
            ok = _has_plain_token(tok.lower(), texts["all"])
        if not ok:
            passed = False

    score = 0.0;
    matched = []

    def add_if(found, wt, kind, mode, token):
        nonlocal score, matched
        if found:
            score += wt
            matched.append({"kind": kind, "mode": mode, "token": token, "weight": wt})

    # plain
    for tok, wt in (trg.should_plain or []):
        add_if(hit_plain(tok, texts["all"]), wt, "should", "plain", tok)
        add_if(hit_plain(tok, texts["intro"]), wt * (M_IN - 1), "should", "plain", tok)
        add_if(hit_plain(tok, texts["results"]), wt * (M_RS - 1), "should", "plain", tok)
        add_if(hit_plain(tok, texts["captions"]), wt * (M_CP - 1), "should", "plain", tok)

    for tok, wt in (trg.neg_plain or []):
        add_if(hit_plain(tok, texts["all"]), -wt, "negative", "plain", tok)
        add_if(hit_plain(tok, texts["intro"]), -wt * (M_IN - 1), "negative", "plain", tok)
        add_if(hit_plain(tok, texts["results"]), -wt * (M_RS - 1), "negative", "plain", tok)
        add_if(hit_plain(tok, texts["captions"]), -wt * (M_CP - 1), "negative", "plain", tok)

    # regex
    for rx, wt in (trg.should_regex or []):
        add_if(hit_rx(rx, texts["all"]), wt, "should", "regex", rx.pattern)
        add_if(hit_rx(rx, texts["intro"]), wt * (M_IN - 1), "should", "regex", rx.pattern)
        add_if(hit_rx(rx, texts["results"]), wt * (M_RS - 1), "should", "regex", rx.pattern)
        add_if(hit_rx(rx, texts["captions"]), wt * (M_CP - 1), "should", "regex", rx.pattern)

    for rx, wt in (trg.neg_regex or []):
        add_if(hit_rx(rx, texts["all"]), -wt, "negative", "regex", rx.pattern)
        add_if(hit_rx(rx, texts["intro"]), -wt * (M_IN - 1), "negative", "regex", rx.pattern)
        add_if(hit_rx(rx, texts["results"]), -wt * (M_RS - 1), "negative", "regex", rx.pattern)
        add_if(hit_rx(rx, texts["captions"]), -wt * (M_CP - 1), "negative", "regex", rx.pattern)

    matched.sort(key=lambda m: float(m["weight"]), reverse=True)
    det = {"score": float(score), "passed_must": passed, "matched": matched, "top_triggers": matched[:8],
           "unmet_must": []}
    return score, passed, det


def explain_themes_for_debug(s0: Dict, registry: ThemeRegistry, chosen: List[ThemeScore]) -> List[Dict[str, Any]]:
    texts = build_showcase_buckets(s0)
    out = []
    for t in (chosen or []):
        trg = registry.themes.get(t.name)
        if not trg:
            out.append({"name": t.name, "score": t.score, "threshold": t.threshold, "chosen": t.chosen})
            continue
        det = _score_theme_detail_verbose_buckets(texts, trg)[2]
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


def build_showcase_buckets(s0: Dict) -> Dict[str, str]:
    intro, results, rest, caps = [], [], [], []
    for sec in s0.get("sections", []):
        nm = (sec.get("name") or "").lower()
        t = _normalize(sec.get("text") or "")
        if not t: continue
        if nm.startswith("abstract") or nm.startswith("introduction"):
            intro.append(t)
        elif "results" in nm:
            results.append(t)
        else:
            rest.append(t)
    for c in (s0.get("captions") or []):
        caps.append(_normalize(c.get("text") or ""))
    return {
        "intro": " ".join(intro),
        "results": " ".join(results),
        "rest": " ".join(rest),
        "captions": " ".join(caps),
        "all": " ".join([*intro, *results, *rest, *caps]),
    }


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
    texts = build_showcase_buckets(s0)
    chosen_block = explain_themes_for_debug(s0, registry, chosen)

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
        det = _score_theme_detail_verbose_buckets(texts, trg)[2]
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

# -*- coding: utf-8 -*-
"""
S1 (spaCy + data-driven patterns) → s1_graph.json / s1_debug.json

Вход теперь ТОЛЬКО плоский S0:
  s0["sentences"] = [
    {"text": "...", "page": 0, "bbox": [x0,y0,x1,y1], "section_hint": "INTRO|METHODS|RESULTS|DISCUSSION|REFERENCES|OTHER",
     "is_caption": false, "caption_type": ""|"Figure"|"Table"}
  ]

Паттерны: themes/<topic>/patterns/{matcher.json, depmatcher.json}, плюс общие themes/common/...
Выход: s1_graph.json (кандидаты узлов + рёбра внутри статьи), s1_debug.json (сводка)

Типы узлов (ровно 8):
  Input Fact, Hypothesis, Experiment, Technique, Result, Dataset, Analysis, Conclusion

Рёбра (базовые, внутри статьи):
  Technique → Experiment : uses
  Experiment → Result    : produces
  Result → Hypothesis    : supports|refutes (по polarity)
  Technique → Result     : uses        (если рядом)
  Dataset → Experiment   : feeds
  Dataset → Analysis     : feeds
  Analysis → Result      : informs

Fallback (если рёбер нет, а узлы есть):
  Result → Hypothesis : supports|refutes (1–2 ребра)
  Technique → (Experiment|Result) : uses (1–2 ребра)

Примечание:
- rules_path не используется (оставлен ради совместимости интерфейса).
- theme_override — список имён тем (['biomed','physics']) или None → подключаем только 'common'.
"""

from __future__ import annotations

import json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set

from spacy.tokens import Doc
from spacy.matcher import Matcher, DependencyMatcher  # type hints
from .spacy_loader import load_spacy_model, load_spacy_patterns

# ─────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────

NODE_TYPES = [
    "Input Fact", "Hypothesis", "Experiment", "Technique",
    "Result", "Dataset", "Analysis", "Conclusion"
]

TYPE_CANON = {t.lower().replace(" ", ""): t for t in NODE_TYPES}

SECTION_PRIORS = {
    "INTRO": {"Hypothesis": 1.10, "Input Fact": 1.05, "Conclusion": 1.00},
    "METHODS": {"Technique": 1.15, "Experiment": 1.12, "Dataset": 1.10, "Analysis": 1.05},
    "RESULTS": {"Result": 1.18, "Analysis": 1.08},
    "DISCUSSION": {"Conclusion": 1.15, "Hypothesis": 1.05, "Result": 1.03, "Analysis": 1.03},
    "REFERENCES": {},
    "OTHER": {"Input Fact": 1.02}
}

# базовые веса по типу (на случай равных матчей)
BASE_TYPE_WEIGHTS = {
    "Input Fact": 1.00,
    "Hypothesis": 1.12,
    "Experiment": 1.08,
    "Technique": 1.04,
    "Result": 1.15,
    "Dataset": 1.06,
    "Analysis": 1.05,
    "Conclusion": 1.10,
}

# штрафы за хеджинг
HEDGING = {
    "may", "might", "could", "appears", "appeared", "seems", "seemed", "likely", "tend", "tends", "potentially",
    "approximately", "suggest", "suggests", "suggested", "possibly", "probable", "plausible"
}
HEDGE_PENALTY = 0.08

# бонус за числовые факты в Result/Analysis/Dataset
NUMERIC_BONUS = 0.06

# пороги
CONF_NODE_MIN = 0.40
CONF_EDGE_MIN = 0.55

# полярность
POS_MARKERS = {
    "demonstrate", "demonstrates", "demonstrated", "show", "shows", "shown",
    "increase", "increases", "increased", "improve", "improves", "improved", "significant", "significantly",
    "supports", "fit", "fits", "fitted", "correlates", "reduce", "reduces", "reduced", "outperform", "outperforms"
}
NEG_MARKERS = {
    "no", "not", "lack", "lacks", "didnt", "didn't", "failed", "fails",
    "insignificant", "non-significant", "nonsignificant", "refute", "refutes", "refuted",
    "decrease", "decreases", "decreased", "worse", "worsened", "negative result", "null result"
}

LINK_WINDOW_SENTENCES = 2


# ─────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────

def _canon_type(label: str) -> Optional[str]:
    return TYPE_CANON.get(label.lower().replace(" ", ""))


def _contains_any(tokens_lower: List[str], vocab: Set[str]) -> bool:
    return any(t in vocab for t in tokens_lower)


def _has_numeric(text: str) -> bool:
    # быстрый признак «есть числа/проценты/p-values»
    return any(c.isdigit() for c in text) or "%" in text or " p<" in text.lower() or " p =" in text.lower()


def _polarity(text: str) -> str:
    t = text.lower()
    pos = any(w in t for w in POS_MARKERS)
    neg = any(w in t for w in NEG_MARKERS)
    if pos and not neg: return "positive"
    if neg and not pos: return "negative"
    if pos and neg: return "neutral"
    return "neutral"


def _hedge_penalty_from_doc(nlp_doc: Doc) -> float:
    tokens_lower = [t.text.lower() for t in nlp_doc]
    return HEDGE_PENALTY if _contains_any(tokens_lower, HEDGING) else 0.0


def _section_prior(imrad_hint: str, tname: str) -> float:
    return SECTION_PRIORS.get(imrad_hint or "OTHER", {}).get(tname, 1.0)


def _base_type_weight(tname: str) -> float:
    return BASE_TYPE_WEIGHTS.get(tname, 1.0)


def _sent_iter_flat(s0: Dict) -> Iterable[Tuple[int, Dict]]:
    """
    Итерирует плоский список предложений s0["sentences"].
    Возвращает (global_idx, sent_obj)
    """
    for i, s in enumerate(s0.get("sentences", [])):
        yield i, s


def _mk_rule_key(label_type: str, source_rel: str, idx: int, engine: str) -> str:
    # пример: "Result||src=themes/biomed/patterns/matcher.json#12||engine=token"
    return f"{label_type}||src={source_rel}#{idx}||engine={engine}"


def _parse_rule_key(vocab, match_id) -> dict:
    key = vocab.strings[match_id] if isinstance(match_id, int) else str(match_id)
    # ожидаем формат как в _mk_rule_key
    out = {"type": None, "source": None, "index": None, "engine": None, "raw": key}
    parts = key.split("||")
    if parts:
        out["type"] = parts[0]
    for p in parts[1:]:
        if p.startswith("src="):
            src = p[4:]
            if "#" in src:
                src_path, idx = src.rsplit("#", 1)
                out["source"] = src_path
                try:
                    out["index"] = int(idx)
                except:
                    out["index"] = idx
            else:
                out["source"] = src
        elif p.startswith("engine="):
            out["engine"] = p.split("=", 1)[1]
    return out


def _canon_type_from_key(key_type: str) -> str | None:
    # TYPE_CANON объявлен выше (как и раньше)
    return TYPE_CANON.get((key_type or "").lower().replace(" ", ""))


def _load_spacy_patterns(nlp,
                         themes_root: str,
                         theme_override: Optional[List[str]] = None
                         ) -> Tuple[Matcher, DependencyMatcher, Dict[str, float], List[dict]]:
    """
    Возвращает: matcher, depmatcher, type_boosts, registry
    registry: [{ "type": "Result", "engine":"token", "source":"themes/biomed/patterns/matcher.json", "index":12, "key":"..." }, ...]
    """
    matcher = Matcher(nlp.vocab)
    depmatcher = DependencyMatcher(nlp.vocab)
    registry: List[dict] = []

    themes_root_path = Path(themes_root or "/app/rules/themes")
    roots = []
    if theme_override:
        for t in theme_override:
            roots.append(themes_root_path / t / "patterns")
    roots.append(themes_root_path / "common" / "patterns")  # common всегда

    def relpath(p: Path) -> str:
        try:
            return str(p.relative_to(themes_root_path.parent))
        except Exception:
            return str(p)

    for r in roots:
        if not r.exists():
            continue
        # token matcher
        mfile = r / "matcher.json"
        if mfile.exists():
            try:
                items = json.loads(mfile.read_text(encoding="utf-8"))
                if isinstance(items, list):
                    for idx, it in enumerate(items):
                        label = _canon_type(str(it.get("label", "")))
                        pattern = it.get("pattern")
                        if not label or not pattern:
                            continue
                        key = _mk_rule_key(label, relpath(mfile), idx, "token")
                        matcher.add(key, [pattern])
                        registry.append(
                            {"type": label, "engine": "token", "source": relpath(mfile), "index": idx, "key": key})
                # else ignore
            except Exception:
                pass
        # dependency matcher
        dfile = r / "depmatcher.json"
        if dfile.exists():
            try:
                items = json.loads(dfile.read_text(encoding="utf-8"))
                if isinstance(items, list):
                    for idx, it in enumerate(items):
                        label = _canon_type(str(it.get("label", "")))
                        pat = it.get("pattern")
                        if not label or not pat:
                            continue
                        key = _mk_rule_key(label, relpath(dfile), idx, "dep")
                        depmatcher.add(key, [pat])
                        registry.append(
                            {"type": label, "engine": "dep", "source": relpath(dfile), "index": idx, "key": key})
            except Exception:
                pass

    # type boosts
    type_boosts: Dict[str, float] = {}
    for r in roots:
        lfile = r.parent / "lexicon.json"
        if lfile.exists():
            try:
                lex = json.loads(lfile.read_text(encoding="utf-8"))
                for k, v in (lex.get("type_boosts") or {}).items():
                    t = _canon_type(str(k))
                    if t:
                        type_boosts[t] = float(v)
            except Exception:
                pass

    return matcher, depmatcher, type_boosts, registry


# ─────────────────────────────────────────────────────────────
# Оценка и выбор типа
# ─────────────────────────────────────────────────────────────

def _score_sentence(nlp_doc: Doc,
                    text: str,
                    imrad_hint: str,
                    matcher: Matcher,
                    depmatcher: DependencyMatcher,
                    type_boosts: Dict[str, float],
                    dep_enabled: bool) -> Tuple[Optional[str], float, Dict[str, Any], List[dict]]:
    hits: Dict[str, float] = {}
    matched_rules: List[dict] = []

    # token patterns
    for match_id, start, end in matcher(nlp_doc):
        meta = _parse_rule_key(nlp_doc.vocab, match_id)
        tname = _canon_type_from_key(meta["type"]) or meta["type"]
        if tname in NODE_TYPES:
            hits[tname] = hits.get(tname, 0.0) + 1.0
            frag = nlp_doc[start:end].text
            meta_out = {**meta, "span": [start, end], "text": frag}
            matched_rules.append(meta_out)

    # dep patterns
    if dep_enabled:
        for match_id, (token_ids,) in depmatcher(nlp_doc):
            meta = _parse_rule_key(nlp_doc.vocab, match_id)
            tname = _canon_type_from_key(meta["type"]) or meta["type"]
            if tname in NODE_TYPES:
                hits[tname] = hits.get(tname, 0.0) + 1.2
                # соберём компактный фрагмент-окно
                toks = [nlp_doc[i].text for i in token_ids if 0 <= i < len(nlp_doc)]
                meta_out = {**meta, "tokens": toks}
                matched_rules.append(meta_out)

    if not hits:
        return None, 0.0, {"hits": {}}, matched_rules

    prior_mult = {t: SECTION_PRIORS.get(imrad_hint or "OTHER", {}).get(t, 1.0) for t in hits.keys()}
    base_mult = {t: BASE_TYPE_WEIGHTS.get(t, 1.0) for t in hits.keys()}
    theme_mult = {t: type_boosts.get(t, 1.0) for t in hits.keys()}
    hedge = HEDGE_PENALTY if any(tok.text.lower() in HEDGING for tok in nlp_doc) else 0.0
    num_bonus = NUMERIC_BONUS if _has_numeric(text) else 0.0

    scored: Dict[str, float] = {}
    for t, raw in hits.items():
        val = raw * prior_mult.get(t, 1.0) * base_mult.get(t, 1.0) * theme_mult.get(t, 1.0)
        if t in ("Result", "Analysis", "Dataset"):
            val += num_bonus
        val -= hedge
        scored[t] = val

    best_t, best_v = max(scored.items(), key=lambda kv: kv[1])
    dbg = {
        "hits": hits,
        "prior_mult": prior_mult,
        "base_mult": base_mult,
        "theme_mult": theme_mult,
        "hedge": hedge,
        "num_bonus": num_bonus,
        "scored": scored
    }
    return best_t, float(best_v), dbg, matched_rules


# ─────────────────────────────────────────────────────────────
# Построение узлов/рёбер
# ─────────────────────────────────────────────────────────────

def _node_id(doc_id: str, tname: str, idx: int) -> str:
    slug = doc_id.replace("/", "-").replace(":", "-")
    tslug = (tname or "").replace(" ", "")
    return f"{slug}:{tslug}:{idx:04d}"


def _mk_node(doc_id: str, idx: int, tname: str, text: str, conf: float,
             section_name: str, imrad_hint: str, s_idx: int,
             page: int, bbox: List[float], polarity: str) -> Dict[str, Any]:
    # coords (для фронта) — из bbox
    coords = []
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        coords = [{"x": float(x0), "y": float(y0), "w": float(x1) - float(x0), "h": float(y1) - float(y0)}]
    return {
        "id": _node_id(doc_id, tname, idx),
        "type": tname,
        "label": f"spacy/{tname}",
        "text": text,
        "conf": round(conf, 3),
        "polarity": polarity,
        "prov": {
            "section": section_name,
            "imrad": imrad_hint,
            "sec_idx": -1,
            "block_idx": -1,
            "sent_idx": s_idx,
        },
        "page": int(page) if page is not None else 0,
        "coords": coords,
        "bbox": list(bbox) if isinstance(bbox, (list, tuple)) else []
    }


def _distance(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    # на плоском списке меряем дистанцию по глобальному sent_idx
    sa = int(a["prov"]["sent_idx"])
    sb = int(b["prov"]["sent_idx"])
    return abs(sa - sb)


def _link_inside(doc_id: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []

    by_type: Dict[str, List[Dict[str, Any]]] = {t: [] for t in NODE_TYPES}
    for n in nodes:
        by_type.get(n["type"], []).append(n)

    def add_edge(src: Dict, dst: Dict, etype: str, conf: float):
        edges.append({
            "from": src["id"],
            "to": dst["id"],
            "type": etype,
            "conf": round(conf, 3),
            "prov": {
                "hint": "prox",
                "from": {"sec": src["prov"]["section"], "s": src["prov"]["sent_idx"]},
                "to": {"sec": dst["prov"]["section"], "s": dst["prov"]["sent_idx"]},
            }
        })

    def nearest(src_list: List[Dict], dst_list: List[Dict], max_k=2) -> List[Tuple[Dict, Dict, int]]:
        cands = []
        for a in src_list:
            best: List[Tuple[Dict, Dict, int]] = []
            for b in dst_list:
                d = _distance(a, b)
                if d > LINK_WINDOW_SENTENCES:
                    continue
                best.append((a, b, d))
            best.sort(key=lambda t: t[2])
            cands.extend(best[:max_k])
        # dedup
        seen = set()
        out = []
        for a, b, d in sorted(cands, key=lambda t: t[2]):
            key = (a["id"], b["id"])
            if key in seen: continue
            seen.add(key)
            out.append((a, b, d))
        return out

    # Technique → Experiment / Result
    for a, b, _ in nearest(by_type["Technique"], by_type["Experiment"], max_k=2):
        add_edge(a, b, "uses", (a["conf"] + b["conf"]) / 2)
    for a, b, _ in nearest(by_type["Technique"], by_type["Result"], max_k=1):
        add_edge(a, b, "uses", (a["conf"] + b["conf"]) / 2)

    # Experiment → Result
    for a, b, _ in nearest(by_type["Experiment"], by_type["Result"], max_k=2):
        add_edge(a, b, "produces", (a["conf"] + b["conf"]) / 2)

    # Result → Hypothesis
    for a, b, _ in nearest(by_type["Result"], by_type["Hypothesis"], max_k=2):
        et = "supports" if a.get("polarity") != "negative" else "refutes"
        add_edge(a, b, et, (a["conf"] + b["conf"]) / 2)

    # Dataset → Experiment/Analysis
    for a, b, _ in nearest(by_type["Dataset"], by_type["Experiment"], max_k=1):
        add_edge(a, b, "feeds", (a["conf"] + b["conf"]) / 2)
    for a, b, _ in nearest(by_type["Dataset"], by_type["Analysis"], max_k=1):
        add_edge(a, b, "feeds", (a["conf"] + b["conf"]) / 2)

    # Analysis → Result
    for a, b, _ in nearest(by_type["Analysis"], by_type["Result"], max_k=1):
        add_edge(a, b, "informs", (a["conf"] + b["conf"]) / 2)

    edges = [e for e in edges if e["conf"] >= CONF_EDGE_MIN]

    # fallback, если пусто
    if not edges and nodes:
        res = sorted(by_type["Result"], key=lambda n: -n["conf"])[:2]
        hyp = sorted(by_type["Hypothesis"], key=lambda n: -n["conf"])[:2]
        if res and hyp:
            for i, (a, b) in enumerate([(res[0], hyp[0])] + ([(res[1], hyp[0])] if len(res) > 1 else [])):
                et = "supports" if a.get("polarity") != "negative" else "refutes"
                add_edge(a, b, et, (a["conf"] + b["conf"]) / 2)
        tech = sorted(by_type["Technique"], key=lambda n: -n["conf"])[:1]
        exp = sorted(by_type["Experiment"], key=lambda n: -n["conf"])[:1]
        if tech and (exp or res):
            target = exp[0] if exp else res[0]
            add_edge(tech[0], target, "uses", (tech[0]["conf"] + target["conf"]) / 2)

    return edges


# ─────────────────────────────────────────────────────────────
# Основной раннер
# ─────────────────────────────────────────────────────────────

def run_s1(s0_path: str,
           rules_path: Optional[str],
           graph_path: str,
           *,
           themes_root: Optional[str] = None,
           theme_override: Optional[List[str]] = None
           ) -> None:
    """
    S0 (плоский sentences[]) → S1 кандидаты (узлы + рёбра).
    """
    s0 = json.loads(Path(s0_path).read_text(encoding="utf-8"))
    out_dir = Path(graph_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # загрузка spaCy и паттернов
    nlp, dep_enabled, model_name = load_spacy_model()
    matcher, depmatcher, type_boosts, registry = load_spacy_patterns(
        nlp, themes_root or "/app/rules/themes", theme_override
    )
    themes_used = theme_override or ["common"]
    pattern_sources = sorted({r["source"] for r in registry})

    nodes: List[Dict[str, Any]] = []
    debug_hits: List[Dict[str, Any]] = []
    idx = 1
    doc_id = s0.get("doc_id", "doc")

    # единый проход по плоскому списку предложений
    for s_idx, sent in _sent_iter_flat(s0):
        text = (sent.get("text") or "").strip()
        if not text:
            continue
        imrad = (sent.get("section_hint") or "OTHER").upper()
        page = sent.get("page", 0)
        bbox = sent.get("bbox") or []
        is_cap = bool(sent.get("is_caption", False))
        cap_type = (sent.get("caption_type") or "").strip()

        # doc для spaCy
        doc = nlp(text)
        tname, conf, dbg, rule_hits = _score_sentence(doc, text, imrad, matcher, depmatcher, type_boosts, dep_enabled)

        # --- citation guard:References/цитатные фразы → Input Fact, не Result/Technique/... ---
        text_l = text.lower()
        # эвристики цитирования: [12], (Smith, 2010), [Medline], [CrossRef], длинные блоки ссылок
        looks_like_citation = (
                (text.count("[") + text.count("]") >= 2) or
                bool(re.search(r"\(\s?[A-Z][a-z]+(?:\s+et al\.)?,\s?\d{4}\s?\)", text)) or
                "[medline]" in text_l or "[crossref]" in text_l or
                bool(re.findall(r"\[\d{1,3}(?:[,\-\s]\d{1,3})*\]", text))  # группы ссылок [1,2-5,7]
        )
        in_refs = (imrad == "REFERENCES")

        if in_refs or looks_like_citation:
            # жёстко понижаем/переназначаем тип
            if tname is None:
                tname, conf = "Input Fact", 0.45
            elif tname != "Input Fact":
                tname = "Input Fact"
                conf = min(conf, 0.45)  # не даём «перетягивать» семантику
            # можно также выкинуть такие предложения вовсе:
            # if in_refs: continue

        if not tname or conf < CONF_NODE_MIN:
            continue

        pol = _polarity(text)

        # секционное имя
        if is_cap and cap_type == "Figure":
            section_name = "FigureCaption"
        elif is_cap and cap_type == "Table":
            section_name = "TableCaption"
        else:
            section_name = "Body"

        node = _mk_node(
            doc_id, idx, tname, text, conf,
            section_name=section_name, imrad_hint=imrad,
            s_idx=s_idx, page=page, bbox=bbox, polarity=pol
        )
        node["matched_rules"] = rule_hits  # список объектов: {type, source, index, engine, span/text|tokens, raw}
        nodes.append(node)
        debug_hits.append({
            "sec": section_name, "imrad": imrad, "text": text[:200],
            "chosen": tname, "conf": round(conf, 3), **dbg
        })
        idx += 1

    # линковка
    edges = _link_inside(doc_id, nodes)

    # артефакты
    s1_graph = {
        "doc_id": doc_id,
        "themes_used": themes_used,  # НОВОЕ
        "pattern_sources": pattern_sources,  # НОВОЕ
        "spacy_model": model_name,  # опционально, но полезно
        "dep_enabled": bool(dep_enabled),  # опционально
        "nodes": nodes,
        "edges": edges
    }
    (out_dir / "s1_graph.json").write_text(json.dumps(s1_graph, ensure_ascii=False, indent=2), encoding="utf-8")

    # s1_debug = {
    #     "summary": {
    #         "nodes_total": len(nodes),
    #         "edges_total": len(edges),
    #         "conf_node_min": CONF_NODE_MIN,
    #         "conf_edge_min": CONF_EDGE_MIN,
    #         "themes_loaded": theme_override or ["common"],
    #         "spacy_model": model_name,
    #         "dep_enabled": bool(dep_enabled),
    #     },
    #     "hits": debug_hits
    # }
    # (out_dir / "s1_debug.json").write_text(json.dumps(s1_debug, ensure_ascii=False, indent=2), encoding="utf-8")

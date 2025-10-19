# -*- coding: utf-8 -*-
"""
S1 (spaCy + data-driven patterns) → s1_graph.json / s1_debug.json

- Вход: s0.json (тонкий, с sections[].blocks[].sentences[] и captions[])
- Паттерны: themes/<topic>/patterns/{matcher.json, depmatcher.json}, плюс общие themes/common/...
- Выход: s1_graph.json (кандидаты узлов + рёбра внутри статьи), s1_debug.json (сводка)

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

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set

import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Doc

# ─────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────

NODE_TYPES = [
    "Input Fact", "Hypothesis", "Experiment", "Technique",
    "Result", "Dataset", "Analysis", "Conclusion"
]

TYPE_CANON = {t.lower().replace(" ", ""): t for t in NODE_TYPES}

TYPE_ORDER = [
    "Input Fact", "Hypothesis", "Experiment", "Technique",
    "Dataset", "Analysis", "Result", "Conclusion"
]

# веса секций (IMRAD hint) — мягкие приоры
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

ALLOWED_EDGE_TYPES = {
    ("Technique", "Experiment"): "uses",
    ("Technique", "Result"): "uses",
    ("Experiment", "Result"): "produces",
    ("Result", "Hypothesis"): "supports",  # может стать refutes по polarity
    ("Dataset", "Experiment"): "feeds",
    ("Dataset", "Analysis"): "feeds",
    ("Analysis", "Result"): "informs",
}

# сколько соседей смотреть для ближней линковки
LINK_WINDOW_SENTENCES = 2  # ±2 предложения
LINK_WINDOW_PARAGRAPHS = 1  # ±1 абзац

# ─────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────

def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _ensure_list(x: Any) -> List:
    if x is None: return []
    if isinstance(x, list): return x
    return [x]


def _canon_type(label: str) -> Optional[str]:
    k = label.lower().replace(" ", "")
    return TYPE_CANON.get(k)


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


def _hedge_penalty(tokens_lower: List[str]) -> float:
    return HEDGE_PENALTY if _contains_any(tokens_lower, HEDGING) else 0.0


def _section_prior(imrad_hint: str, tname: str) -> float:
    return SECTION_PRIORS.get(imrad_hint or "OTHER", {}).get(tname, 1.0)


def _base_type_weight(tname: str) -> float:
    return BASE_TYPE_WEIGHTS.get(tname, 1.0)


def _sent_iter(s0: Dict) -> Iterable[Tuple[int, int, int, Dict, str, str]]:
    """
    Итерирует предложения.
    Возвращает: (sec_idx, blk_idx, s_idx, sent_obj, section_name, imrad_hint)
    """
    for si, sec in enumerate(s0.get("sections", [])):
        sname = sec.get("name") or "Body"
        hint = sec.get("imrad_hint") or "OTHER"
        for pi, blk in enumerate(sec.get("blocks", [])):
            for s in blk.get("sentences", []):
                yield si, pi, int(s.get("s_idx", 0)), s, sname, hint


def _captions_to_sent_like(s0: Dict) -> List[Tuple[str, Dict]]:
    """
    Делает из captions «квазипредложения» для матчеров (с меткой секции FigureCaption/TableCaption).
    Возвращает список (section_name, sent_like_dict)
    """
    out = []
    for c in s0.get("captions", []):
        sid = c.get("id", "")
        txt = c.get("text", "") or ""
        if not txt.strip():
            continue
        sec_name = "FigureCaption" if sid.lower().startswith("figure") else "TableCaption"
        out.append((sec_name, {"text": txt, "page": c.get("page", 0), "coords": c.get("coords", [])}))
    return out

# ─────────────────────────────────────────────────────────────
# Загрузка spaCy-паттернов
# ─────────────────────────────────────────────────────────────

def _load_spacy_model() -> tuple[spacy.language.Language, bool, str]:
    """
    Пытаемся загрузить модель из ENV (SPACY_MODEL) или en_core_web_sm.
    Если не получилось — fallback на spacy.blank('en').
    Возвращаем (nlp, dep_enabled, model_name).
    """
    name = os.getenv("SPACY_MODEL", "en_core_web_sm")
    try:
        # оставляем теггер/парсер, вырубаем тяжёлое
        nlp = spacy.load(name, disable=["ner", "textcat"])
        dep_enabled = nlp.has_pipe("parser")
        return nlp, dep_enabled, name
    except Exception:
        # мягкая деградация: только токенизация (Matcher по TEXT будет работать, DependencyMatcher — нет)
        nlp = spacy.blank("en")
        if not nlp.has_pipe("sentencizer"):
            nlp.add_pipe("sentencizer")
        return nlp, False, f"blank:en (fallback; missing {name})"


def _load_spacy_patterns(nlp,
                         themes_root: str,
                         theme_override: Optional[List[str]] = None
                         ) -> Tuple[Matcher, DependencyMatcher, Dict[str, float]]:
    """
    Загружает паттерны из themes/<topic>/patterns/{matcher.json, depmatcher.json}
    Если темы не заданы → используем только themes/common.
    Возвращает (matcher, depmatcher, type_boosts) — где type_boosts можно использовать как доп. веса типа.
    Формат matcher.json: [{ "label": "Result", "pattern": [...] }, ...]
    Формат depmatcher.json: [{ "label": "Technique", "pattern": { "nodes": [...], "edges": [...] } }, ...]
    """
    matcher = Matcher(nlp.vocab)
    depmatcher = DependencyMatcher(nlp.vocab)

    roots = []
    themes_root_path = Path(themes_root or "/app/rules/themes")
    if theme_override:
        for t in theme_override:
            roots.append(themes_root_path / t / "patterns")
    # common всегда подключаем
    roots.append(themes_root_path / "common" / "patterns")

    added = 0
    for r in roots:
        if not r.exists():
            continue
        # Matcher
        mfile = r / "matcher.json"
        if mfile.exists():
            try:
                items = _load_json(mfile)
                if isinstance(items, list):
                    for it in items:
                        label = _canon_type(str(it.get("label", "")))
                        pattern = it.get("pattern")
                        if not label or not pattern:
                            continue
                        matcher.add(label, [pattern])
                        added += 1
            except Exception:
                pass
        # DependencyMatcher
        dfile = r / "depmatcher.json"
        if dfile.exists():
            try:
                items = _load_json(dfile)
                if isinstance(items, list):
                    for it in items:
                        label = _canon_type(str(it.get("label", "")))
                        pat = it.get("pattern")
                        if not label or not pat:
                            continue
                        depmatcher.add(label, [pat])
                        added += 1
            except Exception:
                pass

    # Доп. бусты типов можно хранить в themes/<topic>/lexicon.json → {"type_boosts":{"Result":1.02,...}}
    type_boosts: Dict[str, float] = {}
    for r in roots:
        lfile = r.parent / "lexicon.json"
        if lfile.exists():
            try:
                lex = _load_json(lfile)
                for k, v in (lex.get("type_boosts") or {}).items():
                    t = _canon_type(str(k))
                    if t:
                        type_boosts[t] = float(v)
            except Exception:
                pass

    return matcher, depmatcher, type_boosts

# ─────────────────────────────────────────────────────────────
# Оценка и выбор типа для предложения
# ─────────────────────────────────────────────────────────────

def _score_sentence(nlp_doc: Doc,
                    text: str,
                    imrad_hint: str,
                    matcher: Matcher,
                    depmatcher: DependencyMatcher,
                    type_boosts: Dict[str, float],
                    dep_enabled: bool) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Возвращает (best_type, conf, debug_hits)
    conf ∈ [0, +∞), порог отсечки на уровне узла — CONF_NODE_MIN
    """
    hits = {}  # type -> raw score
    # 1) token-level hits
    for label, start, end in matcher(nlp_doc):
        tname = nlp_doc.vocab.strings[label]
        tname = _canon_type(tname) or tname
        if tname not in NODE_TYPES:
            continue
        hits[tname] = hits.get(tname, 0) + 1.0

    # 2) dependency hits (только если есть парсер)
    if dep_enabled:
        for label, _ in depmatcher(nlp_doc):
            tname = nlp_doc.vocab.strings[label]
            tname = _canon_type(tname) or tname
            if tname not in NODE_TYPES:
                continue
            hits[tname] = hits.get(tname, 0) + 1.2  # чуть сильнее, чем token-matcher

    if not hits:
        return None, 0.0, {"hits": {}}

    # 3) приоры секции
    prior_mult = {t: _section_prior(imrad_hint, t) for t in hits.keys()}

    # 4) базовый вес типа
    base_mult = {t: _base_type_weight(t) for t in hits.keys()}

    # 5) boost темы (из lexicon.json)
    theme_mult = {t: type_boosts.get(t, 1.0) for t in hits.keys()}

    # 6) hedge penalty
    tokens_lower = [t.text.lower() for t in nlp_doc]
    hedge = _hedge_penalty(tokens_lower)

    # 7) numeric bonus для некоторых типов
    num_bonus = NUMERIC_BONUS if _has_numeric(text) else 0.0

    # 8) итоговый скор
    scored = {}
    for t, raw in hits.items():
        val = raw * prior_mult.get(t, 1.0) * base_mult.get(t, 1.0) * theme_mult.get(t, 1.0)
        if t in ("Result", "Analysis", "Dataset"):
            val += num_bonus
        val -= hedge
        scored[t] = val

    best_t = max(scored.items(), key=lambda kv: kv[1])
    return (best_t[0], float(best_t[1]), {
        "hits": hits,
        "prior_mult": prior_mult,
        "base_mult": base_mult,
        "theme_mult": theme_mult,
        "hedge": hedge,
        "num_bonus": num_bonus,
        "scored": scored
    })

# ─────────────────────────────────────────────────────────────
# Построение узлов/рёбер
# ─────────────────────────────────────────────────────────────

def _node_id(doc_id: str, tname: str, idx: int) -> str:
    slug = doc_id.replace("/", "-").replace(":", "-")
    return f"{slug}:{tname.replace(' ', ''):%s}" % (f"{idx:04d}")


def _mk_node(doc_id: str, idx: int, tname: str, text: str, conf: float,
             section_name: str, imrad_hint: str, sec_idx: int, blk_idx: int, s_idx: int,
             page: int, coords: List[Dict[str, Any]], polarity: str) -> Dict[str, Any]:
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
            "sec_idx": sec_idx,
            "block_idx": blk_idx,
            "sent_idx": s_idx,
        },
        "page": page,
        "coords": coords
    }


def _distance(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    # расстояние по предложению/абзацу (внутри секции), чем меньше — тем ближе
    ap = (a["prov"]["sec_idx"], a["prov"]["block_idx"], a["prov"]["sent_idx"])
    bp = (b["prov"]["sec_idx"], b["prov"]["block_idx"], b["prov"]["sent_idx"])
    if ap[0] != bp[0]:
        return 999999
    # приоритет по абзацам, потом предложениям
    para_d = abs(ap[1] - bp[1])
    sent_d = abs(ap[2] - bp[2])
    return para_d * 10 + sent_d


def _link_inside(doc_id: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []

    # индексы по типу
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
                "from": {"sec": src["prov"]["section"], "blk": src["prov"]["block_idx"], "s": src["prov"]["sent_idx"]},
                "to": {"sec": dst["prov"]["section"], "blk": dst["prov"]["block_idx"], "s": dst["prov"]["sent_idx"]},
            }
        })

    # полезная функция поиска ближайших
    def nearest(src_list: List[Dict], dst_list: List[Dict], max_k=2) -> List[Tuple[Dict, Dict, int]]:
        cands = []
        for a in src_list:
            best: List[Tuple[Dict, Dict, int]] = []
            for b in dst_list:
                d = _distance(a, b)
                if d >= 999999: continue
                # окно
                para_d = abs(a["prov"]["block_idx"] - b["prov"]["block_idx"])
                sent_d = abs(a["prov"]["sent_idx"] - b["prov"]["sent_idx"])
                if para_d > LINK_WINDOW_PARAGRAPHS or sent_d > LINK_WINDOW_SENTENCES:
                    continue
                best.append((a, b, d))
            best.sort(key=lambda t: t[2])
            cands.extend(best[:max_k])
        # убрать дубли пар
        seen = set()
        out = []
        for a, b, d in sorted(cands, key=lambda t: t[2]):
            key = (a["id"], b["id"])
            if key in seen:
                continue
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

    # Result → Hypothesis (supports/refutes)
    for a, b, _ in nearest(by_type["Result"], by_type["Hypothesis"], max_k=2):
        et = "supports"
        if a.get("polarity") == "negative":
            et = "refutes"
        add_edge(a, b, et, (a["conf"] + b["conf"]) / 2)

    # Dataset → Experiment/Analysis
    for a, b, _ in nearest(by_type["Dataset"], by_type["Experiment"], max_k=1):
        add_edge(a, b, "feeds", (a["conf"] + b["conf"]) / 2)
    for a, b, _ in nearest(by_type["Dataset"], by_type["Analysis"], max_k=1):
        add_edge(a, b, "feeds", (a["conf"] + b["conf"]) / 2)

    # Analysis → Result
    for a, b, _ in nearest(by_type["Analysis"], by_type["Result"], max_k=1):
        add_edge(a, b, "informs", (a["conf"] + b["conf"]) / 2)

    # фильтр по порогу
    edges = [e for e in edges if e["conf"] >= CONF_EDGE_MIN]

    # fallback: если рёбер нет, а узлы есть
    if not edges and nodes:
        # 1–2 ребра Result → Hypothesis
        res = sorted(by_type["Result"], key=lambda n: -n["conf"])[:2]
        hyp = sorted(by_type["Hypothesis"], key=lambda n: -n["conf"])[:2]
        if res and hyp:
            for i, (a, b) in enumerate([(res[0], hyp[0])] + ([(res[1], hyp[0])] if len(res) > 1 else [])):
                et = "supports" if a.get("polarity") != "negative" else "refutes"
                add_edge(a, b, et, (a["conf"] + b["conf"]) / 2)
        # Technique → (Experiment|Result)
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
           rules_path: Optional[str],  # не используется (ради совместимости)
           graph_path: str,
           *,
           themes_root: Optional[str] = None,
           theme_override: Optional[List[str]] = None
           ) -> None:
    """
    Преобразует s0.json → s1_graph.json (+ s1_debug.json).
    graph_path — путь к будущему финальному graph.json (S2 его перезапишет),
                 но тут мы используем его директорию для сохранения артефактов S1.
    """
    s0 = json.loads(Path(s0_path).read_text(encoding="utf-8"))
    out_dir = Path(graph_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # построим «квази-секции» для captions, чтобы матчеры могли их ловить
    captions_sent_like = _captions_to_sent_like(s0)

    # загрузка spaCy и паттернов (с фолбэком)
    nlp, dep_enabled, model_name = _load_spacy_model()
    matcher, depmatcher, type_boosts = _load_spacy_patterns(
        nlp, themes_root or "/app/rules/themes", theme_override
    )

    # обход предложений и матчинг
    nodes: List[Dict[str, Any]] = []
    debug_hits: List[Dict[str, Any]] = []
    idx = 1
    doc_id = s0.get("doc_id", "doc")

    # 1) обычные предложения из секций
    for si, sec in enumerate(s0.get("sections", [])):
        sname = sec.get("name") or "Body"
        imrad = sec.get("imrad_hint") or "OTHER"
        for bi, blk in enumerate(sec.get("blocks", [])):
            for sent in blk.get("sentences", []):
                text = (sent.get("text") or "").strip()
                if not text:
                    continue
                # ВАЖНО: полный прогон пайплайна, а не make_doc
                doc = nlp(text)
                tname, conf, dbg = _score_sentence(doc, text, imrad, matcher, depmatcher, type_boosts, dep_enabled)
                if not tname or conf < CONF_NODE_MIN:
                    continue
                pol = _polarity(text)
                node = _mk_node(
                    doc_id, idx, tname, text, conf,
                    section_name=sname, imrad_hint=imrad,
                    sec_idx=si, blk_idx=bi, s_idx=int(sent.get("s_idx", 0)),
                    page=int(sent.get("page", blk.get("page", 0))), coords=sent.get("coords", blk.get("coords", [])),
                    polarity=pol
                )
                nodes.append(node)
                debug_hits.append({
                    "sec": sname, "imrad": imrad, "text": text[:200],
                    "chosen": tname, "conf": round(conf, 3), **dbg
                })
                idx += 1

    # 2) captions как отдельные «предложения»
    for sec_name, cap in captions_sent_like:
        text = (cap.get("text") or "").strip()
        if not text:
            continue
        imrad = "RESULTS" if sec_name.startswith("Figure") else "RESULTS"
        doc = nlp(text)
        tname, conf, dbg = _score_sentence(doc, text, imrad, matcher, depmatcher, type_boosts, dep_enabled)
        if not tname or conf < CONF_NODE_MIN:
            continue
        pol = _polarity(text)
        node = _mk_node(
            doc_id, idx, tname, text, conf,
            section_name=sec_name, imrad_hint=imrad,
            sec_idx=-1, blk_idx=-1, s_idx=0,
            page=int(cap.get("page", 0)), coords=cap.get("coords", []),
            polarity=pol
        )
        nodes.append(node)
        debug_hits.append({
            "sec": sec_name, "imrad": imrad, "text": text[:200],
            "chosen": tname, "conf": round(conf, 3), **dbg
        })
        idx += 1

    # линковка
    edges = _link_inside(doc_id, nodes)

    # артефакты
    s1_graph = {
        "doc_id": doc_id,
        "nodes": nodes,
        "edges": edges
    }
    (out_dir / "s1_graph.json").write_text(json.dumps(s1_graph, ensure_ascii=False, indent=2), encoding="utf-8")

    s1_debug = {
        "summary": {
            "nodes_total": len(nodes),
            "edges_total": len(edges),
            "conf_node_min": CONF_NODE_MIN,
            "conf_edge_min": CONF_EDGE_MIN,
            "themes_loaded": theme_override or ["common"],
            "spacy_model": model_name,
            "dep_enabled": bool(dep_enabled),
        },
        "hits": debug_hits
    }
    (out_dir / "s1_debug.json").write_text(json.dumps(s1_debug, ensure_ascii=False, indent=2), encoding="utf-8")

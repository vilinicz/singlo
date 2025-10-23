# -*- coding: utf-8 -*-
"""
S1 (spaCy + data-driven patterns) → s1_graph.json / s1_debug.json

Вход ТОЛЬКО плоский S0:
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
"""

from __future__ import annotations

# Stage S1 overview:
# - Loads a spaCy model and data-driven patterns (token and optional dependency)
# - Scores each S0 sentence into one of 8 node types with a confidence
# - Applies light heuristics (hedging penalty, numeric bonus, IMRAD priors)
# - Links nearby nodes into edges within a sliding sentence window
# - Writes s1_graph.json with nodes/edges + provenance for debugging

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
    "Result", "Dataset", "Analysis", "Conclusion", "Other"
]

TYPE_CANON = {t.lower().replace(" ", ""): t for t in NODE_TYPES}

SECTION_PRIORS = {
    "ABSTRACT": {"Hypothesis": 1.18},
    "INTRO": {"Hypothesis": 1.25, "Input Fact": 1.06, "Conclusion": 1.00},
    "METHODS": {"Technique": 1.15, "Experiment": 1.12, "Dataset": 1.10, "Analysis": 1.05},
    "RESULTS": {"Result": 1.18, "Analysis": 1.08},
    # NOTE: усилили Result в DISCUSSION, чтобы «словесные» связи проходили увереннее
    "DISCUSSION": {
        "Conclusion": 1.20,
        "Result": 1.12,
        "Hypothesis": 0.92,  # штрафуем гипотезы в DISCUSSION
        "Dataset": 0.95,
        "Analysis": 1.03
    },
    "CONCLUSION": {"Conclusion": 1.22},
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
CONF_EDGE_MIN = 0.52

# ── Тонкая настройка эвристик и бустов ───────────────────────
# Усиление для подписей к рисункам/таблицам (после citation guard)
CAPTION_BOOST_FIG = 0.08
CAPTION_BOOST_TAB = 0.08
CAPTION_BOOST_TYPES = {"Result", "Analysis", "Experiment", "Dataset"}

# Эвристика «фраз вывода» (переключение в Conclusion)
CONCLUSION_HINTS = {
    "enable": True,
    "sections": {"DISCUSSION", "RESULTS"},
    # требуем, чтобы оценка Conclusion была не ниже min_ratio от лучшего класса
    "min_ratio": 0.90
}

# Fallback-рёбра (минимальные связи при их полном отсутствии)
# Режимы: "off" (по умолчанию), "safe" (только если >1 типа узлов и конфиденс высок)
FALLBACK_EDGES_MODE = "off"

# Леммы, сигнализирующие «позитивный» исход/оценку
_POS_LEMMAS = {
    "demonstrate", "show",
    "increase", "improve", "enhance", "benefit", "boost", "broaden", "expand", "raise",
    "validate", "confirm", "corroborate", "replicate", "support",
    "fit", "correlate", "outperform", "reduce"  # reduce — слабый позитив (шум/ошибка/вред)
}

# Леммы, сигнализирующие «негативный» исход/оценку
_NEG_LEMMAS = {
    "decrease", "worsen", "worse",
    "fail", "lack",
}

# Фразовые негативы, которые не всегда удобно ловить леммами
_NEG_PHRASES = (
    "not significant",
    "not statistically significant",
    "non significant",
    "non-significant",
    "null result",
    "negative result",
)

# Нейтрализаторы: явная негация перед «негативными» леммами
# (e.g., "no decrease", "not worsened") — не считаем это за negative
_NEUTRALIZE_NEG = {
    "decrease", "worsen", "worse"
}

LINK_WINDOW_SENTENCES = 2
LINK_WINDOW_FORWARD = 6  # чуть шире, но только вперёд по тексту
# max allowed page distance for edge candidates
PAGE_DELTA_MAX = 2

_RX_CIT_AUTHOR_YEAR = re.compile(
    r'\((?:[A-Z][a-zA-Z\-]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][a-zA-Z\-]+)?,?\s*)\d{4}[a-z]?\)', re.U)
_RX_CIT_BRACKETS_NUM = re.compile(r'\[(?:\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\]', re.U)
_RX_GUARD_NOT_CIT = re.compile(r'\((?:n\s*=\s*\d+|p\s*[<≤]\s*0\.\d+|Fig\.|Table|Eq\.)', re.I | re.U)

_RX_NUMERIC_HINT = re.compile(
    r'(?:\b\d+(?:\.\d+)?\s?%|\bby\s+\d+(?:\.\d+)?\b|\bp\s*[<≤]\s*0\.(?:0[1-5]|05)\b|\bAUC\b|\bRMSE\b)',
    re.I | re.U)

IMRAD_NORMAL = {
    "INTRO": "INTRODUCTION", "BACKGROUND": "INTRODUCTION",
    "METHODS": "METHODS", "MATERIALS AND METHODS": "METHODS",
    "RESULTS": "RESULTS", "RESULTS AND DISCUSSION": "DISCUSSION",
    "DISCUSSION": "DISCUSSION", "CONCLUSION": "CONCLUSION", "CONCLUSIONS": "CONCLUSION",
    "OTHER": "OTHER", "UNKNOWN": "OTHER", "ABSTRACT": "INTRODUCTION",
    "REFERENCES": "REFERENCES", "BIBLIOGRAPHY": "REFERENCES"
}

_RX_FIG_REF = re.compile(r'\b(?:Fig\.|Figure)\s*([IVXLC\d]+)\b', re.I)  # римские и арабские
_RX_TABLE_REF = re.compile(r'\b(?:Table)\s*([IVXLC\d]+)\b', re.I)


# ─────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────


def extract_fig_table_refs(text: str):
    figs = [m.group(1) for m in _RX_FIG_REF.finditer(text or "")]
    tabs = [m.group(1) for m in _RX_TABLE_REF.finditer(text or "")]
    # нормализуем римские → верхний регистр, арабские как есть
    return set(_normalize_ref_token(t) for t in figs), set(_normalize_ref_token(t) for t in tabs)


def _normalize_ref_token(tok: str) -> str:
    t = (tok or "").strip().upper()
    # простой нормалайзер: "II" → "II", "2" → "2"
    return t


def imrad_norm(sec: str) -> str:
    return IMRAD_NORMAL.get((sec or "").upper(), "OTHER")


# Разрешённые пары с направлением "вперёд":
# (from_type, to_type) -> функция-предикат по секциям
def imrad_ok(from_type: str, from_sec: str, to_type: str, to_sec: str) -> bool:
    fs = imrad_norm(from_sec)
    ts = imrad_norm(to_sec)
    if fs == "REFERENCES" or ts == "REFERENCES":
        return False  # никаких ребер с References

    # Типичные траектории:
    if from_type == "Experiment" and to_type in {"Result", "Analysis", "Conclusion"}:
        return ts in {"RESULTS", "DISCUSSION", "CONCLUSION"}
    if from_type == "Result" and to_type in {"Result", "Conclusion", "Hypothesis"}:
        return ts in {"RESULTS", "DISCUSSION", "CONCLUSION"}
    if from_type in {"Technique", "Dataset"} and to_type in {"Experiment", "Analysis"}:
        return ts in {"METHODS", "RESULTS", "DISCUSSION"}
    if from_type == "Input Fact" and to_type in {"Hypothesis", "Experiment", "Result", "Conclusion"}:
        # факты чаще из INTRO → дальше куда угодно вперёд
        return ts in {"METHODS", "RESULTS", "DISCUSSION", "CONCLUSION"}

    # по умолчанию — разрешим, если не явная аномалия
    return True


def edge_min_threshold(base_min: float, dsent: int, dpage: int) -> float:
    """
    Гистерезис порога: для очень близких ребер порог ниже, для дальних — выше.
    base_min: базовый порог (например, 0.55)
    """
    # бонус для близких
    bonus = 0.0
    if dsent <= 1:
        bonus += 0.05  # самые близкие — проще пропустить
    elif dsent <= 2:
        bonus += 0.03
    if dpage == 0:
        bonus += 0.02  # на одной странице — проще
    # штраф для дальних
    malus = 0.0
    if dsent >= 8:
        malus += 0.04
    if dpage >= 2:
        malus += 0.03

    thr = base_min - bonus + malus
    # ограничим разумными рамками
    return max(0.40, min(0.70, thr))


def _estimate_citation_strength(text: str) -> float:
    """Heuristic strength: 1.0 structural (we can't see TEI here), 0.7 author-year, 0.5 [n]."""
    t = text or ""
    if not t:
        return 0.0
    # Guards to avoid misreading (n=), (p<), (Fig.), ...
    if _RX_GUARD_NOT_CIT.search(t):
        # allow brackets outside guard
        has_num = bool(_RX_CIT_BRACKETS_NUM.search(t))
        return 0.5 if has_num else 0.0
    if _RX_CIT_AUTHOR_YEAR.search(t):
        return 0.7
    if _RX_CIT_BRACKETS_NUM.search(t):
        return 0.5
    return 0.0


def _citation_heavy(text: str) -> bool:
    """True if the sentence is mostly citations (to demote via citation_soft)."""
    if not text:
        return False
    cit_tokens = len(_RX_CIT_AUTHOR_YEAR.findall(text)) + len(_RX_CIT_BRACKETS_NUM.findall(text))
    # crude density: many cites or small text dominated by them
    return cit_tokens >= 2 or (cit_tokens >= 1 and len(text) < 120)


## Canonicalize a free-form label into one of NODE_TYPES, or None.
def _canon_type(label: str) -> Optional[str]:
    return TYPE_CANON.get(label.lower().replace(" ", ""))


## Return True if any token is present in the given vocabulary set.
def _contains_any(tokens_lower: List[str], vocab: Set[str]) -> bool:
    return any(t in vocab for t in tokens_lower)


## Heuristic: does text contain numbers/percent/p-values.
def _has_numeric(text: str) -> bool:
    # быстрый признак «есть числа/проценты/p-values»
    t = text.lower()
    return any(c.isdigit() for c in t) or "%" in t or " p<" in t or " p =" in t


# ——— Reference-like detector (bibliography-looking lines) ———
_REF_INLINE = re.compile(r"\b(vol\.?|no\.?|pp\.?|doi:?|issn|et al\.)\b", re.I)
_REF_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_REF_PAGES = re.compile(r"\b\d{1,4}\s*[–\-]\s*\d{1,4}\b")


def looks_like_reference(text: str) -> bool:
    if not text or len(text) < 30:
        return False
    # явные маркеры "vol./no./pp./doi/ISSN/et al."
    if _REF_INLINE.search(text):
        return True
    # год + диапазон страниц (частый паттерн библиографии)
    if _REF_YEAR.search(text) and _REF_PAGES.search(text):
        return True
    return False


TIEBREAK_ORDER = ["Result", "Experiment", "Technique", "Analysis", "Dataset", "Hypothesis", "Conclusion", "Input Fact",
                  "Other"]


def tiebreak_label(scored: Dict[str, float], current_best: str) -> str:
    if not scored:
        return current_best
    # берём максимум и список «ничьих»
    best_val = max(scored.values())
    cands = [k for k, v in scored.items() if abs(v - best_val) < 1e-9]
    if len(cands) == 1:
        return cands[0]
    for lab in TIEBREAK_ORDER:
        if lab in cands:
            return lab
    return current_best


def _has_phrase(hay: str, needle: str) -> bool:
    return needle in hay


def _polarity(doc: Doc) -> str:
    """
    Лёгкая, но устойчивая полярность:
      - смотрим на леммы (spaCy) из _POS_LEMMAS/_NEG_LEMMAS,
      - учитываем зависимость neg (scope отрицания),
      - ловим частые фразовые паттерны ("not significant", "non-significant", ...),
      - нейтрализуем "no decrease"/"not worsened" и т.п.
    Возвращает: 'positive' | 'negative' | 'neutral'
    """
    if doc is None or len(doc) == 0:
        return "neutral"

    t_lower = doc.text.lower()

    # 1) Фразовые негативы (самые надёжные сигналы)
    has_neg_phrase = any(_has_phrase(t_lower, p) for p in _NEG_PHRASES)

    # 2) Леммы и neg-scope
    pos_hit = False
    neg_hit = False

    # Соберём набор токенов, над которыми висит neg (или сами neg-токены)
    # Пример: "did not improve" → "improve" имеет child.dep_ == 'neg'
    negated_heads = set()
    for tok in doc:
        if tok.dep_ == "neg":
            # Обычно neg → head — глагол/прилагательное, которого касается отрицание
            negated_heads.add(tok.head.i)

    # Нейтрализаторы «no decrease / not worsened»
    neutralize_negative = False
    for tok in doc:
        if tok.dep_ == "neg" and tok.head.lemma_ in _NEUTRALIZE_NEG:
            neutralize_negative = True
            break
        # или простая линейная форма "no decrease"
        if tok.text.lower() == "no":
            nxt = tok.nbor(1) if tok.i + 1 < len(doc) else None
            if nxt is not None and nxt.lemma_ in _NEUTRALIZE_NEG:
                neutralize_negative = True
                break

    # Сигналы по леммам с учётом neg-области
    for tok in doc:
        lem = tok.lemma_.lower()

        # позитивный индикатор: токен из _POS_LEMMAS и НЕ под neg
        if lem in _POS_LEMMAS and tok.i not in negated_heads:
            pos_hit = True

        # негативный индикатор: токен из _NEG_LEMMAS и НЕ нейтрализован "no/not <neg-lemma>"
        if lem in _NEG_LEMMAS:
            if neutralize_negative:
                continue
            # если сам негативный токен под neg — это уже «двойная негация» → не считаем её негативом
            if tok.i in negated_heads:
                continue
            neg_hit = True

    # 3) Комбинируем признаки
    # Фразовые негативы имеют приоритет и включают neg_hit,
    # но «нейтрализатор» может их смягчить до neutral (редко встречается вместе)
    if has_neg_phrase:
        neg_hit = True

    if pos_hit and not neg_hit:
        return "positive"
    if neg_hit and not pos_hit:
        return "negative"
    return "neutral"


## Compute a confidence penalty if hedging words are present in the doc.
def _hedge_penalty_from_doc(nlp_doc: Doc) -> float:
    tokens_lower = [t.text.lower() for t in nlp_doc]
    return HEDGE_PENALTY if _contains_any(tokens_lower, HEDGING) else 0.0


## Prior multiplier based on IMRAD section hint and target node type.
def _section_prior(imrad_hint: str, tname: str) -> float:
    return SECTION_PRIORS.get(imrad_hint or "OTHER", {}).get(tname, 1.0)


## Base weight for a node type before heuristics and matches.
def _base_type_weight(tname: str) -> float:
    return BASE_TYPE_WEIGHTS.get(tname, 1.0)


## Iterate flat sentences from S0 as (global_idx, sentence_dict).
def _sent_iter_flat(s0: Dict) -> Iterable[Tuple[int, Dict]]:
    """
    Итерирует плоский список предложений s0["sentences"].
    Возвращает (global_idx, sent_obj)
    """
    for i, s in enumerate(s0.get("sentences", [])):
        yield i, s


# ─────────────────────────────────────────────────────────────
# Оценка и выбор типа
# ─────────────────────────────────────────────────────────────

## Score a sentence to a node type using token/dep matchers and heuristics.
def _score_sentence(nlp_doc: Doc,
                    text: str,
                    imrad_hint: str,
                    matcher: Matcher | None,
                    depmatcher: Optional[DependencyMatcher],
                    type_boosts: Dict[str, float],
                    rule_configs: Dict[str, Dict[str, Any]],
                    dep_enabled: bool) -> Tuple[Optional[str], float, Dict[str, Any], List[dict]]:
    """
    Считает очки по token- и dep-паттернам, устойчив к разным формам token_maps из DependencyMatcher:
    - list[dict[str,int]]
    - list[list[int]] / list[tuple[int, ...]]
    - одиночный int
    """
    hits: Dict[str, float] = {}
    matched_rules: List[dict] = []

    # --- Token patterns (Matcher)
    imrad_normed = (imrad_hint or "OTHER").upper()

    if matcher is not None:
        for match_id, start, end in matcher(nlp_doc):
            try:
                rule_name = nlp_doc.vocab.strings[match_id]
            except Exception:
                continue
            cfg = rule_configs.get(rule_name, {})
            meta = cfg.get("meta") or {}
            allowed_sections = {str(sec).upper() for sec in meta.get("sections", []) if isinstance(sec, str)}
            if allowed_sections and imrad_normed not in allowed_sections:
                continue
            tname = cfg.get("label") or _canon_type(rule_name) or "Other"
            if tname in NODE_TYPES:
                hits[tname] = hits.get(tname, 0.0) + 1.0
                frag = nlp_doc[start:end].text
                matched_rules.append({
                    "type": tname,
                    "engine": "token",
                    "rule": rule_name,
                    "span": [start, end],
                    "text": frag
                })

    has_dep_result = False

    # --- Dependency patterns (DependencyMatcher)
    if dep_enabled and depmatcher is not None:
        for match_id, token_maps in depmatcher(nlp_doc):
            try:
                rule_name = nlp_doc.vocab.strings[match_id]
            except Exception:
                continue
            cfg = rule_configs.get(rule_name, {})
            meta = cfg.get("meta") or {}
            allowed_sections = {str(sec).upper() for sec in meta.get("sections", []) if isinstance(sec, str)}
            if allowed_sections and imrad_normed not in allowed_sections:
                continue
            tname = cfg.get("label") or _canon_type(rule_name) or "Other"
            if tname not in NODE_TYPES:
                continue
            if tname == "Result":
                has_dep_result = True

            # Нормализуем token_maps в последовательность "итерируемых контейнеров" индексов
            # Возможные входы:
            #   - token_maps: list[dict[str,int]]
            #   - token_maps: list[list[int]] / list[tuple[int,...]]
            #   - token_maps: int
            #   - token_maps: неизвестно -> считаем как 1 совпадение без токенов
            norm_maps: List[Iterable[int]] = []
            if isinstance(token_maps, (list, tuple)):
                for m in token_maps:
                    if isinstance(m, dict):
                        norm_maps.append(m.values())
                    elif isinstance(m, (list, tuple)):
                        norm_maps.append(m)
                    elif isinstance(m, int):
                        norm_maps.append([m])
                    else:
                        # пропускаем странные элементы
                        continue
            elif isinstance(token_maps, int):
                norm_maps = [[token_maps]]
            else:
                norm_maps = []

            # Считаем совпадения и вытаскиваем токены для debug
            hit_count = 0
            toks: List[str] = []
            for it in norm_maps:
                any_token = False
                for i in it:
                    if isinstance(i, int) and 0 <= i < len(nlp_doc):
                        toks.append(nlp_doc[i].text)
                        any_token = True
                # если хотя бы один токен валиден — считаем это отдельным совпадением
                hit_count += 1 if any_token else 0

            # Если ничего не распознали — всё равно зачтём хотя бы одну сработку по шаблону
            if hit_count <= 0:
                hit_count = 1

            # Вес dep-хита немного выше, чем token-хита
            hits[tname] = hits.get(tname, 0.0) + 1.2 * hit_count
            matched_rules.append({"type": tname, "engine": "dep", "rule": rule_name, "tokens": toks})

    if not hits:
        return None, 0.0, {"hits": {}}, matched_rules

    # --- Приоры/веса/штрафы
    prior_mult = {t: _section_prior(imrad_hint, t) for t in hits.keys()}
    base_mult = {t: _base_type_weight(t) for t in hits.keys()}
    theme_mult = {t: type_boosts.get(t, 1.0) for t in hits.keys()}
    hedge = _hedge_penalty_from_doc(nlp_doc)
    num_bonus = NUMERIC_BONUS if _has_numeric(text) else 0.0

    scored: Dict[str, float] = {}
    for t, raw in hits.items():
        val = raw * prior_mult.get(t, 1.0) * base_mult.get(t, 1.0) * theme_mult.get(t, 1.0)
        if t in ("Result", "Analysis", "Dataset"):
            val += num_bonus
        val -= hedge
        scored[t] = val

    # --- Выбор лучшего типа
    best_t, best_v = max(scored.items(), key=lambda kv: kv[1])

    # Перетягиваем спорные случаи из Input Fact → Result в RESULTS/DISCUSSION,
    # если есть dep-хит на Result и он близок по баллу.
    if imrad_hint in ("RESULTS", "DISCUSSION") and best_t == "Input Fact":
        if has_dep_result and (scored.get("Result", 0.0) >= 0.88 * best_v):
            best_t, best_v = "Result", scored["Result"]

    dbg = {
        "hits": hits,
        "prior_mult": prior_mult,
        "base_mult": base_mult,
        "theme_mult": theme_mult,
        "hedge": hedge,
        "num_bonus": num_bonus,
        "scored": scored,
        "has_dep_result": has_dep_result
    }
    return best_t, float(best_v), dbg, matched_rules


# ─────────────────────────────────────────────────────────────
# Построение узлов/рёбер
# ─────────────────────────────────────────────────────────────

## Stable node id composed of doc_id, type, and a running index.
def _node_id(doc_id: str, tname: str, idx: int) -> str:
    slug = doc_id.replace("/", "-").replace(":", "-")
    tslug = (tname or "").replace(" ", "")
    return f"{slug}:{tslug}:{idx:04d}"


## Build a canonical node dict with provenance and basic layout hints.
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


def add_figure_table_edges(nodes, base_min=CONF_EDGE_MIN):
    new_edges = []

    # индекс по ссылкам
    by_fig, by_tab = {}, {}
    for n in nodes:
        figs, tabs = extract_fig_table_refs(n.get("text", ""))
        for f in figs: by_fig.setdefault(f, []).append(n)
        for t in tabs: by_tab.setdefault(t, []).append(n)

    def link_group(group, kind: str):
        # разбиваем на подписи и «тело»
        caps = [n for n in group if n.get("prov", {}).get("section") in ("FigureCaption", "TableCaption")]
        body = [n for n in group if n not in caps]
        for c in caps:
            # ближайшие вперёд по тексту
            body_sorted = sorted(
                (b for b in body if int(b["prov"]["sent_idx"]) > int(c["prov"]["sent_idx"])),
                key=lambda b: abs(int(b["prov"]["sent_idx"]) - int(c["prov"]["sent_idx"]))
            )[:3]
            for b in body_sorted:
                etype = "summarizes" if b["type"] in ("Result", "Conclusion") else "informs"
                ds = abs(int(b["prov"]["sent_idx"]) - int(c["prov"]["sent_idx"]))
                dp = abs(int(b.get("page", 0)) - int(c.get("page", 0)))
                # базовый конф — повыше, т.к. это сильный сигнал совпадения
                conf = 0.60
                thr = edge_min_threshold(base_min, ds, dp) - 0.05  # чуть мягче порог за счёт совпадения
                if conf >= thr:
                    new_edges.append({
                        "from": c["id"], "to": b["id"], "type": etype, "conf": round(conf, 3),
                        "prov": {"hint": kind.lower(), kind.lower(): True, "prox": {"ds": ds, "dp": dp}}
                    })

    for f, group in by_fig.items():
        link_group(group, "Figure")
    for t, group in by_tab.items():
        link_group(group, "Table")

    return new_edges


## Sentence distance between two nodes (by their s_idx), for linking.
def _distance(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    # на плоском списке меряем дистанцию по глобальному sent_idx
    sa = int(a["prov"]["sent_idx"])
    sb = int(b["prov"]["sent_idx"])
    return abs(sa - sb)


## Create edges within a small window based on type pairs and polarity.
def _link_inside(doc_id: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []

    by_type: Dict[str, List[Dict[str, Any]]] = {t: [] for t in NODE_TYPES}
    for n in nodes:
        by_type.get(n["type"], []).append(n)

    def add_edge_hysteresis(src: Dict, dst: Dict, etype: str, base_conf: float):
        # локальные расстояния
        try:
            ds = abs(int(src["prov"]["sent_idx"]) - int(dst["prov"]["sent_idx"]))
            dp = abs(int(src.get("page", 0)) - int(dst.get("page", 0)))
        except Exception:
            return  # перестраховка при битых индексах

        thr = edge_min_threshold(CONF_EDGE_MIN, ds, dp)
        if base_conf >= thr:
            add_edge(src, dst, etype, base_conf)

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
        # --- PATCH: small post-hoc reinforcement for nodes that participate in edges
        for n in (src, dst):
            try:
                n["conf"] = float(min(1.0, (n.get("conf") or 0.0) + 0.02))
            except Exception:
                pass

    def _page_of(n: dict) -> int | None:
        """
        Возвращает номер страницы из верхнеуровневого поля node['page'].
        Если страницы нет или она нечисловая — вернёт None.
        """
        try:
            p = n.get("page", None)
            if p is None:
                return None
            return int(p)
        except Exception:
            return None

    def nearest(src_list: List[Dict], dst_list: List[Dict], max_k=2, forward_only: bool = True) -> List[
        Tuple[Dict, Dict, int]]:
        cands = []
        for a in src_list:
            best: List[Tuple[Dict, Dict, int]] = []
            for b in dst_list:
                # --- forward-only: запрещаем рёбра "назад" по тексту
                if forward_only:
                    try:
                        if int(b["prov"]["sent_idx"]) <= int(a["prov"]["sent_idx"]):
                            continue
                        if abs(int(a.get("page", 0)) - int(b.get("page", 0))) > PAGE_DELTA_MAX:
                            continue
                        if not imrad_ok(a["type"], a["prov"]["section"], b["type"], b["prov"]["section"]):
                            continue
                    except Exception:
                        # если индексы битые — лучше перестраховаться и не линковать
                        continue

                # --- page filter: если у обоих есть страницы, ограничиваем разницу
                pa, pb = _page_of(a), _page_of(b)
                if pa is not None and pb is not None:
                    if abs(pb - pa) > PAGE_DELTA_MAX:
                        continue

                d = _distance(a, b)
                # для forward-окна разрешим чуть больше расстояние
                max_d = LINK_WINDOW_FORWARD if forward_only else LINK_WINDOW_SENTENCES
                if d > max_d:
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
    for a, b, _ in nearest(by_type["Technique"], by_type["Experiment"], max_k=2, forward_only=True):
        add_edge_hysteresis(a, b, "uses", (a["conf"] + b["conf"]) / 2)
    for a, b, _ in nearest(by_type["Technique"], by_type["Result"], max_k=1, forward_only=True):
        add_edge_hysteresis(a, b, "uses", (a["conf"] + b["conf"]) / 2)

    # Experiment → Result
    for a, b, _ in nearest(by_type["Experiment"], by_type["Result"], max_k=3, forward_only=True):
        add_edge_hysteresis(a, b, "produces", (a["conf"] + b["conf"]) / 2)

    # Result → Hypothesis
    for a, b, _ in nearest(by_type["Result"], by_type["Hypothesis"], max_k=3, forward_only=True):
        et = "supports" if a.get("polarity") != "negative" else "refutes"
        add_edge_hysteresis(a, b, et, (a["conf"] + b["conf"]) / 2)

    # Dataset → Experiment/Analysis
    for a, b, _ in nearest(by_type["Dataset"], by_type["Experiment"], max_k=1, forward_only=True):
        add_edge_hysteresis(a, b, "feeds", (a["conf"] + b["conf"]) / 2)
    for a, b, _ in nearest(by_type["Dataset"], by_type["Analysis"], max_k=1, forward_only=True):
        add_edge_hysteresis(a, b, "feeds", (a["conf"] + b["conf"]) / 2)

    # Analysis → Result
    for a, b, _ in nearest(by_type["Analysis"], by_type["Result"], max_k=1, forward_only=True):
        add_edge_hysteresis(a, b, "informs", (a["conf"] + b["conf"]) / 2)

    # --- Result → Result  (цепочки промежуточных результатов)
    # строгие ворота уже обеспечены: только вперёд, близко по страницам, малое окно предложений
    for a, b, _ in nearest(by_type["Result"], by_type["Result"], max_k=1, forward_only=True):
        if a["id"] == b["id"]:
            continue
        add_edge_hysteresis(a, b, "follows", (a["conf"] + b["conf"]) / 2)

    # --- Dataset → Result  (описательные/репортинговые результаты по датасету)
    for a, b, _ in nearest(by_type["Dataset"], by_type["Result"], max_k=1, forward_only=True):
        add_edge_hysteresis(a, b, "summarizes", (a["conf"] + b["conf"]) / 2)

    edges = [e for e in edges if e["conf"] >= CONF_EDGE_MIN]

    edges.extend(add_figure_table_edges(nodes, base_min=CONF_EDGE_MIN))

    # Fallback-режим по требованию
    if not edges and nodes and FALLBACK_EDGES_MODE != "off":
        # «безопасный» режим: только если есть хотя бы два разных типа узлов
        types_present = {n["type"] for n in nodes}
        if FALLBACK_EDGES_MODE == "safe" and len(types_present) >= 2:
            res = sorted(by_type["Result"], key=lambda n: -n["conf"])[:2]
            hyp = sorted(by_type["Hypothesis"], key=lambda n: -n["conf"])[:2]
            if res and hyp and (res[0]["conf"] >= 0.65 and hyp[0]["conf"] >= 0.60):
                et = "supports" if res[0].get("polarity") != "negative" else "refutes"
                add_edge_hysteresis(res[0], hyp[0], et, (res[0]["conf"] + hyp[0]["conf"]) / 2)
            tech = sorted(by_type["Technique"], key=lambda n: -n["conf"])[:1]
            if tech and (res or hyp):
                target = (res[0] if res else hyp[0])
                if tech[0]["conf"] >= 0.60 and target["conf"] >= 0.60:
                    add_edge_hysteresis(tech[0], target, "uses", (tech[0]["conf"] + target["conf"]) / 2)

    return edges


# ——— Debug outputs: per-sentence JSONL + grouped Markdown ———
from collections import defaultdict


def write_jsonl_per_sentence(path: str, records: List[Dict[str, Any]]) -> None:
    """Пишет JSONL: по строке на предложение с меткой/конфом/секцией."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_markdown_summary(path: str, records: List[Dict[str, Any]], max_per_label: int = 12) -> None:
    """Группируем по label, показываем топ-фрагменты."""
    buckets = defaultdict(list)
    for r in records:
        buckets[r["label"]].append(r)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# S1 summary (preview)\n\n")
        for lab in sorted(buckets.keys()):
            f.write(f"## {lab}\n\n")
            show = sorted(buckets[lab], key=lambda x: -x.get("conf", 0.0))[:max_per_label]
            for it in show:
                sec = it.get("section") or it.get("imrad", "")
                pg = it.get("page", 0)
                conf = it.get("conf", 0.0)
                f.write(f"- **p.{pg} · {sec} · conf={conf:.3f}** — {it['text']}\n")
            f.write("\n")


# --- PATCH: merge_adjacent pre-pass
def merge_adjacent(nodes: list) -> list:
    """
    Склеивает соседние однотипные узлы в пределах одной страницы и одной секции,
    ТОЛЬКО если у обоих узлов есть валидные sent_idx (>=0) и |Δsent_idx| <= 1.
    Текст конкатенируется, conf усредняется, span объединяется.
    """
    if not nodes:
        return nodes

    def _prov(n: dict) -> dict:
        return n.get("prov", {}) if isinstance(n, dict) else {}

    def _sec(n: dict) -> str:
        return (_prov(n).get("section") or "").lower()

    def _page(n: dict) -> int:
        """
        Страница берётся только из node['page'].
        Если нет — возвращаем -1 (будет считаться «неизвестной» и не склеится по странице).
        """
        try:
            p = n.get("page", None)
            return int(p) if p is not None else -1
        except Exception:
            return -1

    def _sent_idx(n: dict) -> int:
        p = _prov(n)
        try:
            si = p.get("sent_idx", -1)
            return int(si) if si is not None else -1
        except Exception:
            return -1

    def _same_page(a: dict, b: dict) -> bool:
        return _page(a) == _page(b)

    def _same_section(a: dict, b: dict) -> bool:
        return _sec(a) == _sec(b)

    # стабильная сортировка, чтобы соседние кандидаты оказались рядом
    def _key(n: dict):
        return (n.get("type"),
                _sec(n),
                _page(n),
                _sent_idx(n))

    nodes_sorted = sorted(nodes, key=_key)

    out = []
    buf = []

    for n in nodes_sorted:
        if not buf:
            buf.append(n)
            continue

        last = buf[-1]

        # базовые условия
        same_type = (n.get("type") == last.get("type"))
        same_sec = _same_section(n, last)
        same_pg = _same_page(n, last)

        # индексы предложений — оба валидные и рядом
        si_last = _sent_idx(last)
        si_cur = _sent_idx(n)
        both_valid = (si_last >= 0 and si_cur >= 0)
        close_enough = both_valid and (abs(si_cur - si_last) <= 1)

        if same_type and same_sec and same_pg and close_enough:
            # склейка
            last_text = (last.get("text") or "").strip()
            cur_text = (n.get("text") or "").strip()
            if cur_text:
                last["text"] = (last_text + " " + cur_text).strip() if last_text else cur_text

            # усредняем уверенность (можно заменить на max, если предпочтительнее)
            try:
                a = float(last.get("conf") or 0.0)
                b = float(n.get("conf") or 0.0)
                last["conf"] = float(min(1.0, (a + b) / 2.0))
            except Exception:
                pass

            # объединяем span, если есть у обоих
            lp, np = _prov(last), _prov(n)
            ls, ns = lp.get("span"), np.get("span")
            if isinstance(ls, (list, tuple)) and len(ls) == 2 and isinstance(ns, (list, tuple)) and len(ns) == 2:
                try:
                    last["prov"]["span"] = [min(int(ls[0]), int(ns[0])), max(int(ls[1]), int(ns[1]))]
                except Exception:
                    # оставляем прежний span при ошибке преобразования
                    pass

            # page/section/ids оставляем от первого узла
        else:
            out.append(buf[-1])
            buf = [n]

    if buf:
        out.append(buf[-1])

    return out


# ─────────────────────────────────────────────────────────────
# Основной раннер
# ─────────────────────────────────────────────────────────────

def run_s1(s0_path: str,
           # Stage S1: load S0, apply patterns/heuristics, write s1_graph.json
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
    # NOTE: spacy_loader теперь возвращает loaded_paths, а не registry
    matcher, depmatcher, type_boosts, rule_configs, loaded_paths = load_spacy_patterns(
        nlp, themes_root or "/app/rules/themes", theme_override
    )
    themes_used = theme_override or ["common"]

    nodes: List[Dict[str, Any]] = []
    debug_records: List[Dict[str, Any]] = []
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
        tname, conf, dbg, rule_hits = _score_sentence(
            doc, text, imrad, matcher, depmatcher, type_boosts, rule_configs, dep_enabled
        )
        if tname is not None and dbg.get("scored"):
            tname = tiebreak_label(dbg["scored"], tname)

        # 1) Жёсткий guard библиографии/референсов
        if imrad == "REFERENCES" or looks_like_reference(text):
            # всё, что «похоже на библиографию», не должно становиться содержательными узлами
            tname, conf = "Other", 0.39

        # 2) Подружить S1 с «структурной» цитатой из S0:
        has_struct_cit = bool(sent.get("has_citation"))
        cit_strength = float(sent.get("citation_strength") or 0.0)

        # 3.a) Непрерывный бонус от цитат (используем S0 strength, иначе оцениваем по тексту)
        if tname in ("Input Fact", "Result", "Conclusion"):
            if cit_strength <= 0.0:
                cit_strength = _estimate_citation_strength(text)
            conf = min(1.0, conf + min(0.12, 0.08 * round(10 * cit_strength) / 10.0))

        # 3.b) Сохраняем жёсткий буст для Input Fact при структурной цитате (TEI <ref>)
        if has_struct_cit and tname == "Input Fact":
            conf = min(1.0, conf + 0.12)  # уменьшили до +0.12, т.к. есть общий бонус выше

        # 3.c) citation-soft: «почти целиком цитатное» — мягко понижаем
        if _citation_heavy(text):
            conf = max(0.0, conf - 0.08)
        # а если это ещё и Input Fact без структурной цитаты — можно «уронить» в Other при слабой уверенности
        if tname == "Input Fact" and not has_struct_cit and conf < (CONF_NODE_MIN + 0.02):
            tname, conf = "Other", 0.39

        # Более строгая и настраиваемая эвристика «фраз вывода»
        if tname and conf >= CONF_NODE_MIN and CONCLUSION_HINTS["enable"]:
            if imrad in CONCLUSION_HINTS["sections"]:
                tl = text.lower().strip()
                if tl.startswith(("these results", "overall", "in conclusion", "in summary")) and \
                        any(w in tl for w in ("suggest", "indicate", "support", "imply", "show")):
                    best_score = conf
                    concl_score = float((dbg.get("scored") or {}).get("Conclusion", 0.0))
                    if concl_score >= CONCLUSION_HINTS["min_ratio"] * best_score and tname != "Conclusion":
                        tname = "Conclusion"
                        if imrad == "DISCUSSION":
                            conf = max(conf, 0.52)

        # Caption-boost (только ПОСЛЕ citation guard и только для «содержательных» типов)
        if tname and tname in CAPTION_BOOST_TYPES and is_cap:
            if cap_type == "Figure":
                conf = min(1.0, conf + CAPTION_BOOST_FIG)
            elif cap_type == "Table":
                conf = min(1.0, conf + CAPTION_BOOST_TAB)

        # 4) Лёгкий bias в секции RESULTS: числовой фрагмент и слабый выбор класса → подтолкнуть к Result
        if tname in ("Other", "Input Fact", "Analysis") and imrad.startswith("RESULT") and _RX_NUMERIC_HINT.search(
                text):
            tname = "Result"
            conf = max(conf, 0.46)

        if not tname or conf < CONF_NODE_MIN:
            continue

        pol = _polarity(doc)

        # секционное имя (для фронта/аналитики)
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
        node["matched_rules"] = rule_hits
        rule_summary: Dict[str, int] = {}
        for rh in rule_hits:
            rname = rh.get("rule")
            if rname:
                rule_summary[rname] = rule_summary.get(rname, 0) + 1
        nodes.append(node)
        debug_records.append(
            {
                "label": tname,  # выбранный класс
                "text": text,
                "page": page,
                "bbox": bbox,
                "section": section_name,  # Body / FigureCaption / TableCaption
                "imrad": imrad,  # INTRO / METHODS / RESULTS / DISCUSSION / ...
                "conf": round(conf, 3),
                "hits": dbg.get("hits", {}),
                "rule_hits": rule_summary,
            }
        )
        idx += 1

    #nodes = merge_adjacent(nodes)

    # линковка
    edges = _link_inside(doc_id, nodes)

    # артефакты
    s1_graph = {
        "doc_id": doc_id,
        "themes_used": themes_used,  # НОВОЕ
        "pattern_sources": loaded_paths,
        "spacy_model": model_name,  # опционально, но полезно
        "dep_enabled": bool(dep_enabled),  # опционально
        "nodes": nodes,
        "edges": edges
    }
    (out_dir / "s1_graph.json").write_text(json.dumps(s1_graph, ensure_ascii=False, indent=2), encoding="utf-8")

    # Сохраняем JSONL/MD рядом с graph_path
    base = Path(graph_path)
    write_jsonl_per_sentence(str(base.with_suffix(".jsonl")), debug_records)
    write_markdown_summary(str(base.with_suffix(".md")), debug_records)

# pipeline/s1.py
from __future__ import annotations
import json
import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pipeline.themes_router import preload as themes_preload, route_themes, select_rule_files, ThemeRegistry
from pipeline.themes_router import explain_themes_for_debug, build_showcase_text


# ---------- Rule model ----------
@dataclass
class Rule:
    id: str
    type: str
    sections: set
    weight: float
    regex: re.Pattern
    negatives: List[re.Pattern]
    captures: List[str]


# ---------- Utils ----------
def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _section_match(sname: str, rule_sections: set) -> bool:
    """Мягкое сопоставление секций: равенство или подстрока в обе стороны."""
    if not rule_sections:
        return True
    s = _norm(sname)
    for rs in rule_sections:
        r = _norm(str(rs))
        if r == s or r in s or s in r:
            return True
    return False


def _contains_hedge(text: str, hedge_words: set) -> bool:
    t = _norm(text)
    return any(h in t for h in hedge_words)


def _polarity_from_text(text: str) -> str:
    t = (text or "").lower()
    neg = [
        "no evidence", "did not", "does not",
        "fail to", "fails to", "not significant", " ns ",
        "decrease in", "decreased", "reduced"
    ]
    pos = [
        "significant", "improved", "increase", "increased",
        "higher", "better", "enhanced", "fits well",
        "goodness of fit", "matches the data", "demonstrate that", "we show that"
    ]
    if any(p in t for p in pos):
        return "positive"
    if any(n in t for n in neg):
        return "negative"
    return "neutral"


def _make_node_id(doc_id: str, ntype: str, idx: int) -> str:
    return f"{doc_id}:{ntype}:{idx:04d}"


# ---------- Load rules with validation ----------
def load_rules(path: str):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Rules file is not a mapping: {path}")

    meta = cfg.get("meta", {}) or {}
    # совместимость: слова хеджинга могли лежать в meta.hedges или hedging.words
    hedges_list = (meta.get("hedges") or [])
    if not hedges_list:
        hedges_list = (cfg.get("hedging") or {}).get("words", []) or []
    hedge_words = set(hedges_list)

    raw = cfg.get("elements") or []
    if not isinstance(raw, list):
        raise ValueError(f"'elements' must be a list in {path}")

    rules: List[Rule] = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise ValueError(f"elements[{i}] is not a mapping in {path}")
        try:
            rid = r["id"]
            rtype = r["type"]
            patt = r["pattern"]
        except KeyError as ke:
            raise ValueError(f"Missing key {ke} in elements[{i}] (id={r.get('id')}) in {path}")

        sections = set(r.get("sections", []))
        weight = float(r.get("weight", 0.5))
        negatives = [re.compile(p, re.I | re.M | re.S) for p in r.get("negatives", [])]
        captures = r.get("captures", [])
        rules.append(Rule(
            id=rid, type=rtype, sections=sections, weight=weight,
            regex=re.compile(patt, re.I | re.M | re.S),
            negatives=negatives, captures=captures
        ))
    relations = cfg.get("relations", []) or []
    return meta, hedge_words, rules, relations


def _merge_dict_add(dst: Dict[str, float], src: Dict[str, float]):
    for k, v in (src or {}).items():
        try:
            dst[k] = float(v) if k not in dst else max(float(dst[k]), float(v))
        except Exception:
            continue


def load_rules_merged(paths: List[str]) -> Tuple[Dict[str, Any], set, List[Rule], List[Dict[str, Any]]]:
    """
    Загружает несколько YAML-файлов с правилами и аккуратно МЕРДЖИТ:
      - meta: section_weights (max), conf_boosts (max), hedging.words (union)
      - elements: конкатенация (Rule[]) — дубли id допустимы, но их лучше избегать линтером
      - relations: конкатенация
    """
    merged_meta: Dict[str, Any] = {"section_weights": {}, "conf_boosts": {}}
    hedges_all: set = set()
    all_rules: List[Rule] = []
    all_rel: List[Dict[str, Any]] = []

    for p in paths:
        meta, hedge_words, rules, relations = load_rules(p)
        # meta.section_weights
        _merge_dict_add(merged_meta.setdefault("section_weights", {}), meta.get("section_weights") or {})
        # meta.conf_boosts
        _merge_dict_add(merged_meta.setdefault("conf_boosts", {}), meta.get("conf_boosts") or {})
        # hedges
        hedges_all |= set(hedge_words or [])
        # elements & relations
        all_rules.extend(rules or [])
        all_rel.extend(relations or [])

    return merged_meta, hedges_all, all_rules, all_rel


# ---------- Matching ----------
def _yield_matches(text: str, rule: Rule):
    for m in rule.regex.finditer(text):
        span = (m.start(), m.end())
        frag = text[span[0]:span[1]]
        # negatives?
        if any(neg.search(frag) for neg in rule.negatives):
            continue
        yield frag, span, m


# --- Sentence expansion (robust to 60.0 and line breaks) ---
SENT_END = re.compile(r'[.!?]')
DOT_IN_NUMBER = re.compile(r'(\d)\.(\d)')  # 60.0 — не конец
HARD_BREAK = re.compile(r'(\n{2,}|[\r])')  # абзац


def expand_to_sentence_robust(text: str, span: Tuple[int, int], max_len: int = 600) -> str:
    n = len(text)
    start, end = span

    # влево
    l = start
    while l > 0:
        ch = text[l - 1]
        if HARD_BREAK.match(text[l - 2:l + 1] if l >= 2 else ''):
            break
        if SENT_END.match(ch):
            if l - 2 >= 0 and not DOT_IN_NUMBER.match(text[l - 2:l + 1]):
                break
        l -= 1
        if start - l > max_len:
            break

    # вправо
    r = end
    while r < n:
        ch = text[r]
        if HARD_BREAK.match(text[r:r + 2]):
            break
        if SENT_END.match(ch):
            if not (r - 1 >= 0 and DOT_IN_NUMBER.match(text[r - 1:r + 1])):
                r += 1
                break
        r += 1
        if r - end > max_len:
            break

    frag = text[l:r].strip()
    return frag


# --- Number/percent tidy (fix '% of' spacing safely) ---
def tidy_numbers_and_percents(frag: str) -> str:
    # "40 . 0" → "40.0"
    frag = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', frag)
    # если после % идёт буква (of/units/words) — ставим ОДИН пробел
    frag = re.sub(r'(?<=\d%)\s*(?=[A-Za-z])', ' ', frag)
    # если после % идёт пунктуация/скобка — пробел убираем
    frag = re.sub(r'(?<=\d%)\s+(?=[)\]\}\.,;:])', '', frag)
    # нормализуем множественные пробелы
    frag = re.sub(r'[ \t]{2,}', ' ', frag)
    return frag.strip()


def match_rules(doc_id: str,
                sections: List[Dict[str, Any]],
                captions: List[Dict[str, Any]],
                meta: Dict[str, Any],
                hedge_words: set,
                rules: List[Rule]) -> List[Dict[str, Any]]:
    section_weights = {_norm(k): float(v) for k, v in (meta.get("section_weights") or {}).items()}
    nodes: List[Dict[str, Any]] = []
    idx = 0

    conf_boosts = (meta.get("conf_boosts") or {})
    caption_boost = float(conf_boosts.get("caption", 0.08))
    capture_bonus = float(conf_boosts.get("capture", 0.05))
    short_penalty = float(conf_boosts.get("short_penalty", 0.06))

    # helper to compute confidence
    def conf_for(rule: Rule, sec_name: str, text: str, *, has_captures: bool, is_caption: bool) -> float:
        w_rule = rule.weight
        w_sec = section_weights.get(_norm(sec_name), 0.6)
        if is_caption:
            w_sec = min(1.0, w_sec + caption_boost)
        pen = 0.1 if _contains_hedge(text, hedge_words) else 0.0
        base = w_rule * w_sec - pen
        # бонус за валидные captures
        if has_captures:
            base += capture_bonus
        # штраф за крайне короткие фразы (меньше 40 символов)
        if len(text.strip()) < 40:
            base -= short_penalty
        return max(0.0, min(1.0, base))

    # 1) sections
    for sec in sections:
        sname = sec.get("name", "Unknown")
        text = sec.get("text", "")
        if not text:
            continue

        for rule in rules:
            if not _section_match(sname, rule.sections):
                continue
            for frag, span, m in _yield_matches(text, rule):
                idx += 1
                has_caps = bool(rule.captures and any((m.groupdict().get(c) or "") for c in rule.captures))
                node_text = tidy_numbers_and_percents(expand_to_sentence_robust(text, span))
                nodes.append({
                    "id": _make_node_id(doc_id, rule.type, idx),
                    "type": rule.type,
                    "label": rule.id,
                    "text": node_text,
                    "conf": round(conf_for(rule, sname, frag, has_captures=has_caps, is_caption=False), 3),
                    "polarity": _polarity_from_text(frag),
                    "prov": {
                        "section": sname,
                        "span": [int(span[0]), int(span[1])]
                    }
                })

    # 2) captions (сильные секции)
    for cap in captions:
        sname = "FigureCaption" if str(cap.get("id", "")).lower().startswith("figure") else "TableCaption"
        text = cap.get("text", "")
        if not text:
            continue
        for rule in rules:
            if not _section_match(sname, rule.sections):
                continue
            for frag, span, m in _yield_matches(text, rule):
                idx += 1
                has_caps = bool(rule.captures and any((m.groupdict().get(c) or "") for c in rule.captures))
                node_text = tidy_numbers_and_percents(expand_to_sentence_robust(text, span))
                nodes.append({
                    "id": _make_node_id(doc_id, rule.type, idx),
                    "type": rule.type,
                    "label": rule.id,
                    "text": node_text,
                    "conf": round(conf_for(rule, sname, frag, has_captures=has_caps, is_caption=True), 3),
                    "polarity": _polarity_from_text(frag),
                    "prov": {
                        "section": sname,
                        "caption_id": cap.get("id")
                    }
                })
    return nodes


# ---------- Post-match utilities (safe prov handling) ----------
def _collect_provs(n: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Собирает провенансы в унифицированном виде [{'section':..., 'span':[s,e]}, ...]."""
    provs: List[Dict[str, Any]] = []

    def norm_one(p) -> Optional[Dict[str, Any]]:
        if not isinstance(p, dict):
            return None
        if "span" in p and isinstance(p["span"], (list, tuple)) and len(p["span"]) == 2:
            s, e = int(p["span"][0]), int(p["span"][1])
        elif "spans" in p and isinstance(p["spans"], list) and p["spans"]:
            s, e = int(p["spans"][0][0]), int(p["spans"][0][1])
        else:
            return None
        return {"section": p.get("section", "Unknown"), "span": [s, e]}

    p = n.get("prov")
    if isinstance(p, dict):
        x = norm_one(p)
        if x: provs.append(x)
    elif isinstance(p, list):
        for it in p:
            x = norm_one(it)
            if x: provs.append(x)

    pm = n.get("prov_multi")
    if isinstance(pm, list):
        for it in pm:
            x = norm_one(it)
            if x: provs.append(x)

    return provs


def _span_and_section(n: Dict[str, Any]) -> Tuple[str, Tuple[int, int]]:
    """Возвращает ('section', (start, end)); если пров несколько — охватывающий интервал."""
    provs = _collect_provs(n)
    if not provs:
        return "Unknown", (0, 0)
    sec = provs[0]["section"] or "Unknown"
    starts = [p["span"][0] for p in provs]
    ends = [p["span"][1] for p in provs]
    return sec, (min(starts), max(ends))


def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    _, (s1, e1) = _span_and_section(a)
    _, (s2, e2) = _span_and_section(b)
    inter = max(0, min(e1, e2) - max(s1, s2))
    uni = (e1 - s1) + (e2 - s2) - inter
    return inter / uni if uni > 0 else 0.0


def suppress_overlaps(cands: List[Dict[str, Any]], iou_thr: float = 0.45) -> List[Dict[str, Any]]:
    """
    Если два узла одного типа/секции сильно перекрываются:
      - предпочитаем более 'специфичный' ID (pair > single),
      - затем более длинный текст,
      - затем больший conf.
    Безопасно работает при prov=dict|list и наличии prov_multi.
    """

    def sort_key(n):
        sec, (s, _) = _span_and_section(n)
        return (sec or "Unknown", int(s or 0))

    kept: List[Dict[str, Any]] = []
    for n in sorted(cands, key=sort_key):
        drop = False
        n_sec, _ = _span_and_section(n)
        for m in kept[:]:
            m_sec, _ = _span_and_section(m)
            if n["type"] != m["type"] or (n_sec or "Unknown") != (m_sec or "Unknown"):
                continue
            if _iou(n, m) < iou_thr:
                continue

            # Специфичность: 'pair' выигрывает у 'single'
            n_is_pair = "percent_worse_improve_pair" in (n.get("label") or "")
            m_is_pair = "percent_worse_improve_pair" in (m.get("label") or "")
            if n_is_pair and not m_is_pair:
                kept.remove(m)
                break
            if m_is_pair and not n_is_pair:
                drop = True
                break

            # Иначе — длина текста, затем conf
            ln, lm = len(n.get("text", "") or ""), len(m.get("text", "") or "")
            if ln > lm:
                kept.remove(m)
                break
            if ln == lm and float(n.get("conf", 0.0)) > float(m.get("conf", 0.0)):
                kept.remove(m)
                break

            drop = True
            break

        if not drop:
            kept.append(n)
    return kept


def merge_fragment_nodes(nodes: List[Dict[str, Any]], max_gap: int = 24) -> List[Dict[str, Any]]:
    """Склейка соседних фрагментов одного правила/типа в пределах max_gap по спанам."""

    def sort_key(n):
        sec, (s, _) = _span_and_section(n)
        return (n.get("type") or "", n.get("label") or "", sec or "Unknown", int(s or 0))

    nodes = sorted(nodes, key=sort_key)
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(nodes):
        cur = dict(nodes[i]);
        i += 1
        cur_sec, (cur_s, cur_e) = _span_and_section(cur)
        while i < len(nodes):
            nxt = nodes[i]
            same = (
                    nxt.get("type") == cur.get("type")
                    and (nxt.get("label") == cur.get("label"))
                    and _span_and_section(nxt)[0] == cur_sec
            )
            nxt_s, nxt_e = _span_and_section(nxt)[1]
            near = nxt_s - cur_e <= max_gap
            if not (same and near):
                break
            # склеиваем
            cur["text"] = (cur.get("text", "") + " " + (nxt.get("text", "") or "")).strip()
            # расширяем span (пишем обратно в prov как охватывающий)
            cur["prov"] = {"section": cur_sec, "span": [min(cur_s, nxt_s), max(cur_e, nxt_e)]}
            cur_s, cur_e = min(cur_s, nxt_s), max(cur_e, nxt_e)
            cur["conf"] = max(float(cur.get("conf", 0.0)), float(nxt.get("conf", 0.0)))
            i += 1
        out.append(cur)
    return out


def drop_nested_overlaps(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Удаляет узлы, полностью вложенные в другие (при равном типе + не короче текст)."""

    def sort_key(n):
        _, (s, e) = _span_and_section(n)
        return (int(s or 0), -int(e or 0))

    nodes = sorted(nodes, key=sort_key)
    keep: List[Dict[str, Any]] = []
    for n in nodes:
        s1, e1 = _span_and_section(n)[1]
        nested = False
        for m in nodes:
            if m is n:
                continue
            s2, e2 = _span_and_section(m)[1]
            if s2 <= s1 and e1 <= e2 and len((m.get("text") or "")) >= len((n.get("text") or "")):
                nested = True
                break
        if not nested:
            keep.append(n)
    return keep


# ---------- Linking (intra-paper) ----------
def link_inside(doc_id: str, nodes: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        by_type.setdefault(n["type"], []).append(n)

    seen = set()

    def add(frm, to, etype, conf=0.6, hint="rule"):
        key = (frm["id"], to["id"], etype)
        if key in seen:
            return
        seen.add(key)
        edges.append({
            "from": frm["id"], "to": to["id"], "type": etype,
            "conf": round(conf, 3),
            "prov": {
                "hint": hint,
                "from_section": _span_and_section(frm)[0],
                "to_section": _span_and_section(to)[0]
            }
        })
        # пост-хок буст уверенности узлов, вошедших в ребро
        boost = float((meta.get("conf_boosts") or {}).get("edge_participation", 0.02))
        for n in (frm, to):
            n["conf"] = float(min(1.0, (n.get("conf") or 0.0) + boost))

    def _order_by_proximity(a, bs):
        # по секции и расстоянию начала спана
        a_sec, (sa, _) = _span_and_section(a)

        def dist(b):
            b_sec, (sb, _) = _span_and_section(b)
            same = 0 if a_sec == b_sec else 100000
            return same + abs((sa or 0) - (sb or 0))

        return sorted(bs, key=dist)

    # Technique/Method -> Result/Experiment
    for m in by_type.get("Technique", []) + by_type.get("Method", []):
        neigh_r = _order_by_proximity(m, by_type.get("Result", []))[:2]
        neigh_e = _order_by_proximity(m, by_type.get("Experiment", []))[:1]
        for r in neigh_r:
            add(m, r, "uses", conf=0.64, hint="prox")
        for e in neigh_e:
            add(m, e, "uses", conf=0.62, hint="prox")

    # Dataset -> Experiment
    for d in by_type.get("Dataset", []):
        neigh_e = _order_by_proximity(d, by_type.get("Experiment", []))[:1]
        for e in neigh_e:
            add(d, e, "feeds", conf=0.60, hint="prox")

    # Experiment -> Result
    for ex in by_type.get("Experiment", []):
        neigh_r = _order_by_proximity(ex, by_type.get("Result", []))[:2]
        for r in neigh_r:
            add(ex, r, "produces", conf=0.62, hint="prox")

    # Result -> Hypothesis
    for r in by_type.get("Result", []):
        neigh_h = _order_by_proximity(r, by_type.get("Hypothesis", []))[:2]
        for h in neigh_h:
            et = "supports" if r.get("polarity") != "negative" else "refutes"
            add(r, h, et, conf=0.63, hint="prox")

    # Input Fact -> Hypothesis
    for f in by_type.get("Input Fact", []):
        neigh_h = _order_by_proximity(f, by_type.get("Hypothesis", []))[:2]
        for h in neigh_h:
            add(f, h, "informs", conf=0.58, hint="prox")

    # Result/Dataset/Technique -> Analysis (чтобы в усечённых случаях были рёбра)
    for a in by_type.get("Analysis", []):
        for r in _order_by_proximity(a, by_type.get("Result", []))[:1]:
            add(r, a, "informs", conf=0.57, hint="prox")
        for d in _order_by_proximity(a, by_type.get("Dataset", []))[:1]:
            add(d, a, "informs", conf=0.56, hint="prox")
        for t in _order_by_proximity(a, by_type.get("Technique", []))[:1]:
            add(t, a, "uses", conf=0.56, hint="prox")

    return edges


# ---------- Gaps (S1.5 seeds) ----------
_SENT_MIN = 40
_SENT_MAX = 400
_SENT_SPLIT = re.compile(r'[^.!?\n]{40,400}[.!?]')


def _split_sentences(text: str):
    """Быстрая нарезка на псевдо-предложения по длине и финальной пунктуации."""
    for m in _SENT_SPLIT.finditer(text or ""):
        yield m.group(0).strip()


def _compile_seed_regexes(meta: Dict[str, Any]):
    pats = (meta.get("seeds", {}) or {}).get("regexes", []) or []
    return [re.compile(p, re.I | re.S) for p in pats]


def _looks_like_seed(sent: str, seed_res: List[re.Pattern]) -> bool:
    t = sent or ""
    if len(t) < _SENT_MIN or len(t) > _SENT_MAX:
        return False
    return any(r.search(t) for r in seed_res)


def _has_lexicon(sent: str, meta: Dict[str, Any]) -> bool:
    lex = (meta.get("seeds", {}) or {}).get("method_lexicon", []) or []
    t = _norm(sent)
    return any(w in t for w in lex)


def gather_gaps(sections: List[Dict[str, Any]], meta: Dict[str, Any], max_gaps: int = 50) -> List[Dict[str, str]]:
    """Возвращает список {section, text} — предложения-кандидаты, похожие на элементы, но не пойманные правилами."""
    seed_res = _compile_seed_regexes(meta)
    section_weights = {_norm(k): float(v) for k, v in (meta.get("section_weights") or {}).items()}
    min_w = float((meta.get("seeds", {}) or {}).get("min_section_weight", 0.5))

    out: List[Dict[str, str]] = []
    for sec in sections:
        sname = sec.get("name", "Unknown") or "Unknown"
        text = sec.get("text", "") or ""
        if not text:
            continue
        if section_weights.get(_norm(sname), 0.0) < min_w:
            continue

        for sent in _split_sentences(text):
            if _looks_like_seed(sent, seed_res):
                if _has_lexicon(sent, meta) or not (meta.get("seeds", {}) or {}).get("method_lexicon"):
                    out.append({"section": sname, "text": sent[:420]})
                    if len(out) >= max_gaps:
                        return out
    return out


def _safe_preload(themes_root: str | Path) -> ThemeRegistry:
    try:
        return themes_preload(themes_root)
    except Exception:
        return ThemeRegistry(Path(themes_root), {}, {})


# ---------- Runner ----------
def run_s1(s0_path: str, rules_path: str, out_path: str, *,
           themes_root: str = "/app/rules/themes",
           theme_override: Optional[List[str]] = None,
           theme_registry: Optional[ThemeRegistry] = None) -> Dict[str, Any]:
    """
    1) грузит s0.json и rules (common.yaml + learned packs);
    2) матч правил по секциям/капшенам -> кандидаты узлов;
    3) постобработка (склейка/анти-дубликат/супресс перекрытий);
    4) пороги -> nodes;
    5) линковка -> edges;
    6) собирает debug, включая gaps;
    7) пишет s1_graph.json и s1_debug.json.
    """
    # --- Загрузка артефактов S0 ---
    s0 = json.loads(Path(s0_path).read_text(encoding="utf-8"))
    doc_id = s0.get("doc_id") or Path(s0_path).parent.name
    sections = s0.get("sections", []) or []
    captions = s0.get("captions", []) or []

    # --- Тематический роутинг ---
    registry = theme_registry if theme_registry is not None else _safe_preload(themes_root)
    chosen = route_themes(
        s0, registry=registry,
        global_topk=2, stop_threshold=1.8,
        override=theme_override
    )

    # --- выбрать файлы правил/лексиконов по выбранным темам ---
    files = select_rule_files(
        themes_root=themes_root,  # напр. "/app/rules/themes"
        chosen=chosen,
        common_path=rules_path  # напр. "/app/rules/common.yaml"
    )

    rule_files = [str(p) for p in files.get("rules", [])]
    lexicon_files = [str(p) for p in files.get("lexicons", [])]

    # --- мерж правил из нескольких файлов ---
    meta, hedge_words, rule_objs, relations = load_rules_merged(rule_files)

    # --- домножаем веса тематических правил на theme_score ---
    _apply_theme_weight_mix(rule_objs, chosen)

    # --- подмешиваем shared-lexicon и тематические лексиконы ---
    abbr_map = {}
    synonyms = set()
    hedge_extra = set()

    for lp in lexicon_files:
        lx = load_lexicon_yaml(lp)
        for short, long in lx["abbr"]:
            if isinstance(short, str) and isinstance(long, str):
                abbr_map[short.lower()] = long
        for a, b in lx["synonyms"]:
            if isinstance(a, str) and isinstance(b, str):
                synonyms.add(tuple(sorted([a.lower(), b.lower()])))
        for w in lx["hedging_extra"]:
            if isinstance(w, str):
                hedge_extra.add(w.lower())

    # hedging = из правил + из лексиконов
    hedge_words = set(hedge_words or [])
    hedge_words |= hedge_extra

    # --- Кандидаты узлов по правилам ---
    candidates = match_rules(
        doc_id=doc_id,
        sections=sections,
        captions=captions,
        meta=meta,
        hedge_words=hedge_words,
        rules=rule_objs
    )

    # пост-склейка и анти-дубликаты
    candidates = merge_fragment_nodes(candidates, max_gap=20)
    candidates = drop_nested_overlaps(candidates)
    candidates = suppress_overlaps(candidates, iou_thr=0.45)

    # --- Пороги для узлов ---
    node_thr = float((meta.get("conf_thresholds") or {}).get("node", 0.40))
    nodes = [n for n in candidates if float(n.get("conf", 0.0)) >= node_thr]

    # --- Линковка внутри статьи и порог по рёбрам ---
    edges = link_inside(doc_id, nodes, meta)
    edge_thr = float((meta.get("conf_thresholds") or {}).get("edge", 0.55))
    edges = [e for e in edges if float(e.get("conf", 0.0)) >= edge_thr]

    # --- Сбор семян для авто-правил (gaps) ---
    try:
        gaps = gather_gaps(sections, meta, max_gaps=50)
    except Exception:
        gaps = []

    # --- Итог для фронта (S1) ---
    s1_graph = {
        "doc_id": doc_id,
        "nodes": nodes,
        "edges": edges
    }

    # подробный блок по темам: скор, выбранность, топ-триггеры
    theme_dbg = explain_themes_for_debug(s0, registry, chosen)

    # --- Отладочная сводка ---
    debug = {"doc_id": doc_id, "summary": {
        "candidates_total": len(candidates),
        "nodes_after_threshold": len(nodes),
        "edges_after_threshold": len(edges),
        "node_threshold": node_thr,
        "edge_threshold": edge_thr,
        "section_names": list({(s.get("name") or "Unknown") for s in sections}),
        "gap_count": len(gaps)
    }, "samples": {
        "candidates_head": candidates[:20],
        "gaps_head": gaps[:20]
    }, "theme_routing": theme_dbg, "rules_loaded": rule_files, "lexicons_loaded": lexicon_files}

    # --- Запись артефактов ---
    outp = Path(out_path)
    out_dir = outp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "s1_graph.json").write_text(json.dumps(s1_graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "s1_debug.json").write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    return s1_graph


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_s1(args.s0, args.rules, args.out)


def load_lexicon_yaml(path: str) -> dict:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {
        "abbr": d.get("abbr", []) or [],
        "synonyms": d.get("synonyms", []) or [],
        "hedging_extra": d.get("hedging_extra", []) or [],
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _apply_theme_weight_mix(rule_objs: List[Rule], chosen_themes) -> None:
    """Домножаем rule.weight для правил, чьи id начинаются с '<theme>/'."""
    # name -> score
    tmap = {t.name: float(t.score) for t in (chosen_themes or [])}
    if not tmap:
        return
    for r in rule_objs:
        rid = r.id or ""
        for tname, score in tmap.items():
            if rid.startswith(f"{tname}/"):
                scale = _clamp(1.0 + 0.12 * score, 0.85, 1.45)
                r.weight = float(round(r.weight * scale, 6))
                break

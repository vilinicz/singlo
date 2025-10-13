# pipeline/s1.py
from __future__ import annotations
import json, re, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


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
    neg = ["no evidence", "did not", "does not", "fail to", "fails to", "not significant", "ns", "decrease in", "decreased", "reduced"]
    pos = ["significant", "improved", "increase", "increased", "higher", "better", "enhanced",
           "fits well", "goodness of fit", "matches the data", "demonstrate that", "we show that"]
    if any(p in t for p in pos):
        return "positive"
    if any(n in t for n in neg):
        return "negative"
    return "neutral"


def _make_node_id(doc_id: str, ntype: str, idx: int) -> str:
    return f"{doc_id}:{ntype}:{idx:04d}"


# ---------- Load rules with validation ----------
def load_rules(path: str):
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Rules file is not a mapping: {path}")

    meta = cfg.get("meta", {}) or {}
    hedges = set((cfg.get("hedging") or {}).get("words", []) or [])
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
    return meta, hedges, rules, relations


# ---------- Matching ----------
def _yield_matches(text: str, rule: Rule):
    for m in rule.regex.finditer(text):
        span = (m.start(), m.end())
        frag = text[span[0]:span[1]]
        # negatives?
        if any(neg.search(frag) for neg in rule.negatives):
            continue
        yield frag, span, m

SENT_BOUND = re.compile(r'[.!?;]\s+|\n+')

def expand_to_sentence(text: str, span: tuple[int,int], max_len: int = 320) -> str:
    s, e = span
    # влево до ближайшей границы
    left = max(text.rfind('. ', 0, s), text.rfind('? ', 0, s), text.rfind('! ', 0, s), text.rfind('; ', 0, s), text.rfind('\n', 0, s))
    left = 0 if left == -1 else left + 2 if text[left:left+2] in ('. ','? ', '! ', '; ') else left + 1
    # вправо до ближайшей границы
    right_candidates = [text.find(ch, e) for ch in ('.','?','!',';','\n') if text.find(ch, e) != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    frag = text[left:right].strip()
    if len(frag) > max_len:
        frag = frag[:max_len].rsplit(' ',1)[0] + '…'
    return frag

def match_rules(doc_id: str,
                sections: List[Dict[str, Any]],
                captions: List[Dict[str, Any]],
                meta: Dict[str, Any],
                hedge_words: set,
                rules: List[Rule]) -> List[Dict[str, Any]]:
    section_weights = {_norm(k): v for k, v in (meta.get("section_weights") or {}).items()}
    nodes: List[Dict[str, Any]] = []
    idx = 0

    # helper to compute confidence
    def conf_for(rule: Rule, sec_name: str, text: str) -> float:
        w_rule = rule.weight
        w_sec = section_weights.get(_norm(sec_name), 0.6)
        pen = 0.1 if _contains_hedge(text, hedge_words) else 0.0
        base = w_rule * w_sec - pen
        return max(0.0, min(1.0, base))

    # 1) sections
    for sec in sections:
        sname = sec.get("name", "Unknown")
        if not any(_section_match(sname, r.sections) for r in rules):
            # нет ни одного правила, куда секция подходит — но не фильтруем;
            pass
        text = sec.get("text", "")
        if not text:
            continue

        for rule in rules:
            if not _section_match(sname, rule.sections):
                continue
            for frag, span, m in _yield_matches(text, rule):
                idx += 1
                nodes.append({
                    "id": _make_node_id(doc_id, rule.type, idx),
                    "type": rule.type,
                    "label": rule.id,
                    "text": expand_to_sentence(text, span).strip(),
                    "conf": round(conf_for(rule, sname, frag), 3),
                    "polarity": _polarity_from_text(frag),
                    "prov": {
                        "section": sname,
                        "span": [int(span[0]), int(span[1])]
                    }
                })

    # 2) captions as strong sections
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
                nodes.append({
                    "id": _make_node_id(doc_id, rule.type, idx),
                    "type": rule.type,
                    "label": rule.id,
                    "text": expand_to_sentence(text, span).strip(),
                    "conf": round(conf_for(rule, sname, frag), 3),
                    "polarity": _polarity_from_text(frag),
                    "prov": {
                        "section": sname,
                        "caption_id": cap.get("id")
                    }
                })

    return nodes


# ---------- Linking (intra-paper) ----------
def link_inside(doc_id: str, nodes: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    # naive proximity by order
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        by_type.setdefault(n["type"], []).append(n)

    seen = set()

    def add(frm, to, etype, conf=0.6, hint="rule"):
        key = (frm["id"], to["id"], etype)
        if key in seen: return
        seen.add(key)
        edges.append({
            "from": frm["id"], "to": to["id"], "type": etype,
            "conf": round(conf, 3),
            "prov": {"hint": hint, "from_section": frm["prov"].get("section"), "to_section": to["prov"].get("section")}
        })

    def _order_by_proximity(a, bs):
        # по секции и по расстоянию span, если доступно
        def dist(b):
            sa = a.get("prov", {}).get("span", [0, 0])[0]
            sb = b.get("prov", {}).get("span", [0, 0])[0]
            same = 0 if a["prov"].get("section") == b["prov"].get("section") else 100000
            return same + abs(sa - sb)

        return sorted(bs, key=dist)

    # Method -> Result/Experiment (только ближайшие 2)
    for m in by_type.get("Method", []):
        neigh_r = _order_by_proximity(m, by_type.get("Result", []))[:2]
        neigh_e = _order_by_proximity(m, by_type.get("Experiment", []))[:1]
        for r in neigh_r:
            add(m, r, "uses", conf=0.64, hint="prox")
        for e in neigh_e:
            add(m, e, "uses", conf=0.62, hint="prox")

    # Result -> Hypothesis (только ближайшая 1–2)
    for r in by_type.get("Result", []):
        neigh_h = _order_by_proximity(r, by_type.get("Hypothesis", []))[:2]
        for h in neigh_h:
            et = "supports" if r.get("polarity") != "negative" else "refutes"
            add(r, h, et, conf=0.63, hint="prox")

    return edges


# ---------- Runner ----------
def run_s1(s0_path: str, rules_path: str, out_path: str) -> Dict[str, Any]:
    meta, hedge_words, rule_objs, relations = load_rules(rules_path)
    s0 = json.loads(Path(s0_path).read_text())
    # doc_id — из s0.json (он у тебя теперь = имени директории)
    doc_id = s0.get("doc_id") or Path(s0_path).parent.name

    sections = s0.get("sections", [])
    captions = s0.get("captions", [])

    # кандидаты
    candidates = match_rules(doc_id, sections, captions, meta, hedge_words, rule_objs)

    # фильтры по порогам
    node_thr = float(meta.get("conf_thresholds", {}).get("node", 0.55))
    nodes = [n for n in candidates if n["conf"] >= node_thr]

    edges = link_inside(doc_id, nodes, meta)
    edge_thr = float(meta.get("conf_thresholds", {}).get("edge", 0.6))
    edges = [e for e in edges if e["conf"] >= edge_thr]

    # сохраняем как s1_graph.json (итоговый graph.json пишет S2)
    outp = Path(out_path)
    out_dir = outp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_graph = {"doc_id": doc_id, "nodes": nodes, "edges": edges}
    (out_dir / "s1_graph.json").write_text(json.dumps(s1_graph, ensure_ascii=False, indent=2))

    # отладка
    debug = {
        "doc_id": doc_id,
        "summary": {
            "candidates_total": len(candidates),
            "nodes_after_threshold": len(nodes),
            "edges_after_threshold": len(edges),
            "node_threshold": node_thr,
            "edge_threshold": edge_thr,
            "section_names": list({s.get("name", "Unknown") for s in sections}),
        },
        "samples": {
            "candidates_head": candidates[:20],
        }
    }
    (out_dir / "s1_debug.json").write_text(json.dumps(debug, ensure_ascii=False, indent=2))
    return s1_graph


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_s1(args.s0, args.rules, args.out)

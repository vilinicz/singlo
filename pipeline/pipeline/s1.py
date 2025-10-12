# pipeline/pipeline/s1.py
"""
S1 — Rule-Based Extraction (MVP)
--------------------------------
Берёт S0-вывод (s0.json) и rules/common.yaml, извлекает узлы/связи
без LLM. Полярность (supports/refutes) определяется упрощённо по regex.

Вход (s0.json, минимально):
{
  "doc_id": "demo",
  "sections": [{"name": "Results", "text": "..."}, ...],
  "captions": [{"id":"Fig1","text":"Figure 1: ..."}, ...]
}

Правила: rules/common.yaml (см. пример из ответа)

Выход (graph.json):
{
  "doc_id": "demo",
  "nodes": [{"id":"demo:Result:001","type":"Result","text":"...","prov":{...},"conf":0.83,"polarity":"positive"}],
  "edges": [{"from":"...","to":"...","type":"produces","prov":{...},"conf":0.75}]
}
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

import yaml

# ------------------------- Rules model -------------------------


@dataclass
class Rule:
    id: str
    type: str
    sections: Set[str]
    weight: float
    regex: Pattern
    negatives: List[Pattern]
    captures: List[str]


def _rx(p: str) -> Pattern:
    return re.compile(p, flags=re.I | re.M)


def load_rules(path: str) -> Tuple[Dict[str, Any], Set[str], List[Rule], List[Dict[str, Any]]]:
    cfg = yaml.safe_load(Path(path).read_text())
    meta = cfg.get("meta", {})
    hedge_words = set(cfg.get("hedging", {}).get("words", []))

    rules: List[Rule] = []
    for r in cfg.get("elements", []):
        rules.append(
            Rule(
                id=r["id"],
                type=r["type"],
                sections=set(r.get("sections", [])),
                weight=float(r.get("weight", 0.5)),
                regex=_rx(r["pattern"]),
                negatives=[_rx(p) for p in r.get("negatives", [])],
                captures=r.get("captures", []),
            )
        )

    relations = cfg.get("relations", [])
    return meta, hedge_words, rules, relations


# ------------------------- Helpers -------------------------


def section_weight(meta: Dict[str, Any], name: str) -> float:
    return float(meta.get("section_weights", {}).get(name, 0.5))


def hedging_penalty(text: str, hedges: Set[str]) -> float:
    t = text.lower()
    hits = sum(1 for w in hedges if w in t)
    # Каждая hedge-лексема слегка понижает уверенность:
    return 0.15 * hits


def dedup_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for n in nodes:
        key = (n["type"], n["text"].lower().strip(), n["prov"]["section"])
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


# Простой детектор отрицательной полярности (без spaCy/negspacy)
NEG_PAT = re.compile(
    r"(?i)\b("
    r"no|not|did\s+not|does\s+not|failed\s+to|without|lack\s+of|"
    r"insignificant|non[-\s]?significant|"
    r"contrary\s+to|inconsistent\s+with|"
    r"\bns\b|not\s+significant"
    r")\b"
)


def set_polarity(node: Dict[str, Any]) -> Dict[str, Any]:
    """Бинарная полярность для Result/Conclusion на основе regex."""
    if node["type"] not in ("Result", "Conclusion"):
        return node

    txt = node["text"].lower()
    neg = bool(NEG_PAT.search(txt))

    # Локальные эвристики для частых формулировок:
    # "not increase" → negative; "not decrease" → positive (эффект в «правильную» сторону не подтверждён)
    if "increase" in txt and re.search(r"(?i)\bnot\b.{0,12}\bincrease", txt):
        neg = True
    if "decrease" in txt and re.search(r"(?i)\bnot\b.{0,12}\bdecrease", txt):
        # "did not decrease" — значит «снижения нет» → нет положительного исхода
        neg = True

    node["polarity"] = "negative" if neg else "positive"
    return node


# ------------------------- Core matching -------------------------


def match_rules(
    doc_id: str,
    sections: List[Dict[str, str]],
    captions: Optional[List[Dict[str, str]]],
    meta: Dict[str, Any],
    hedges: Set[str],
    rules: List[Rule],
) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []

    # Виртуальные секции для caption’ов, чтобы правила для Results могли матчить подписи
    fig_text = "\n".join([c["text"] for c in (captions or []) if c.get("id", "").lower().startswith("fig")])
    tab_text = "\n".join([c["text"] for c in (captions or []) if c.get("id", "").lower().startswith("table")])
    virtual_sections = []
    if fig_text.strip():
        virtual_sections.append({"name": "FigureCaption", "text": fig_text})
    if tab_text.strip():
        virtual_sections.append({"name": "TableCaption", "text": tab_text})

    for sec in (sections or []) + virtual_sections:
        sname, text = sec.get("name", "Unknown"), sec.get("text", "")
        if not text:
            continue

        for rule in rules:
            if rule.sections and sname not in rule.sections:
                continue

            for m in rule.regex.finditer(text):
                span = (m.start(), m.end())
                frag = text[span[0] : span[1]]

                # негативные шаблоны правила
                if any(neg.search(frag) for neg in rule.negatives):
                    continue

                conf = rule.weight * section_weight(meta, sname)
                conf -= hedging_penalty(frag, hedges)
                conf = max(0.0, min(1.0, conf))

                node = {
                    "id": f"{doc_id}:{rule.type}:{len(nodes)+1:03d}",
                    "type": rule.type,
                    "text": frag.strip(),
                    "attrs": {},
                    "prov": {"section": sname, "char": [span[0], span[1]], "rule": rule.id},
                    "conf": round(conf, 3),
                }
                nodes.append(node)

    # Полярность только для Result/Conclusion
    nodes = [set_polarity(n) if n["type"] in ("Result", "Conclusion") else n for n in nodes]
    # Дедуп
    nodes = dedup_nodes(nodes)
    return nodes


def link_inside(doc_id: str, nodes: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Простая внутри-документная линковка:
    - Experiment -> Result : produces (same_section)
    - Method -> Experiment/Result : uses (section_context)
    - Result -> Hypothesis : supports/refutes (по polarity)
    """
    edges: List[Dict[str, Any]] = []
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        by_type.setdefault(n["type"], []).append(n)

    def add_edge(a, b, etype, conf=0.65, hint="near"):
        edges.append(
            {
                "from": a["id"],
                "to": b["id"],
                "type": etype,
                "prov": {"section": a["prov"]["section"], "hint": hint},
                "conf": round(conf, 3),
            }
        )

    # produces: Experiment -> Result (в одной секции)
    for e in by_type.get("Experiment", []):
        for r in by_type.get("Result", []):
            if e["prov"]["section"] == r["prov"]["section"]:
                add_edge(e, r, "produces", conf=0.75, hint="same_section")

    # uses: Method -> Experiment/Result
    for m in by_type.get("Method", []):
        for t in by_type.get("Experiment", []) + by_type.get("Result", []):
            if m["prov"]["section"] in ("Methods", "Results"):
                add_edge(m, t, "uses", conf=0.6, hint="section_context")

    # supports/refutes: Result -> Hypothesis
    for r in by_type.get("Result", []):
        for h in by_type.get("Hypothesis", []):
            et = "refutes" if r.get("polarity") == "negative" else "supports"
            add_edge(r, h, et, conf=0.65, hint="basic_polarity")

    return edges


# ------------------------- Public API -------------------------


def run_s1(s0_path: str, rules_path: str, out_path: str) -> Dict[str, Any]:
    meta, hedge_words, rule_objs, relations = load_rules(rules_path)
    s0 = json.loads(Path(s0_path).read_text())
    doc_id = s0.get("doc_id") or Path(s0_path).parent.name

    sections = s0.get("sections", [])
    captions = s0.get("captions", [])

    nodes = match_rules(doc_id, sections, captions, meta, hedge_words, rule_objs)

    # Порог по уверенности для узлов
    node_thr = float(meta.get("conf_thresholds", {}).get("node", 0.55))
    nodes = [n for n in nodes if n["conf"] >= node_thr]

    # Внутри-документные рёбра
    edges = link_inside(doc_id, nodes, meta)

    # Порог для рёбер
    edge_thr = float(meta.get("conf_thresholds", {}).get("edge", 0.6))
    edges = [e for e in edges if e["conf"] >= edge_thr]

    out = {"doc_id": doc_id, "nodes": nodes, "edges": edges}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out

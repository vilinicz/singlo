# -*- coding: utf-8 -*-
"""
S2 — Normalize & Graph (enhanced)

- Canonicalize node types to 8-slot schema and gently retype by label keywords.
- Deduplicate nodes with fingerprints + merge provenance; cluster near-duplicates within type.
- Validate and rescore edges using proximity (section/distance) and lexical overlap.
- Strict whitelist of allowed edge directions; remap supports/refutes by polarity when possible.
- Add a limited, proximity-aware fallback backbone to avoid empty/disconnected graphs.
- Assign stable positions for preset layout: 8 fixed columns in the order:
  Input Fact, Hypothesis, Experiment, Technique, Result, Dataset, Analysis, Conclusion.
- Emit rich s2_debug with drop reasons and a light quality score (Q: SC/GC/QF/PC).

Entry point:
    run_s2(s1_path: str, out_path: str) -> Dict
"""

from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# ----------------------------- constants & schema -----------------------------

CANON_TYPES = (
    "Input Fact",
    "Hypothesis",
    "Experiment",
    "Technique",
    "Result",
    "Dataset",
    "Analysis",
    "Conclusion",
)

TYPE_ORDER: Dict[str, int] = {t: i for i, t in enumerate(CANON_TYPES)}

# synonyms → canonical
TYPE_SYNONYMS = {
    "input": "Input Fact",
    "inputfact": "Input Fact",
    "fact": "Input Fact",
    "hypothesis": "Hypothesis",
    "method": "Technique",  # если S1 даёт "Method", считаем это Technique (см. retyping по label ниже)
    "technique": "Technique",
    "experiment": "Experiment",
    "result": "Result",
    "observation": "Result",
    "dataset": "Dataset",
    "data": "Dataset",
    "analysis": "Analysis",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
}

# допустимые направления (from_type -> set(to_type))
ALLOWED_EDGE_PAIRS: Dict[str, Set[str]] = {
    "Input Fact": {"Hypothesis", "Analysis"},
    "Hypothesis": {"Experiment", "Analysis", "Conclusion"},
    "Technique": {"Experiment", "Result", "Analysis"},
    "Experiment": {"Result", "Dataset", "Analysis"},
    "Dataset": {"Experiment", "Analysis"},
    "Result": {"Hypothesis", "Analysis", "Conclusion"},
    "Analysis": {"Result", "Conclusion", "Hypothesis"},
    "Conclusion": {"Hypothesis", "Analysis"},
}

# допустимые типы рёбер — не ограничиваем жёстко, но нормализуем и ремапим при необходимости
CANON_EDGE_TYPES = {"uses", "produces", "supports", "refutes", "feeds", "informs"}

# приоритет, если несколько рёбер между одинаковой парой узлов
EDGE_PRIORITY = {"uses": 5, "produces": 4, "supports": 3, "refutes": 3, "feeds": 2, "informs": 1}


# ----------------------------- small utilities -------------------------------

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _first_span_start(n: Dict[str, Any]) -> int:
    sp = n.get("prov", {}).get("span")
    if isinstance(sp, list) and len(sp) >= 1:
        return _safe_int(sp[0], 1_000_000_000)
    return 1_000_000_000


def _prov_section(n: Dict[str, Any]) -> str:
    return (n.get("prov", {}).get("section") or "").strip()


def _norm_txt_for_fp(t: str) -> str:
    # нормализация текста для фингерпринта: lower, убираем пунктуацию, сворачиваем числа/проценты
    t = (t or "").lower()
    t = re.sub(r"\b\d+(\.\d+)?\s*%", "<PCT>", t)
    t = re.sub(r"\b\d+(?:[\.,]\d+)?\b", "<NUM>", t)
    t = re.sub(r"[\s\-_/]+", " ", t)
    t = re.sub(r"[^\w <>]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize_content(t: str) -> List[str]:
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9% ]+", " ", t)
    toks = [w for w in t.split() if
            len(w) > 2 and w not in {"the", "and", "for", "with", "are", "was", "were", "that", "this", "from", "into",
                                     "onto", "our"}]
    return toks


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _canon_type(t: str) -> str:
    t0 = (t or "").strip().lower().replace("-", "").replace("_", "").replace("  ", " ")
    return TYPE_SYNONYMS.get(t0, t if t in CANON_TYPES else (t.title() if t else "Analysis"))


def _guess_type_from_label(label: str) -> Optional[str]:
    s = (label or "").lower()
    # ориентиры по id правилам: "*/Hypothesis/*" и т.п.
    for key, canon in [
        ("hypothesis", "Hypothesis"),
        ("input", "Input Fact"),
        ("technique", "Technique"),
        ("method", "Technique"),
        ("experiment", "Experiment"),
        ("result", "Result"),
        ("dataset", "Dataset"),
        ("analysis", "Analysis"),
        ("conclusion", "Conclusion"),
    ]:
        if f"/{key}/" in s or s.startswith(f"{key}/") or s.endswith(f"/{key}"):
            return canon
    return None


def _edge_type_canon(s: str) -> str:
    s = (s or "").lower()
    if s in CANON_EDGE_TYPES:
        return s
    # лёгкий мап common-синонимов
    if s in {"used", "apply", "applies"}: return "uses"
    if s in {"produce", "produced", "yields"}: return "produces"
    if s in {"support"}: return "supports"
    if s in {"refute"}: return "refutes"
    if s in {"feed", "feeds into"}: return "feeds"
    if s in {"inform"}: return "informs"
    return s or "informs"


def _polarity_to_edge_type(polarity: str) -> str:
    return "refutes" if (polarity or "").lower().startswith("neg") else "supports"


# ----------------------------- core: nodes -----------------------------------

def canonicalize_and_cluster_nodes(s1_nodes: List[Dict[str, Any]], debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    """1) канонизация типов (+ мягкая перетипизация по label);
       2) дедуп по fingerprint;
       3) кластеризация 'почти-дублей' внутри типа.
    """
    dropped = []
    retyped = 0

    # 1) канонизация и перетипизация
    canon_nodes = []
    for n in s1_nodes or []:
        t_raw = n.get("type") or ""
        t = _canon_type(t_raw)
        lbl = n.get("label") or ""
        t_guess = _guess_type_from_label(lbl)
        if t_guess and t_guess != t:
            t = t_guess
            n["retyped_by_label"] = True
            retyped += 1
        n["type"] = t
        canon_nodes.append(n)

    # 2) дедуп по fingerprint (type + norm_text)
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for n in canon_nodes:
        txt_norm = _norm_txt_for_fp(n.get("text", ""))
        key = (n["type"], txt_norm)
        cur = groups.get(key)
        if not cur:
            groups[key] = {**n, "prov_multi": [n.get("prov", {})]}
        else:
            # оставляем более уверенный/длинный
            if float(n.get("conf", 0.0)) > float(cur.get("conf", 0.0)):
                cur["text"] = n.get("text", cur.get("text"))
                cur["conf"] = n.get("conf")
                cur["label"] = n.get("label", cur.get("label"))
                cur["polarity"] = n.get("polarity", cur.get("polarity"))
                cur["prov"] = n.get("prov", cur.get("prov"))
            cur.setdefault("prov_multi", []).append(n.get("prov", {}))

    dedup_nodes = list(groups.values())

    # 3) кластеризация «почти-дублей» внутри типа (Jaccard по токенам)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for n in dedup_nodes:
        by_type.setdefault(n["type"], []).append(n)

    clustered: List[Dict[str, Any]] = []
    for t, arr in by_type.items():
        used = set()
        arr_sorted = sorted(arr, key=lambda x: (-len(x.get("text", "")), -float(x.get("conf", 0.0))))
        tokens_list = [set(_tokenize_content(x.get("text", ""))) for x in arr_sorted]
        for i, ni in enumerate(arr_sorted):
            if i in used: continue
            cluster = [ni]
            used.add(i)
            tok_i = tokens_list[i]
            for j in range(i + 1, len(arr_sorted)):
                if j in used: continue
                tok_j = tokens_list[j]
                sim = _jaccard(tok_i, tok_j)
                # строгий порог, чтобы не склеивать разные факты
                if sim >= 0.90:
                    cluster.append(arr_sorted[j]);
                    used.add(j)
            if len(cluster) == 1:
                clustered.append(ni)
            else:
                # берём самый уверенный, провены объединяем
                best = max(cluster, key=lambda x: float(x.get("conf", 0.0)))
                merged = {**best}
                pm = []
                for c in cluster:
                    pm.extend(c.get("prov_multi") or [c.get("prov", {})])
                merged["prov_multi"] = pm
                clustered.append(merged)

    debug["nodes_retyped_by_label"] = retyped
    debug["nodes_after_dedup"] = len(dedup_nodes)
    debug["nodes_after_cluster"] = len(clustered)
    return clustered


# ----------------------------- core: edges -----------------------------------

@dataclass
class Edge:
    u: str
    v: str
    etype: str
    conf: float
    prov: Dict[str, Any]


def _edge_features(u: Dict[str, Any], v: Dict[str, Any]) -> Dict[str, Any]:
    sec_u, sec_v = _prov_section(u), _prov_section(v)
    same_section = int(bool(sec_u) and (sec_u == sec_v))
    d = abs(_first_span_start(u) - _first_span_start(v))
    # лексика
    toks_u = set(_tokenize_content(u.get("text", "")))
    toks_v = set(_tokenize_content(v.get("text", "")))
    jlex = _jaccard(toks_u, toks_v)
    return {"same_section": same_section, "char_dist": d, "jlex": jlex}


def _proximity_bonus(feat: Dict[str, Any]) -> float:
    bonus = 0.0
    if feat["same_section"]:
        bonus += 0.06
    d = feat["char_dist"]
    if d <= 400:
        bonus += 0.04
    elif d <= 1200:
        bonus += 0.02
    elif d > 2000:
        bonus -= 0.04
    # лексическое пересечение
    j = feat["jlex"]
    if j >= 0.25:
        bonus += min(0.05, j * 0.10)  # до +0.05
    return bonus


def _edge_allowed(t_from: str, t_to: str) -> bool:
    return t_to in ALLOWED_EDGE_PAIRS.get(t_from, set())


def _best_edge_for_pair(existing: Optional[Edge], candidate: Edge) -> Edge:
    if not existing:
        return candidate
    # сравниваем по приоритету типа, затем по conf
    pe = EDGE_PRIORITY.get(existing.etype, 0)
    pc = EDGE_PRIORITY.get(candidate.etype, 0)
    if pc > pe:
        return candidate
    if pc == pe and candidate.conf > existing.conf:
        return candidate
    return existing


def _index_nodes_by_id(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in nodes}


def relink_and_rescore_edges(
        s1_edges: List[Dict[str, Any]],
        nodes: List[Dict[str, Any]],
        drop_log: Dict[str, int],
        remap_log: Dict[str, int],
        edge_features_samples: List[Dict[str, Any]],
) -> List[Edge]:
    by_id = _index_nodes_by_id(nodes)
    best_per_pair: Dict[Tuple[str, str], Edge] = {}
    kept = 0

    for e in s1_edges or []:
        u_id = e.get("from");
        v_id = e.get("to")
        if u_id not in by_id or v_id not in by_id:
            drop_log["missing_node"] = drop_log.get("missing_node", 0) + 1
            continue

        u, v = by_id[u_id], by_id[v_id]
        t_from = u.get("type");
        t_to = v.get("type")

        etype = _edge_type_canon(e.get("type"))
        # если тип не разрешён, попытаемся ремапнуть (на supports/refutes Result→Hypothesis)
        if not _edge_allowed(t_from, t_to):
            # special: если (Result -> Hypothesis) и у результата есть polarity — можно мапнуть
            if t_from == "Result" and t_to == "Hypothesis":
                etype_new = _polarity_to_edge_type(u.get("polarity"))
                remap_log["pair_remap"] = remap_log.get("pair_remap", 0) + 1
                etype = etype_new
            else:
                drop_log["pair_not_allowed"] = drop_log.get("pair_not_allowed", 0) + 1
                continue

        # рескоринг
        feat = _edge_features(u, v)
        bonus = _proximity_bonus(feat)
        base_conf = float(e.get("conf", 0.40))
        new_conf = max(0.0, min(1.0, base_conf + bonus))

        edge = Edge(u=u_id, v=v_id, etype=etype, conf=new_conf, prov=e.get("prov", {}))
        key = (u_id, v_id)
        best_per_pair[key] = _best_edge_for_pair(best_per_pair.get(key), edge)
        kept += 1

        # собираем небольшую выборку фич для дебага
        if len(edge_features_samples) < 40:
            edge_features_samples.append({
                "from": u_id, "to": v_id, "type": etype, "base_conf": base_conf, "conf": new_conf, **feat
            })

    return list(best_per_pair.values())


# ----------------------------- fallback backbone ------------------------------

def _nearest_by_type(src: Dict[str, Any], pool: List[Dict[str, Any]], want_type: str, k: int = 1) -> List[
    Dict[str, Any]]:
    cand = [n for n in pool if n.get("type") == want_type and n["id"] != src["id"]]
    # сортируем по расстоянию символов и по тому, совпадает ли секция
    cand.sort(key=lambda n: (abs(_first_span_start(n) - _first_span_start(src)),
                             0 if _prov_section(n) == _prov_section(src) else 1))
    return cand[:k]


def ensure_minimal_backbone(
        nodes: List[Dict[str, Any]],
        edges: List[Edge],
        drop_log: Dict[str, int],
        added_log: Dict[str, int],
) -> List[Edge]:
    have_edges = {(e.u, e.v) for e in edges}
    by_id = _index_nodes_by_id(nodes)

    # быстрые индексы по типу
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        by_type.setdefault(n["type"], []).append(n)

    def add(u: Dict[str, Any], v: Dict[str, Any], etype: str, base: float = 0.45):
        if (u["id"], v["id"]) in have_edges:
            return
        feat = _edge_features(u, v)
        conf = max(0.0, min(1.0, base + _proximity_bonus(feat)))
        edges.append(Edge(u=u["id"], v=v["id"], etype=etype, conf=conf, prov={"hint": "fallback"}))
        have_edges.add((u["id"], v["id"]))
        added_log[etype] = added_log.get(etype, 0) + 1

    # 1) Result -> Hypothesis (supports/refutes by polarity), максимум 2
    for r in by_type.get("Result", [])[:8]:
        targets = _nearest_by_type(r, by_type.get("Hypothesis", []), "Hypothesis", k=2)
        etype = _polarity_to_edge_type(r.get("polarity"))
        for h in targets:
            add(r, h, etype)

    # 2) Technique -> (Experiment | Result), максимум 2 на технику
    for t in by_type.get("Technique", [])[:8]:
        # предпочитаем Experiment, иначе Result
        targets = _nearest_by_type(t, nodes, "Experiment", k=2) or _nearest_by_type(t, nodes, "Result", k=2)
        for z in targets:
            add(t, z, "uses", base=0.42)

    # 3) Dataset -> Experiment, по одному ближайшему
    for d in by_type.get("Dataset", [])[:10]:
        tgt = _nearest_by_type(d, nodes, "Experiment", k=1)
        if tgt:
            add(d, tgt[0], "feeds", base=0.42)

    # 4) Experiment -> Result, по одному ближайшему
    for ex in by_type.get("Experiment", [])[:10]:
        tgt = _nearest_by_type(ex, nodes, "Result", k=1)
        if tgt:
            add(ex, tgt[0], "produces", base=0.44)

    return edges


# ----------------------------- layout (preset) --------------------------------

def assign_column_positions(nodes: List[Dict[str, Any]], col_gap: int = 140, row_gap: int = 120) -> None:
    """
    Присваивает каждой ноде (col,row) и абсолютные позиции x,y для preset-лейаута.
    Стабильная сортировка по порядку появления в тексте (span.start), затем по секции.
    """
    buckets: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(CANON_TYPES))}
    for n in nodes:
        t = n.get("type", "Analysis")
        col = TYPE_ORDER.get(t, TYPE_ORDER["Analysis"])
        n.setdefault("data", {})
        n["data"]["col"] = col
        buckets[col].append(n)

    # сортировка внутри колонок по позиции в тексте/секции
    for col, arr in buckets.items():
        arr.sort(key=lambda n: (_first_span_start(n), _prov_section(n)))
        for r, n in enumerate(arr):
            n["data"]["row"] = r

    # абсолютные координаты
    for n in nodes:
        col = n["data"]["col"]
        row = n["data"]["row"]
        n["position"] = {"x": 140 + col * col_gap, "y": 120 + row * row_gap}


# ----------------------------- quality score (lite) ---------------------------

def compute_q_score(nodes: List[Dict[str, Any]], edges: List[Edge]) -> Dict[str, Any]:
    # SC: structural completeness
    have = set(n["type"] for n in nodes)
    need_sets = [
        {"Hypothesis"}, {"Experiment", "Technique"}, {"Result"}, {"Conclusion", "Analysis", "Input Fact"}
    ]
    ok = sum(1 for s in need_sets if have & s)
    SC = ok / len(need_sets)

    # QF: quantified facts density (примитивно)
    txt = " ".join(n.get("text", "") for n in nodes if n["type"] in {"Result", "Analysis"})
    qf = 0.0
    qf += len(re.findall(r"\b\d{1,3}\s*%(\b|[^a-z])", txt))
    qf += len(re.findall(r"\bp\s*[<≤]\s*0\.\d+", txt))
    QF = min(1.0, qf / 8.0)

    # GC: graph connectivity proxy = E/N clipped
    N = max(1, len(nodes))
    GC = min(1.0, len(edges) / N)

    # PC: polarity consistency — штраф за странные supports при negative и refutes при positive (грубая эвристика)
    bad = 0
    pos = 0
    id2 = {n["id"]: n for n in nodes}
    for e in edges:
        if e.etype in {"supports", "refutes"} and e.u in id2 and e.v in id2:
            src = id2[e.u]
            pol = (src.get("polarity") or "").lower()
            if e.etype == "supports" and pol.startswith("neg"):
                bad += 1
            if e.etype == "refutes" and pol.startswith("pos"):
                bad += 1
            pos += 1
    PC = 1.0 if pos == 0 else max(0.0, 1.0 - bad / pos)

    Q = 0.35 * SC + 0.25 * min(1.0, SC + GC) + 0.15 * QF + 0.15 * GC + 0.10 * PC
    return {"Q": round(Q, 3), "SC": round(SC, 3), "QF": round(QF, 3), "GC": round(GC, 3), "PC": round(PC, 3)}


# ----------------------------- runner -----------------------------------------

def run_s2(s1_path: str, out_path: str) -> Dict[str, Any]:
    """
    s1_path: путь к s1_graph.json или к директории, где лежит s1_graph.json
    out_path: путь, куда писать graph.json (родительская папка используется и для s2_debug.json)
    """
    s1p = Path(s1_path)
    if s1p.is_dir():
        s1_file = s1p / "s1_graph.json"
    else:
        s1_file = s1p
    data = json.loads(s1_file.read_text(encoding="utf-8"))
    doc_id = data.get("doc_id") or s1_file.parent.name
    s1_nodes = data.get("nodes") or []
    s1_edges = data.get("edges") or []

    debug: Dict[str, Any] = {
        "doc_id": doc_id,
        "summary": {},
        "drops": {"nodes": [], "edges": []},
        "edge_features_samples": [],
        "hints": {"allowed_edge_pairs": ALLOWED_EDGE_PAIRS},
        "counters": {"edge_remap": {}, "fallback_added": {}},
    }

    # 1) узлы: канон+кластеризация
    nodes = canonicalize_and_cluster_nodes(s1_nodes, debug)

    # 2) рёбра: валидировать, рескорить, ремапить при необходимости
    drop_log: Dict[str, int] = {}
    remap_log: Dict[str, int] = {}
    edges = relink_and_rescore_edges(
        s1_edges, nodes,
        drop_log=drop_log,
        remap_log=remap_log,
        edge_features_samples=debug["edge_features_samples"]
    )

    # 3) ограничить кратность ребра между парой узлов одним «лучшим»
    best_for_pair: Dict[Tuple[str, str], Edge] = {}
    for e in edges:
        key = (e.u, e.v)
        best_for_pair[key] = _best_edge_for_pair(best_for_pair.get(key), e)
    edges = list(best_for_pair.values())

    # 4) добавить минимальный «скелет» при необходимости
    edges = ensure_minimal_backbone(nodes, edges, drop_log, debug["counters"]["fallback_added"])

    # 5) позиции (preset layout)
    assign_column_positions(nodes)

    # 6) сводки/отладка
    debug["summary"]["nodes_in"] = len(s1_nodes)
    debug["summary"]["nodes_out"] = len(nodes)
    debug["summary"]["edges_in"] = len(s1_edges)
    debug["summary"]["edges_out"] = len(edges)
    debug["summary"]["drop_reasons_edges"] = drop_log
    debug["summary"]["edge_remap"] = remap_log
    debug["summary"]["types_present"] = sorted(list({n["type"] for n in nodes}))
    debug["summary"]["types_missing"] = [t for t in CANON_TYPES if t not in {n["type"] for n in nodes}]

    # 7) Q-score
    q = compute_q_score(nodes, edges)
    debug["quality"] = q

    # 8) финальные объекты
    graph = {
        "doc_id": doc_id,
        "nodes": nodes,
        "edges": [
            {
                "from": e.u, "to": e.v, "type": e.etype, "conf": round(float(e.conf), 3),
                "prov": e.prov
            } for e in edges
        ]
    }

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (outp.parent / "s2_debug.json").write_text(json.dumps(_jsonify(debug), ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    return graph


def _jsonify(obj):
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    return obj


# ----------------------------- cli --------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True, help="Path to s1_graph.json or its folder")
    ap.add_argument("--out", required=True, help="Path to output graph.json")
    args = ap.parse_args()
    run_s2(args.s1, args.out)

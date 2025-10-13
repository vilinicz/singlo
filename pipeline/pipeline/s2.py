# pipeline/pipeline/s2.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# 8 канонических типов и порядок колонок
# ──────────────────────────────────────────────────────────────────────────────
TYPE_ORDER = [
    "Input Fact", "Hypothesis", "Experiment", "Technique",
    "Result", "Dataset", "Analysis", "Conclusion",
]
TYPE_INDEX = {t: i for i, t in enumerate(TYPE_ORDER)}

# Синонимы и «наследие» старых правил → канон
TYPE_SYNONYMS = {
    "Fact": "Input Fact",
    "Input": "Input Fact",
    "Input fact": "Input Fact",
    "Hyp": "Hypothesis",
    "Hypothesis": "Hypothesis",
    "Method": "Technique",
    "Technique": "Technique",
    "Experiment": "Experiment",
    "Observation": "Result",
    "Finding": "Result",
    "Evidence": "Result",
    "Result": "Result",
    "Data": "Dataset",
    "Dataset": "Dataset",
    "Analysis": "Analysis",
    "Conclusion": "Conclusion",
}

# Разрешённые типы связей между парами типов
ALLOWED_EDGE_TYPES: Dict[Tuple[str, str], set] = {
    ("Input Fact", "Hypothesis"): {"supports", "refutes", "informs"},
    ("Hypothesis", "Experiment"): {"tests"},
    ("Technique", "Experiment"): {"uses"},
    ("Dataset", "Experiment"): {"feeds", "used_by"},
    ("Experiment", "Result"): {"produces"},
    ("Result", "Hypothesis"): {"supports", "refutes"},
    ("Result", "Analysis"): {"informs"},
    ("Dataset", "Analysis"): {"informs"},
    ("Technique", "Analysis"): {"uses"},
    ("Analysis", "Conclusion"): {"supports", "refutes", "summarizes"},
    ("Result", "Conclusion"): {"supports", "refutes"},
}


# ──────────────────────────────────────────────────────────────────────────────
# Нормализация/дедуп/линковка/позиционирование
# ──────────────────────────────────────────────────────────────────────────────
def normalize_type(t: str) -> str:
    """
    Приводим произвольный тип к каноническому из 8.
    Терпим опечатки, двоеточия, суффиксы вроде 'HypothesisStatement'.
    """
    if not t:
        return "Analysis"
    t0 = str(t).strip()

    # точное попадание в канон
    if t0 in TYPE_INDEX:
        return t0

    # быстрые нормировки
    tl = t0.lower().strip(" :._-")
    tl = tl.replace("–", "-").replace("—", "-")

    # подстрочные эвристики (часто встречающиеся варианты/опечатки)
    if "hypo" in tl:                      # hypothesis, hypotesis, hypothesisstatement…
        return "Hypothesis"
    if "experi" in tl:                    # experiment, experimental
        return "Experiment"
    if "techni" in tl or "method" in tl:  # technique, methodology, method
        return "Technique"
    if "result" in tl or "observ" in tl or "finding" in tl or "eviden" in tl:
        return "Result"
    if "dataset" in tl or "data set" in tl or "corpus" in tl or "benchmark" in tl or "cohort" in tl or tl == "data":
        return "Dataset"
    if "conclu" in tl:
        return "Conclusion"
    if "analys" in tl:
        return "Analysis"
    if "fact" in tl or "assump" in tl or ("input" in tl and "fact" in tl):
        return "Input Fact"

    # попытки через регистры/синонимы (на случай аккуратных вариантов)
    cand = (
        TYPE_SYNONYMS.get(t0)
        or TYPE_SYNONYMS.get(t0.title())
        or TYPE_SYNONYMS.get(t0.lower().capitalize())
    )
    return cand or "Analysis"


def _txt(n: Dict[str, Any]) -> str:
    return (n.get("text") or "").strip()


# ── dedup_nodes: сохраняем исходный тип в type_raw для дебага ────────────────
def dedup_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Дедуп по (type, text). Сохраняем первую ноду как «главную»,
    остальные аккумулируем в prov_multi, conf = max().
    Пишем исходный тип в 'type_raw' (для отладки нормализации).
    """
    out: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], int] = {}
    for n in nodes:
        m = dict(n)
        m["type_raw"] = m.get("type")     # ← сохраняем оригинал
        m["type"] = normalize_type(m.get("type"))
        key = (m["type"], (m.get("text") or "").strip())
        if key in seen:
            i = seen[key]
            out[i]["conf"] = float(max(out[i].get("conf", 0.0), float(m.get("conf", 0.0))))
            if m.get("prov"):
                out[i].setdefault("prov_multi", []).append(m["prov"])
            # если у исходной «главной» не было type_raw — заполним
            if not out[i].get("type_raw"):
                out[i]["type_raw"] = n.get("type")
        else:
            out.append(m)
            seen[key] = len(out) - 1
    return out


def _node_by_id(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in nodes if n.get("id")}


def _map_polarity_to_relation(frm: Dict[str, Any], to: Dict[str, Any]) -> str:
    """
    Если ребро допустимо по паре типов, но тип ребра «левый»/пустой — мапим по полярности.
    """
    pol = f'{frm.get("polarity", "")}|{to.get("polarity", "")}'.lower()
    return "refutes" if "negative" in pol or "not significant" in pol else "supports"


def relink_edges(nodes: List[Dict[str, Any]],
                 edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Чистим рёбра: удаляем невалидные ID, приводим типы концов к канону,
    пропускаем только разрешённые пары/типы. Для допустимых пар без подходящего
    типа ребра — мягко мапим в supports/refutes по полярности.
    """
    id2n = _node_by_id(nodes)
    out: List[Dict[str, Any]] = []
    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a not in id2n or b not in id2n:
            continue
        frm = id2n[a]
        to = id2n[b]
        frm_t = normalize_type(frm.get("type"))
        to_t = normalize_type(to.get("type"))
        pair = (frm_t, to_t)
        etype = (e.get("type") or "").strip() or "supports"

        if pair in ALLOWED_EDGE_TYPES:
            if etype in ALLOWED_EDGE_TYPES[pair]:
                out.append(e)
            else:
                mapped = _map_polarity_to_relation(frm, to)
                if mapped in ALLOWED_EDGE_TYPES[pair]:
                    e2 = dict(e)
                    e2["type"] = mapped
                    out.append(e2)
        # Иначе — просто отбрасываем
    return out


def ensure_minimal_backbone(nodes: List[Dict[str, Any]],
                            edges: List[Dict[str, Any]],
                            max_per_link: int = 2) -> List[Dict[str, Any]]:
    """
    Если после фильтрации рёбер мало или их ноль — строим минимальный каркас,
    чтобы пользователь видел связность по колонкам.
    """
    ids_by_type: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        ids_by_type[n["type"]].append(n["id"])

    def add(frm_id: str, to_id: str, etype: str, conf: float = 0.56, hint: str = "fallback") -> None:
        edges.append({
            "from": frm_id,
            "to": to_id,
            "type": etype,
            "conf": round(conf, 3),
            "prov": {"hint": hint}
        })

    # Result → Hypothesis
    for r in ids_by_type["Result"][:max_per_link]:
        for h in ids_by_type["Hypothesis"][:max_per_link]:
            add(r, h, "supports")

    # Technique/Dataset → Experiment
    for t in ids_by_type["Technique"][:max_per_link]:
        for ex in ids_by_type["Experiment"][:max_per_link]:
            add(t, ex, "uses")
    for d in ids_by_type["Dataset"][:max_per_link]:
        for ex in ids_by_type["Experiment"][:max_per_link]:
            add(d, ex, "feeds")

    # Result/Dataset/Technique → Analysis → Conclusion
    for a in ids_by_type["Analysis"][:max_per_link]:
        for c in ids_by_type["Conclusion"][:max_per_link]:
            add(a, c, "summarizes")

    # Input Fact → Hypothesis
    for f in ids_by_type["Input Fact"][:max_per_link]:
        for h in ids_by_type["Hypothesis"][:max_per_link]:
            add(f, h, "informs")

    return edges


def assign_column_positions(nodes: List[Dict[str, Any]],
                            x_step: int = 320, y_step: int = 120,
                            x0: int = 60, y0: int = 40) -> List[Dict[str, Any]]:
    """
    Раскладываем узлы по 8 колонкам (preset layout).
    Сортировка внутри колонки — по убыванию conf.
    """
    rows = defaultdict(int)
    # Стабильная сортировка: сперва по колонке, затем по conf убыв.
    nodes_sorted = sorted(
        nodes,
        key=lambda n: (TYPE_INDEX.get(n["type"], 999), -(n.get("conf") or 0.0))
    )
    for n in nodes_sorted:
        col = TYPE_INDEX.get(n["type"], len(TYPE_ORDER) - 1)
        row = rows[col]
        rows[col] += 1
        n["data"] = {**(n.get("data") or {}), "col": col, "row": row}
        n["position"] = {"x": x0 + col * x_step, "y": y0 + row * y_step}
    return nodes_sorted


# ──────────────────────────────────────────────────────────────────────────────
# Основной раннер S2
# ──────────────────────────────────────────────────────────────────────────────
def run_s2(export_dir: str) -> Dict[str, Any]:
    """
    Читает S1-артефакты (s1_graph.json, s1_debug.json), выполняет:
      - канонизацию типов, дедуп,
      - очистку/ремап рёбер к допустимым отношениям,
      - добавление минимального каркаса при пустоте,
      - назначение позиций для 8 колонок,
    и пишет graph.json + s2_debug.json.
    """
    exp = Path(export_dir)
    s1_graph_p = exp / "s1_graph.json"
    s1_debug_p = exp / "s1_debug.json"
    out_graph_p = exp / "graph.json"
    out_debug_p = exp / "s2_debug.json"

    # Базовые структуры
    base = {"doc_id": None, "nodes": [], "edges": []}
    if s1_graph_p.exists():
        try:
            base = json.loads(s1_graph_p.read_text(encoding="utf-8"))
        except Exception:
            pass

    doc_id = base.get("doc_id")
    s1_nodes: List[Dict[str, Any]] = base.get("nodes", []) or []
    s1_edges: List[Dict[str, Any]] = base.get("edges", []) or []

    # Забираем пороги из S1 debug (если есть — пригодится для отчёта)
    node_thr = 0.40
    edge_thr = 0.55
    section_names = []
    if s1_debug_p.exists():
        try:
            s1dbg = json.loads(s1_debug_p.read_text(encoding="utf-8"))
            sum_ = (s1dbg or {}).get("summary", {}) or {}
            node_thr = float(sum_.get("node_threshold", node_thr))
            edge_thr = float(sum_.get("edge_threshold", edge_thr))
            section_names = (s1dbg or {}).get("summary", {}).get("section_names", []) or []
        except Exception:
            pass

    # 1) Канонизация + дедуп
    nodes = dedup_nodes(s1_nodes)

    # 2) Предочистка рёбер по ID + канонические пары/типы
    # (Сначала оставим только рёбра, указывающие на существующие ID)
    valid_ids = {n.get("id") for n in nodes if n.get("id")}
    edges_pre = [e for e in s1_edges if e.get("from") in valid_ids and e.get("to") in valid_ids]

    # Нормализуем типы нод (на случай, если s1 выдал старые) — уже сделано в dedup_nodes,
    # но изменим inplace, чтобы гарантировать канон.
    for n in nodes:
        n["type"] = normalize_type(n.get("type"))

    # Фильтрация/ремап рёбер к ALLOWED_EDGE_TYPES
    edges = relink_edges(nodes, edges_pre)

    # 3) Минимальный каркас, если рёбер нет
    if not edges:
        edges = ensure_minimal_backbone(nodes, edges, max_per_link=2)

    # 4) Позиции (8 колонок), preset layout
    nodes = assign_column_positions(nodes)

    # 5) Итоговый граф
    final_graph = {"doc_id": doc_id, "nodes": nodes, "edges": edges}
    out_graph_p.write_text(json.dumps(final_graph, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6) Отладочный отчёт S2
    # распределение по типам и колонкам
    by_type = defaultdict(int)
    for n in nodes:
        by_type[n["type"]] += 1

    missing_types = [t for t in TYPE_ORDER if by_type.get(t, 0) == 0]

    raw_counts = defaultdict(int)
    for n in nodes:
        raw_counts[str(n.get("type_raw") or "")] += 1

    debug = {
        "doc_id": doc_id,
        "summary": {
            "nodes_in": len(s1_nodes),
            "edges_in": len(s1_edges),
            "nodes_out": len(nodes),
            "edges_out": len(edges),
            "node_threshold_from_s1": node_thr,
            "edge_threshold_from_s1": edge_thr,
            "types_present": {k: int(v) for k, v in sorted(by_type.items(), key=lambda kv: TYPE_INDEX[kv[0]])},
            "types_missing": missing_types,
            "reason": "ok" if nodes else "empty_after_s1",
            "types_raw_seen": {k: int(v) for k, v in raw_counts.items() if k},
        },
        "hints": {
            "section_names": section_names,
            "layout": {
                "mode": "preset",
                "columns": TYPE_ORDER,
                "note": "Nodes carry 'position' and 'data.col/row' to render as 8 fixed columns."
            },
            "allowed_edge_pairs": sorted(list({f"{a}→{b}" for (a, b) in ALLOWED_EDGE_TYPES.keys()})),
        }
    }
    out_debug_p.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    return debug

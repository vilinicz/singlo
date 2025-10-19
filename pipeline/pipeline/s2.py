# pipeline/pipeline/s2.py
from __future__ import annotations
import json, os, time, pathlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from .llm_layer import refine_graph, _log as llm_log

# ──────────────────────────────────────────────────────────────────────────────
# 8 канонических типов и порядок колонок
# ──────────────────────────────────────────────────────────────────────────────
TYPE_ORDER = [
    "Input Fact", "Hypothesis", "Experiment", "Technique",
    "Result", "Dataset", "Analysis", "Conclusion",
]
TYPE_INDEX = {t: i for i, t in enumerate(TYPE_ORDER)}

# Какое отношение брать по умолчанию для допустимой пары типов
DEFAULT_EDGE_FOR_PAIR = {
    ("Input Fact", "Hypothesis"): "informs",
    ("Hypothesis", "Experiment"): "tests",
    ("Technique", "Experiment"): "uses",
    ("Dataset", "Experiment"): "feeds",
    ("Experiment", "Result"): "produces",
    ("Result", "Hypothesis"): "supports",  # полярность может переопределить на refutes
    ("Result", "Analysis"): "informs",
    ("Dataset", "Analysis"): "informs",
    ("Technique", "Analysis"): "uses",
    ("Analysis", "Conclusion"): "summarizes",
    ("Result", "Conclusion"): "supports",
}


def _default_edge_type(pair: tuple, frm: dict, to: dict) -> str:
    # Для R→H учитываем полярность
    if pair == ("Result", "Hypothesis"):
        pol = f'{frm.get("polarity", "")}|{to.get("polarity", "")}'.lower()
        return "refutes" if "negative" in pol or "not significant" in pol else "supports"
    return DEFAULT_EDGE_FOR_PAIR.get(pair, "supports")


# ---- утилиты чтения/записи ----
def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any):
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _collect_sections(s0: Dict[str, Any]) -> List[str]:
    names = []
    for sec in s0.get("sections", []):
        name = (sec.get("name") or "").strip()
        if name:
            names.append(name)
    # de-dup preserving order
    out = []
    seen = set()
    for n in names:
        k = n.lower()
        if k not in seen:
            out.append(n)
            seen.add(k)
    return out


# ---- нормализация типов (на всякий случай) ----
_TMAP = {
    "InputFact": "InputFact",
    "Fact": "InputFact",
    "Observation": "Experiment",
    "Method": "Technique",
    "Technique": "Technique",
    "Experiment": "Experiment",
    "Result": "Result",
    "Dataset": "Dataset",
    "Analysis": "Analysis",
    "Conclusion": "Conclusion",
    "Hypothesis": "Hypothesis",
}
_ALLOWED = set(["InputFact", "Hypothesis", "Experiment", "Technique", "Result", "Dataset", "Analysis", "Conclusion"])


def _map_type(t: str) -> str:
    t = (t or "").strip()
    return _TMAP.get(t, t if t in _ALLOWED else "Result")


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
    if "hypo" in tl:  # hypothesis, hypotesis, hypothesisstatement…
        return "Hypothesis"
    if "experi" in tl:  # experiment, experimental
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
    Дедуп по (type, merge_key|text). Сохраняем первую ноду как «главную»,
    остальные аккумулируем в prov_multi, conf = max().
    Пишем исходный тип в 'type_raw' (для отладки нормализации).
    """
    out: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], int] = {}
    for n in nodes:
        m = dict(n)
        m["type_raw"] = m.get("type")  # ← сохраняем оригинал
        m["type"] = normalize_type(m.get("type"))
        merge_key = None
        try:
            merge_key = ((m.get("norm") or {}).get("merge_key") or "").strip().lower()
        except Exception:
            merge_key = ""
        basis = merge_key if merge_key else (m.get("text") or "").strip()
        key = (m["type"], basis)
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


def relink_edges(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    id2n = _node_by_id(nodes)
    out: List[Dict[str, Any]] = []
    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a not in id2n or b not in id2n:
            continue
        frm, to = id2n[a], id2n[b]
        frm_t = normalize_type(frm.get("type"))
        to_t = normalize_type(to.get("type"))
        pair = (frm_t, to_t)

        if pair not in ALLOWED_EDGE_TYPES:
            continue

        etype = (e.get("type") or "").strip()
        if etype in ALLOWED_EDGE_TYPES[pair]:
            out.append(e)
        else:
            # аккуратно мапим в «правильный» тип, вместо того чтобы терять ребро
            mapped = _default_edge_type(pair, frm, to)
            if mapped in ALLOWED_EDGE_TYPES[pair]:
                e2 = dict(e)
                e2["type"] = mapped
                out.append(e2)
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

def _layout_columns(doc_id: str,
                    nodes: list[dict],
                    edges: list[dict],
                    *,
                    column_order: list[str] | None = None,
                    left_margin: int = 120,
                    top_margin: int = 120,
                    x_step: int = 260,
                    y_step: int = 140,
                    col_padding: int = 0) -> dict:
    """
    Колончатая раскладка для фронта.
    - column_order: фиксированный порядок типов (если None — используем канонический).
    - x_step / y_step: расстояние между столбцами и между блоками по вертикали.
    - left_margin / top_margin: отступы от края канваса.
    - col_padding: дополнительный отступ между столбцами (если нужен).

    Каждому узлу проставляет:
      node["data"]["col"] = индекс столбца
      node["data"]["row"] = порядковый номер в столбце
      node["position"] = {"x": X, "y": Y}
    """
    # Канонический порядок (8 колонок)
    default_order = [
        "Input Fact", "Hypothesis", "Experiment", "Technique",
        "Result", "Dataset", "Analysis", "Conclusion"
    ]
    order = list(column_order or default_order)

    # Нормализация типа (используем вашу normalize_type, если она есть)
    try:
        _norm_type = normalize_type  # noqa: F405
    except Exception:
        def _norm_type(t: str) -> str:
            return t

    # Карта индекс -> x
    x_positions: dict[int, int] = {}
    for ci, _ in enumerate(order):
        x_positions[ci] = left_margin + ci * (x_step + col_padding)

    # Разброс по колонкам
    buckets: dict[str, list[dict]] = {t: [] for t in order}
    other: list[dict] = []
    for n in nodes:
        t = _norm_type(n.get("type", ""))
        if t in buckets:
            buckets[t].append(n)
        else:
            other.append(n)  # тип вне канона — складываем отдельно

    # Сортировка внутри каждой колонки: conf ↓, затем id (стабильно)
    def _score(n: dict) -> tuple:
        c = float(n.get("conf", 0.0))
        nid = str(n.get("id", ""))
        return (c, nid)

    for t in order:
        buckets[t].sort(key=_score, reverse=True)

    # Выкладка (позиции)
    laid_nodes: list[dict] = []
    for ci, t in enumerate(order):
        col_nodes = buckets.get(t, [])
        for ri, n in enumerate(col_nodes):
            nx = x_positions[ci]
            ny = top_margin + ri * y_step
            # гарантируем структуры 'data' и 'position'
            n.setdefault("data", {})
            n["data"]["col"] = ci
            n["data"]["row"] = ri
            n["position"] = {"x": nx, "y": ny}
            # перепишем тип в каноническое имя столбца
            n["type"] = t
            laid_nodes.append(n)

    # Узлы «других» типов (если есть) — добавим колонкой после последних
    if other:
        ci = len(order)
        x_positions[ci] = left_margin + ci * (x_step + col_padding)
        order.append("Other")
        other.sort(key=_score, reverse=True)
        for ri, n in enumerate(other):
            nx = x_positions[ci]
            ny = top_margin + ri * y_step
            n.setdefault("data", {})
            n["data"]["col"] = ci
            n["data"]["row"] = ri
            n["position"] = {"x": nx, "y": ny}
            n["type"] = "Other"
            laid_nodes.append(n)

    # Формируем итоговый граф
    graph = {
        "doc_id": doc_id,
        "nodes": laid_nodes,
        "edges": edges,
        "meta": {
            "layout": {
                "mode": "columns",
                "columns": order,
                "left_margin": left_margin,
                "top_margin": top_margin,
                "x_step": x_step,
                "y_step": y_step,
                "col_padding": col_padding
            }
        }
    }
    return graph


# ──────────────────────────────────────────────────────────────────────────────
# Основной раннер S2
# ──────────────────────────────────────────────────────────────────────────────
def run_s2(export_dir: str):
    """
    Stage S2:
      1) load S1 graph
      2) normalize + deduplicate nodes
      3) relink/validate edges, ensure minimal backbone
      4) conditionally refine with LLM on a compact payload (AFTER dedup!)
      5) final columnar layout and write graph.json
    """
    import os
    from pathlib import Path

    # --- безопасные импорты локальных утилит ---
    try:
        sec_normalize = normalize_type  # noqa: F405 (exists in this module)
    except Exception:
        def sec_normalize(t: str) -> str:
            return t

    try:
        type_order = TYPE_ORDER  # noqa: F405
    except Exception:
        # Фиксированный порядок колонок (как вы просили)
        type_order = [
            "Input Fact", "Hypothesis", "Experiment", "Technique",
            "Result", "Dataset", "Analysis", "Conclusion"
        ]

    # Разрешённые типы рёбер (минимальный базовый набор)
    ALLOWED_EDGE_TYPES_LOCAL = {
        "uses", "produces", "supports", "refutes", "derives", "relates"
    }

    # --- пути ---
    p = Path(export_dir)
    s1_path = p / "s1_graph.json"
    s0_path = p.parent / "s0.json"
    out_path = p / "graph.json"

    # --- чтение входа ---
    s1 = _read_json(str(s1_path))  # noqa: F405
    s0 = _read_json(str(s0_path)) if s0_path.exists() else {}  # noqa: F405
    doc_id = s1.get("doc_id") or s0.get("doc_id") or p.name

    nodes = list(s1.get("nodes", []))
    edges = list(s1.get("edges", []))

    # --- 1) нормализация типа и дедуп узлов ---
    def _norm(n: dict) -> dict:
        n2 = dict(n)
        n2["type"] = sec_normalize(n.get("type", ""))
        return n2

    nodes = [_norm(n) for n in nodes]
    # ожидается, что dedup_nodes уже есть в модуле
    nodes = dedup_nodes(nodes)  # noqa: F405

    # --- 2) приведение рёбер к консистентному виду + фильтрация мусора ---
    def _clean_edges(nodes_list: list, edges_list: list) -> list:
        idset = {n.get("id") for n in nodes_list if n.get("id")}
        cleaned = []
        for e in edges_list:
            f, t = e.get("from"), e.get("to")
            if not f or not t or f not in idset or t not in idset:
                continue
            et = e.get("type") or ""
            et = et.lower().strip()
            if et not in ALLOWED_EDGE_TYPES_LOCAL:
                # минимальный ремап (оставим 'relates' по умолчанию)
                et = "relates"
            conf = float(e.get("conf", 0.0))
            cleaned.append({"from": f, "to": t, "type": et, "conf": conf, "prov": e.get("prov")})
        return cleaned

    edges = _clean_edges(nodes, edges)

    # --- 3) каркас рёбер, если пусто ---
    if not edges:
        edges = ensure_minimal_backbone(nodes, edges)  # noqa: F405

    # --- 4) оценка «нужен ли LLM» ---
    # Связность: доля узлов, у которых есть хотя бы 1 смежное ребро
    touched = 0
    id2deg = {}
    for e in edges:
        id2deg[e["from"]] = id2deg.get(e["from"], 0) + 1
        id2deg[e["to"]] = id2deg.get(e["to"], 0) + 1
    for n in nodes:
        if id2deg.get(n.get("id"), 0) > 0:
            touched += 1
    coherence = (touched / max(1, len(nodes))) if nodes else 0.0

    counts = {t: 0 for t in type_order}
    for n in nodes:
        counts[sec_normalize(n.get("type", ""))] = counts.get(sec_normalize(n.get("type", "")), 0) + 1

    force_llm = os.environ.get("FORCE_LLM", "").lower() in ("1", "true", "yes")
    need_llm = force_llm or (counts.get("Result", 0) < 1) or (counts.get("Hypothesis", 0) < 1) or (coherence < 0.6)

    # --- 5) подготовка компактного payload'a для LLM (встроено тут) ---
    # Собираем немного контекста из S0: abstract/conclusion + 2–4 captions
    s0_ctx_chunks = []
    for sec in s0.get("sections", []):
        nm = (sec.get("name") or "").lower()
        if nm.startswith("abstract") or "conclusion" in nm:
            s0_ctx_chunks.append((sec.get("text") or "")[:800])
    for cap in (s0.get("captions") or [])[:4]:
        s0_ctx_chunks.append((cap.get("text") or "")[:400])

    # секционные веса — чтобы предпочесть Results/Discussion/Captions
    sec_weight = {
        "Results": 1.0, "Discussion": 0.95,
        "FigureCaption": 0.95, "TableCaption": 0.95,
        "Abstract": 0.85, "Conclusion": 0.85,
        "Methods": 0.80, "Body": 0.60, "Unknown": 0.50
    }

    def _node_salience(n: dict) -> tuple:
        prov = n.get("prov")
        sec = None
        if isinstance(prov, dict):
            sec = prov.get("section")
        elif isinstance(prov, list) and prov:
            sec = prov[0].get("section")
        return (float(n.get("conf", 0.0)), float(sec_weight.get(sec or "Unknown", 0.5)))

    # top-K по типам
    topk_by_type = {
        "Result": 25, "Hypothesis": 15, "Experiment": 12, "Technique": 12,
        "Dataset": 10, "Analysis": 10, "Input Fact": 10, "Conclusion": 8
    }

    # лимит длины текста
    try:
        trunc = int(os.environ.get("LLM_S1_TEXT_TRUNC", "220"))
    except Exception:
        trunc = 220

    # сжатые узлы для LLM
    bucket = {t: [] for t in type_order}
    for n in nodes:
        t = sec_normalize(n.get("type", ""))
        bucket.setdefault(t, []).append(n)

    picked_nodes = []
    for t in type_order:
        arr = sorted(bucket.get(t, []), key=_node_salience, reverse=True)[: topk_by_type.get(t, 10)]
        for n in arr:
            txt = (n.get("text") or "").strip()
            if len(txt) > trunc:
                # обрежем «мягко»: по ближайшей точке, если есть в пределах +40 символов
                cut = txt[:trunc]
                tail = txt[trunc:trunc + 40]
                dot = tail.find(".")
                if 0 <= dot <= 40:
                    cut = txt[:trunc + dot + 1]
                txt = cut.rstrip() + ("…" if not txt.endswith(".") else "")
            prov = n.get("prov")
            if isinstance(prov, list) and prov:
                prov = prov[0]
            picked_nodes.append({
                "id": n.get("id"),
                "type": t,
                "text": txt,
                "polarity": n.get("polarity", "neutral"),
                "conf": float(n.get("conf", 0.0)),
                "prov": {"section": (prov or {}).get("section", "Unknown"),
                         "span": (prov or {}).get("span")}
            })

    ids_ok = {n["id"] for n in picked_nodes if n.get("id")}
    picked_edges = [
        {"from": e["from"], "to": e["to"], "type": e.get("type", ""), "conf": float(e.get("conf", 0.0))}
        for e in edges if e.get("from") in ids_ok and e.get("to") in ids_ok
    ]

    # --- 6) условный вызов LLM ---
    refined_nodes = None
    refined_edges = None
    if need_llm:
        # refine_graph ожидает полный граф; передаём компактные ноды/рёбра и контекст S0
        refined = refine_graph(  # noqa: F405
            doc_id,
            {"nodes": picked_nodes, "edges": picked_edges},
            s0_sections=s0_ctx_chunks
        ) or {}

        # --- 7) валидация/вливание ответа LLM ---
        r_nodes = refined.get("nodes") or []
        r_edges = refined.get("edges") or []

        # нормализуем типы и фильтруем по допустимым
        idmap = {}
        valid_nodes = []
        for rn in r_nodes:
            nid = rn.get("id")
            ntype = sec_normalize(rn.get("type", ""))
            if ntype not in type_order:
                continue
            if nid is None:
                continue
            # оставим компактные обязательные поля
            vn = {
                "id": nid,
                "type": ntype,
                "text": (rn.get("text") or "").strip(),
                "polarity": rn.get("polarity", "neutral"),
                "conf": float(rn.get("conf", 0.0)),
                "prov": rn.get("prov") or {}
            }
            idmap[nid] = nid
            valid_nodes.append(vn)

        valid_ids = {v["id"] for v in valid_nodes}
        valid_edges = []
        for re_ in r_edges:
            f, t = re_.get("from"), re_.get("to")
            if f in valid_ids and t in valid_ids:
                et = (re_.get("type") or "").lower().strip()
                if et not in ALLOWED_EDGE_TYPES_LOCAL:
                    et = "relates"
                valid_edges.append({"from": f, "to": t, "type": et, "conf": float(re_.get("conf", 0.0))})

        # если LLM вернул совсем пусто — остаёмся на прежней версии
        if valid_nodes and valid_edges:
            refined_nodes, refined_edges = valid_nodes, valid_edges

    # --- 8) итоговые узлы/рёбра ---
    if refined_nodes is not None:
        nodes = refined_nodes
    if refined_edges is not None:
        edges = refined_edges

    # финальная зачистка на всякий случай
    edges = _clean_edges(nodes, edges)
    if not edges:
        edges = ensure_minimal_backbone(nodes, edges)  # noqa: F405

    # --- 9) раскладка по колонкам и запись ---
    graph = _layout_columns(doc_id, nodes, edges)  # noqa: F405
    _write_json(str(out_path), graph)  # noqa: F405


def _prepare_llm_payload(doc_id: str, nodes: List[dict], edges: List[dict], s0_sections: List[str] | None = None,
                         topk_by_type: dict | None = None, text_trunc: int = 220) -> dict:
    """
    Сжимает граф для LLM:
      - нормализует типы,
      - берёт top-K узлов по типу,
      - обрезает текст, сокращает prov,
      - возвращает компактный JSON для refine_graph.
    """
    topk_by_type = topk_by_type or {
        "Result": 25, "Hypothesis": 15, "Experiment": 12, "Technique": 12,
        "Dataset": 10, "Analysis": 10, "Input Fact": 10, "Conclusion": 8
    }

    # нормализация типов и сортировка по conf, потом по «салентности секции»
    sec_weight = {"Results": 1.0, "Discussion": 0.95, "FigureCaption": 0.95, "TableCaption": 0.95,
                  "Abstract": 0.85, "Conclusion": 0.85, "Methods": 0.8, "Body": 0.6, "Unknown": 0.5}

    def _score(n):
        sec = ((n.get("prov") or {}).get("section") if isinstance(n.get("prov"), dict) else
               (n.get("prov_multi") or [{}])[0].get("section", "Unknown"))
        return (float(n.get("conf", 0.0)), float(sec_weight.get((sec or "Unknown"), 0.5)))

    buckets: dict[str, list[dict]] = {t: [] for t in TYPE_ORDER}
    for n in nodes:
        t = normalize_type(n.get("type"))
        n2 = dict(n)
        n2["type"] = t
        buckets.setdefault(t, []).append(n2)

    pick_nodes: list[dict] = []
    for t, arr in buckets.items():
        arr = sorted(arr, key=_score, reverse=True)[: topk_by_type.get(t, 10)]
        for n in arr:
            txt = (n.get("text") or "").strip()
            if len(txt) > text_trunc:
                txt = txt[:text_trunc].rstrip() + "…"
            prov = n.get("prov")
            if isinstance(prov, list) and prov:
                prov = prov[0]
            n_small = {
                "id": n.get("id"),
                "type": t,
                "text": txt,
                "polarity": n.get("polarity", "neutral"),
                "conf": float(n.get("conf", 0.0)),
                "prov": {"section": (prov or {}).get("section", "Unknown"),
                         "span": (prov or {}).get("span", None)}
            }
            pick_nodes.append(n_small)

    # Рёбра оставляем только те, что соединяют отобранные ноды
    ids_ok = {n["id"] for n in pick_nodes if n.get("id")}
    pick_edges = [
        {"from": e["from"], "to": e["to"], "type": e.get("type", ""), "conf": float(e.get("conf", 0.0))}
        for e in edges if e.get("from") in ids_ok and e.get("to") in ids_ok
    ]

    return {
        "doc_id": doc_id,
        "nodes": pick_nodes,
        "edges": pick_edges,
        "s0_context": (s0_sections or [])[:6]  # например: абстракт, концл, top captions
    }

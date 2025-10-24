"""Stage S2: graph normalization, deduplication, relinking, and layout.

Loads S1 outputs, normalizes node types and text, deduplicates near-duplicates,
relinks/filters edges, ensures a minimal backbone when sparse, optionally refines
with an LLM on a compact payload, and writes export/<doc_id>/graph.json.
"""
# pipeline/pipeline/s2.py
from __future__ import annotations
import json, os, pathlib, re
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from .llm_layer import refine_graph, _log as llm_log

# Stage S2 overview:
# - Load S1 graph, normalize/cleanup nodes
# - Deduplicate near-duplicate nodes; relink and validate edges
# - Ensure a minimal backbone if edges are sparse or missing
# - Optionally refine with an LLM on a compact payload
# - Produce final layout hints and write export/<doc_id>/graph.json

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
    ("Technique", "Result"): "uses",
    ("Analysis", "Conclusion"): "summarizes",
    ("Result", "Conclusion"): "supports",
}

SPACE = re.compile(r"\s+")
PUNCT = re.compile(r"[“”\"'`]+")


# Normalize text for comparison/deduplication (lowercase, strip, collapse).
def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = PUNCT.sub("", s)
    s = SPACE.sub(" ", s)
    return s


# Compute a deduplication key per node (type + canonical text; special-case Hypothesis).
def dedup_key(node):
    t = node["type"]
    imrad_raw = (node.get("prov", {}) or {}).get("imrad", "OTHER")
    key = (t, norm_text(node["text"]))
    if t == "Hypothesis":
        # разделяем гипотезы, если они из разных контекстов INTRO vs DISCUSSION
        imrad_norm = "INTRO" if "INTRO" in imrad_raw else ("DISCUSSION" if "DISCUSSION" in imrad_raw else "OTHER")
        key = (t, norm_text(node["text"]), imrad_norm)
    return key


# Infer default edge type for a node pair; use polarity for Result→Hypothesis.
def _default_edge_type(pair: tuple, frm: dict, to: dict) -> str:
    # Для R→H учитываем полярность
    if pair == ("Result", "Hypothesis"):
        pol = f'{frm.get("polarity", "")}|{to.get("polarity", "")}'.lower()
        return "refutes" if "negative" in pol or "not significant" in pol else "supports"
    return DEFAULT_EDGE_FOR_PAIR.get(pair, "supports")


# ---- утилиты чтения/записи ----
# Read a JSON file (UTF-8) and return object.
def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Write JSON (UTF-8, pretty) and ensure parent directory.
def _write_json(path: str, obj: Any):
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _mark_retyped(prov: dict | None) -> dict:
    """
    Помечает ребро как изменённое типологически: prov.hint += 'retyped'
    (не перетирая существующие подсказки).
    """
    p = dict(prov or {})
    hint = p.get("hint")
    if not hint:
        p["hint"] = "retyped"
    else:
        s = str(hint)
        if "retyped" not in s:
            p["hint"] = f"{s};retyped"
    return p


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
    ("Technique", "Result"): {"uses", "informs"},
    ("Analysis", "Conclusion"): {"supports", "refutes", "summarizes"},
    ("Result", "Conclusion"): {"supports", "refutes"},
    # --- расширение допустимых пар из S1
    ("Result", "Result"): {"follows", "informs"},
    ("Dataset", "Result"): {"summarizes", "informs"},
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
        return "Other"
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
    return cand or "Other"


def _txt(n: Dict[str, Any]) -> str:
    return (n.get("text") or "").strip()


# ── dedup_nodes: сохраняем исходный тип в type_raw для дебага ────────────────
## Merge near-duplicate nodes; preserve the most confident representative.
def dedup_nodes(nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Дедуп по ключу из dedup_key(): (type, norm_text(text), SPECIAL for Hypothesis/IMRAD).
    Сохраняем максимально уверенный узел; аккумулируем провенанс; полярность — neutral, если расходится.
    Возвращает (dedup_nodes, id_map), где id_map: old_id → canonical_id.
    """
    buckets: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for n in nodes:
        m = dict(n)
        m["type_raw"] = m.get("type")
        m["type"] = normalize_type(m.get("type"))
        k = dedup_key(m)
        buckets[k].append(m)

    out: List[Dict[str, Any]] = []
    id_map: Dict[str, str] = {}
    for _, arr in buckets.items():
        if len(arr) == 1:
            node = arr[0]
            out.append(node)
            node_id = node.get("id")
            if node_id:
                id_map[node_id] = node_id
            continue
        arr.sort(key=lambda x: float(x.get("conf", 0.0)), reverse=True)
        top = dict(arr[0])
        top_id = top.get("id")
        if top_id:
            id_map[top_id] = top_id
        provs = [a.get("prov") for a in arr if a.get("prov")]
        if provs:
            top["prov_multi"] = provs
        pols = {(a.get("polarity") or "neutral") for a in arr}
        if len(pols) > 1:
            top["polarity"] = "neutral"
        top["conf"] = float(max(float(a.get("conf", 0.0)) for a in arr))
        out.append(top)
        for dup in arr[1:]:
            dup_id = dup.get("id")
            if dup_id and top_id:
                id_map[dup_id] = top_id
    return out, id_map


## Build an id→node index for quick lookups.
def _node_by_id(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in nodes if n.get("id")}


## Choose supports/refutes for Result→Hypothesis based on polarity.
def _map_polarity_to_relation(frm: Dict[str, Any], to: Dict[str, Any]) -> str:
    """
    Если ребро допустимо по паре типов, но тип ребра «левый»/пустой — мапим по полярности.
    """
    pol = f'{frm.get("polarity", "")}|{to.get("polarity", "")}'.lower()
    return "refutes" if "negative" in pol or "not significant" in pol else "supports"


## Validate/rebuild edges using node types, proximity and polarity hints.
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
                # помечаем, что тип ребра был изменён на допустимый
                e2["prov"] = _mark_retyped(e2.get("prov"))
                out.append(e2)
    return out


## Ensure at least a simple backbone when no edges exist (best-effort hints).
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

    seen = {(e["from"], e["to"], e["type"]) for e in edges}

    def add(frm_id: str, to_id: str, etype: str, conf: float = 0.56, hint: str = "fallback") -> None:
        key = (frm_id, to_id, etype)
        if key in seen:
            return
        seen.add(key)
        edges.append({
            "from": frm_id,
            "to": to_id,
            "type": etype,
            "conf": round(conf, 3),
            "prov": {"hint": hint}
        })

    # Result → Hypothesis
    # for r in ids_by_type["Result"][:max_per_link]:
    #     for h in ids_by_type["Hypothesis"][:max_per_link]:
    #         add(r, h, "supports")
    # Подстроим тип по полярности результата
    id2n = {n["id"]: n for n in nodes}
    for r in ids_by_type["Result"][:max_per_link]:
        rnode = id2n.get(r, {})
        pol = (rnode.get("polarity") or "").lower()
        et = "refutes" if ("negative" in pol or "not significant" in pol) else "supports"
        for h in ids_by_type["Hypothesis"][:max_per_link]:
            add(r, h, et)

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


def _ensure_geo(n: dict, fallback: dict | None = None) -> dict:
    """Гарантируем, что у узла есть page/coords/bbox. Берём с узла, из prov, затем из fallback."""
    prov = n.get("prov") or {}
    page = n.get("page") or prov.get("page")
    coords = n.get("coords") or prov.get("coords")
    bbox = n.get("bbox") or prov.get("bbox")

    if fallback:
        page = page if (isinstance(page, int) and page >= 0) else fallback.get("page")
        coords = coords if isinstance(coords, list) else fallback.get("coords")
        bbox = bbox if isinstance(bbox, list) else fallback.get("bbox")

    # финально нормализуем типы
    try:
        n["page"] = int(page) if page is not None else 0
    except:
        n["page"] = 0
    n["coords"] = coords if isinstance(coords, list) else []
    n["bbox"] = bbox if isinstance(bbox, list) else []
    return n


## Assign simple column/row hints by type ordering to aid UI layout.
def _layout_columns(doc_id: str,
                    nodes: list[dict],
                    edges: list[dict],
                    *,
                    column_order: list[str] | None = None) -> dict:
    """
    Колончатая раскладка для фронта.
    Каждому узлу проставляет:
      node["data"]["col"] = индекс столбца
      node["data"]["row"] = порядковый номер в столбце
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

    # Разброс по колонкам
    buckets: dict[str, list[dict]] = {t: [] for t in order}
    other: list[dict] = []
    for n in nodes:
        t = _norm_type(n.get("type", ""))
        if t in buckets:
            buckets[t].append(n)
        else:
            other.append(n)  # тип вне канона — складываем отдельно

    # Сортировка внутри колонки: текстовый порядок (page, sent_idx, y, id)
    def _ord_key(n: dict) -> tuple:
        prov = (n.get("prov") or {})
        # page: node['page'] или prov['page']; если нет — в конец
        try:
            page = int(n.get("page")) if n.get("page") is not None else int(prov.get("page", 10 ** 9))
        except Exception:
            page = 10 ** 9
        # sent_idx: если нет/отрицательный — в конец своей страницы
        try:
            sent = int(prov.get("sent_idx", 10 ** 9))
            if sent < 0:
                sent = 10 ** 9
        except Exception:
            sent = 10 ** 9
        # y-координата верхнего края bbox (если есть) — стабилизирует порядок на странице
        bbox = n.get("bbox") or []
        try:
            y = float(bbox[1]) if len(bbox) >= 2 else 10 ** 6
        except Exception:
            y = 10 ** 6
        nid = str(n.get("id", ""))
        return (page, sent, y, nid)

    for t in order:
        buckets[t].sort(key=_ord_key)  # возрастающе: как в тексте

    # Выкладка (позиции)
    laid_nodes: list[dict] = []
    for ci, t in enumerate(order):
        col_nodes = buckets.get(t, [])
        for ri, n in enumerate(col_nodes):
            # гарантируем структуры 'data'
            n.setdefault("data", {})
            n["data"]["col"] = ci
            n["data"]["row"] = ri
            # перепишем тип в каноническое имя столбца
            n["type"] = t
            laid_nodes.append(n)

    # Узлы «других» типов (если есть) — добавим колонкой после последних
    if other:
        ci = len(order)
        order.append("Other")
        other.sort(key=_ord_key)
        for ri, n in enumerate(other):
            n.setdefault("data", {})
            n["data"]["col"] = ci
            n["data"]["row"] = ri
            n["type"] = "Other"
            laid_nodes.append(n)

    # Формируем итоговый граф
    graph = {
        "doc_id": doc_id,
        "nodes": laid_nodes,
        "edges": edges,
    }
    return graph


# ──────────────────────────────────────────────────────────────────────────────
# Основной раннер S2
# ──────────────────────────────────────────────────────────────────────────────
def run_s2(export_dir: str):
    # Stage S2: normalize, dedup, relink, optionally refine with LLM, layout, write graph.json
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

    # Используем объединение допустимых типов из всего пайплайна,
    # чтобы не терять семантику LLM-ответа.
    ALLOWED_EDGE_TYPES_LOCAL = {
        "uses", "produces", "supports", "refutes", "derives", "relates",
        "feeds", "informs", "summarizes", "used_by"
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

    # --- GEO SNAPSHOT из S1: чтобы всегда можно было восстановить
    orig_geo = {}
    for n in nodes:
        nid = n.get("id")
        if not nid:
            continue
        prov = n.get("prov") or {}
        orig_geo[nid] = {
            "page": n.get("page") or prov.get("page") or 0,
            "coords": (n.get("coords") if isinstance(n.get("coords"), list) else []) or (
                prov.get("coords") if isinstance(prov.get("coords"), list) else []) or [],
            "bbox": (n.get("bbox") if isinstance(n.get("bbox"), list) else []) or (
                prov.get("bbox") if isinstance(prov.get("bbox"), list) else []) or [],
        }

    # --- 1) нормализация типа и дедуп узлов ---
    def _norm(n: dict) -> dict:
        n2 = dict(n)
        n2["type"] = sec_normalize(n.get("type", ""))
        return n2

    nodes = [_norm(n) for n in nodes]
    # ожидается, что dedup_nodes уже есть в модуле
    nodes, id_map = dedup_nodes(nodes)  # noqa: F405

    # переназначаем id рёбер согласно каноническим узлам
    remapped_edges: List[Dict[str, Any]] = []
    for e in edges:
        frm = id_map.get(e.get("from"))
        to = id_map.get(e.get("to"))
        if not frm or not to:
            continue
        e2 = dict(e)
        e2["from"] = frm
        e2["to"] = to
        remapped_edges.append(e2)
    edges = remapped_edges

    # 2) релинковка и нормализация рёбер по типам узлов
    edges = relink_edges(nodes, edges)

    # 2.1) убрать дубликаты (from,to,type) — оставить максимальный conf
    best = {}
    for e in edges:
        key = (e["from"], e["to"], e["type"])
        if key not in best or float(e.get("conf", 0.0)) > best[key]["conf"]:
            best[key] = {"from": e["from"], "to": e["to"], "type": e["type"], "conf": float(e.get("conf", 0.0)),
                         "prov": e.get("prov")}
    edges = list(best.values())

    # 3) каркас, если пусто
    if not edges:
        edges = ensure_minimal_backbone(nodes, edges)

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

    # todo сомнительно
    force_llm = os.environ.get("FORCE_LLM", "").lower() in ("1", "true", "yes")
    need_llm = force_llm or (counts.get("Result", 0) < 1) or (counts.get("Hypothesis", 0) < 1) or (coherence < 0.6)
    if os.environ.get("DISABLE_LLM", "").lower() in {"1", "true", "yes"}:
        need_llm = False
    if need_llm and not os.environ.get("OPENAI_API_KEY"):
        need_llm = False

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
        sec_key = (sec or "Unknown")
        sk = sec_key.lower()
        # Мягкое сопоставление
        if "result" in sk and "discussion" in sk:
            sec_key = "Results"  # считать как Results
        elif "figure" in sk:
            sec_key = "FigureCaption"
        elif "table" in sk:
            sec_key = "TableCaption"
        return (float(n.get("conf", 0.0)), float(sec_weight.get(sec_key, 0.5)))

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
                et = (re_.get("type") or "").strip()
                et_low = et.lower()
                et = et if et_low in ALLOWED_EDGE_TYPES_LOCAL else "relates"
                valid_edges.append({
                    "from": f, "to": t, "type": et, "conf": float(re_.get("conf", 0.0))
                })
                et = (re_.get("type") or "").strip()
                et_low = et.lower()
                changed = et_low not in ALLOWED_EDGE_TYPES_LOCAL

                if changed:
                    et = "relates"
                prov = re_.get("prov") or {}

                if changed:
                    prov = _mark_retyped(prov)
                valid_edges.append({
                    "from": f,
                    "to": t,
                    "type": et,
                    "conf": float(re_.get("conf", 0.0)),
                    "prov": prov,
                })
        # если LLM вернул совсем пусто — остаёмся на прежней версии
        # if valid_nodes and valid_edges:
        #     refined_nodes, refined_edges = valid_nodes, valid_edges
        # если LLM хорошо нормализовал узлы, но рёбра не успел/не смог — это всё равно полезно
        if valid_nodes:
            refined_nodes = valid_nodes
            refined_edges = valid_edges  # может быть пустым — ниже сгенерим каркас

    # --- 8) итоговые узлы/рёбра ---
    if refined_nodes is not None:
        nodes = refined_nodes
    if refined_edges is not None:
        edges = refined_edges

    # финальная зачистка: порог + дедуп по (from,to,type)
    edges = [e for e in edges if float(e.get("conf", 0.0)) >= 0.55]

    # --- вернуть геометрию из снапшота S1
    for n in nodes:
        fb = orig_geo.get(n.get("id"))
        _ensure_geo(n, fb)

    tmp = {}
    for e in edges:
        k = (e["from"], e["to"], e["type"])
        if k not in tmp or float(e.get("conf", 0.0)) > tmp[k]["conf"]:
            tmp[k] = {
                "from": e["from"],
                "to": e["to"],
                "type": e["type"],
                "conf": float(e.get("conf", 0.0)),
                "prov": e.get("prov"),
            }
    edges = list(tmp.values())

    if not edges:
        edges = ensure_minimal_backbone(nodes, edges)  # noqa: F405

    # --- 9) раскладка по колонкам и запись ---
    graph = _layout_columns(doc_id, nodes, edges)  # noqa: F405
    _write_json(str(out_path), graph)  # noqa: F405

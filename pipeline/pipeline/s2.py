# pipeline/pipeline/s2.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

def run_s2(export_dir: str) -> Dict[str, Any]:
    exp = Path(export_dir)
    s1g = exp / "s1_graph.json"
    s1dbg = exp / "s1_debug.json"
    final = exp / "graph.json"          # ← итоговый файл для фронта
    out_dbg = exp / "s2_debug.json"

    base = {"doc_id": None, "nodes": [], "edges": []}
    if s1g.exists():
        base = json.loads(s1g.read_text())

    # --- здесь делаете нормализацию/линковку/фильтрацию ---
    nodes = base.get("nodes", [])
    edges = base.get("edges", [])

    # пример: жёсткий дедуп по (type, text)
    seen = set()
    dedup_nodes = []
    for n in nodes:
        key = (n["type"], n["text"].strip().lower())
        if key in seen: continue
        seen.add(key); dedup_nodes.append(n)
    nodes = dedup_nodes

    # можно пересчитать conf/thresholds, дорисовать связи и т.п.

    final_graph = {"doc_id": base.get("doc_id"), "nodes": nodes, "edges": edges}
    final.write_text(json.dumps(final_graph, ensure_ascii=False, indent=2))

    # debug
    s1 = json.loads(s1dbg.read_text()) if s1dbg.exists() else None
    debug = {
        "doc_id": final_graph["doc_id"],
        "summary": {
            "nodes_in": len(base.get("nodes", [])),
            "edges_in": len(base.get("edges", [])),
            "nodes_out": len(nodes),
            "edges_out": len(edges),
            "reason": "ok" if nodes or edges else "empty_after_s1",
        },
        "hints": {
            "section_names": (s1 or {}).get("summary", {}).get("section_names", []),
            "node_threshold": (s1 or {}).get("summary", {}).get("node_threshold"),
            "edge_threshold": (s1 or {}).get("summary", {}).get("edge_threshold"),
        }
    }
    out_dbg.write_text(json.dumps(debug, ensure_ascii=False, indent=2))
    return debug


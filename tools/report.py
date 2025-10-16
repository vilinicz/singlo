#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

def _read_json(p: Path):
    if not p.exists(): return None
    return json.loads(p.read_text(encoding="utf-8"))

def _tally_nodes_by_type(nodes):
    d = defaultdict(int)
    for n in nodes or []:
        d[n.get("type","Unknown")] += 1
    return dict(d)

def _tally_edges_by_type(edges):
    d = defaultdict(int)
    for e in edges or []:
        d[e.get("type","Unknown")] += 1
    return dict(d)

def generate_report(doc_dir: str) -> dict:
    base = Path(doc_dir)
    s0 = _read_json(base / "s0.json") or {}
    s1g = _read_json(base / "s1_graph.json") or {}
    s1d = _read_json(base / "s1_debug.json") or {}

    # S0 sections
    sections = (s0.get("sections") or [])
    section_counts = defaultdict(int)
    for s in sections:
        name = (s.get("name") or "Unknown").strip() or "Unknown"
        section_counts[name] += 1

    # S1 nodes/edges
    nodes = s1g.get("nodes") or []
    edges = s1g.get("edges") or []
    nodes_by_type = _tally_nodes_by_type(nodes)
    edges_by_type = _tally_edges_by_type(edges)

    # rules used
    packs = (s1d.get("rules") or {}).get("packs_active") or []
    rules_fired = (s1d.get("rules") or {}).get("rules_fired") or []
    rules_total = (s1d.get("rules") or {}).get("rules_fired_total") or len(rules_fired)

    report = {
        "doc_id": s1g.get("doc_id") or s0.get("doc_id") or base.name,
        "source_pdf": s0.get("source_pdf"),
        "s0": {
            "sections_total": len(sections),
            "section_counts": dict(section_counts)
        },
        "s1": {
            "nodes_total": len(nodes),
            "nodes_by_type": nodes_by_type,
            "edges_total": len(edges),
            "edges_by_type": edges_by_type,
            "packs_active": packs,
            "rules_fired_total": rules_total,
            "rules_fired": rules_fired[:200]  # safety truncate
        }
    }
    return report

def main():
    if len(sys.argv) < 2:
        print("Usage: tools/report.py /dataset")
        sys.exit(2)
    rep = generate_report(sys.argv[1])
    out_json = Path(sys.argv[1]) / "report.json"
    out_md = Path(sys.argv[1]) / "report.md"
    # save json
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    # save md
    md = []
    md.append(f"# Report: {rep['doc_id']}\n")
    if rep.get("source_pdf"):
        md.append(f"- source: `{rep['source_pdf']}`\n")
    md.append("## S0 (sections)\n")
    md.append(f"- total sections: **{rep['s0']['sections_total']}**\n")
    for k,v in sorted(rep['s0']['section_counts'].items()):
        md.append(f"  - {k}: {v}")
    md.append("\n## S1 (extraction)\n")
    md.append(f"- nodes total: **{rep['s1']['nodes_total']}**")
    for k,v in sorted(rep['s1']['nodes_by_type'].items()):
        md.append(f"  - {k}: {v}")
    md.append(f"- edges total: **{rep['s1']['edges_total']}**")
    for k,v in sorted(rep['s1']['edges_by_type'].items()):
        md.append(f"  - {k}: {v}")
    md.append("\n### Rule packs used\n")
    for p in rep['s1']['packs_active'] or []:
        md.append(f"- {p}")
    md.append(f"\n### Rules fired ({rep['s1']['rules_fired_total']})\n")
    for r in rep['s1']['rules_fired'] or []:
        md.append(f"- {r}")
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"✓ Saved {out_json}")
    print(f"✓ Saved {out_md}")

if __name__ == "__main__":
    main()

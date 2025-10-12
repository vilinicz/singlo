# pipeline/pipeline/worker.py
"""
RQ-воркер: оркестрация S0 → S1 → S2 (пока S2-заглушка)
Сохраняет прогресс и артефакты в Redis для отображения во фронте.
"""

import os
import time
import json
from pathlib import Path
from rq import Worker, Queue
from rq.connections import Connection
import redis
from pipeline.s1 import run_s1
from pipeline.s0 import build_s0

# -------- Redis helpers --------
def _r():
    return redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))

def _status_key(doc_id): return f"status:{doc_id}"

def _init_status(doc_id):
    r = _r()
    payload = {
        "state": "running",
        "stage": "init",
        "started_at": time.time(),
        "stages": json.dumps([]),
        "artifacts": json.dumps({}),
    }
    r.hset(_status_key(doc_id), mapping=payload)

def _push_stage(doc_id, name, t0, notes=None):
    r = _r()
    stages = json.loads(r.hget(_status_key(doc_id), "stages") or "[]")
    t1 = time.time()
    stages.append({
        "name": name,
        "t_start": t0,
        "t_end": t1,
        "duration_ms": int((t1 - t0) * 1000),
        "notes": notes or ""
    })
    r.hset(_status_key(doc_id), "stages", json.dumps(stages))
    r.hset(_status_key(doc_id), "stage", name)

def _set_artifact(doc_id, key, path):
    r = _r()
    artifacts = json.loads(r.hget(_status_key(doc_id), "artifacts") or "{}")
    artifacts[key] = path
    r.hset(_status_key(doc_id), "artifacts", json.dumps(artifacts))

def _finish_status(doc_id, ok=True, err_msg=None):
    r = _r()
    r.hset(_status_key(doc_id), mapping={
        "state": "done" if ok else "error",
        "ended_at": time.time(),
        "error": err_msg or ""
    })

# -------- основной pipeline --------
def run_pipeline(doc_id: str):
    """
    S0 → S1 → S2 (заглушка)
    """
    _init_status(doc_id)
    data_dir = Path("/app/data") / doc_id
    export_dir = Path("/app/export") / doc_id
    export_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = data_dir / "input.pdf"
    s0_path = data_dir / "s0.json"
    graph_path = export_dir / "graph.json"
    rules_path = os.getenv("S1_RULES_PATH", "/app/rules/common.yaml")

    try:
        # ---------- S0 ----------
        t0 = time.time()
        if not s0_path.exists():
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            build_s0(str(pdf_path), str(data_dir))
        _push_stage(doc_id, "S0", t0, notes="real S0")
        _set_artifact(doc_id, "s0", str(s0_path))

        # ---------- S1 ----------
        t1 = time.time()
        run_s1(str(s0_path), rules_path, str(graph_path))
        _push_stage(doc_id, "S1", t1, notes="regex extraction")
        _set_artifact(doc_id, "graph", str(graph_path))

        # ---------- S2 (заглушка) ----------
        t2 = time.time()
        time.sleep(0.05)
        _push_stage(doc_id, "S2", t2, notes="semantic linking stub")

        # ---------- finalize ----------
        t3 = time.time()
        _push_stage(doc_id, "finalize", t3)
        _finish_status(doc_id, ok=True)
        return {"doc_id": doc_id, "graph": str(graph_path)}

    except Exception as e:
        # если упало — создаём пустой граф для фронта
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            Path(graph_path).write_text(json.dumps({
                "doc_id": doc_id,
                "nodes": [],
                "edges": [],
                "error": str(e)
            }, indent=2))
        except Exception:
            pass
        _finish_status(doc_id, ok=False, err_msg=str(e))
        raise

# -------- вспомогательная задача: только S0 --------
def run_s0(doc_id: str):
    data_dir = Path("/app/data") / doc_id
    pdf_path = data_dir / "input.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    t0 = time.time()
    _init_status(doc_id)
    s0 = build_s0(str(pdf_path), str(data_dir))
    _push_stage(doc_id, "S0", t0, notes="standalone S0")
    _set_artifact(doc_id, "s0", str(data_dir / "s0.json"))
    _finish_status(doc_id, ok=True)
    return {"doc_id": doc_id, "s0": str(data_dir / "s0.json")}

# -------- entrypoint --------
def main():
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    conn = redis.from_url(redis_url)
    with Connection(conn):
        w = Worker([Queue("singularis")])
        w.work(with_scheduler=True)

if __name__ == "__main__":
    main()

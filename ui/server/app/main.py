import os, json, time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
from rq import Queue

# -------------------- App --------------------
app = FastAPI(title="Singularis API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# -------------------- Config --------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
DATA_DIR = Path("/app/data")
EXPORT_DIR = Path("/app/export")
RULES_PATH = os.getenv("S1_RULES_PATH", "/app/rules/common.yaml")

# -------------------- Redis helpers --------------------
def _r():
    return redis.from_url(REDIS_URL)

def _status_key(doc_id):
    return f"status:{doc_id}"

# -------------------- Models --------------------
class ParseResp(BaseModel):
    doc_id: str
    s0_path: str

# -------------------- Routes --------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse", response_model=ParseResp)
async def parse_pdf(doc_id: str, file: UploadFile = File(...)):
    """
    Мини-S0: сохраняем PDF и создаём простую s0-заглушку.
    Этого достаточно, чтобы пройти цикл Parse -> Extract -> Graph.
    """
    doc_dir = DATA_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = doc_dir / "input.pdf"
    with pdf_path.open("wb") as f:
        f.write(await file.read())

    # S0-заглушка: одна секция + caption, чтобы S1 не падал
    s0 = {
        "doc_id": doc_id,
        "sections": [
            {"name": "Results", "text": "Example results text with p=0.008 and 9% increase."}
        ],
        "captions": [
            {"id": "Fig1", "text": "Figure 1: Example caption with significant effect (p=0.008)."}
        ]
    }
    s0_path = doc_dir / "s0.json"
    s0_path.write_text(json.dumps(s0, ensure_ascii=False, indent=2))
    return {"doc_id": doc_id, "s0_path": str(s0_path)}

@app.post("/extract")
def extract_graph(doc_id: str):
    """
    Кладём задачу в очередь RQ; прогресс смотри через /status/{doc_id}
    """
    q = Queue("singularis", connection=_r())
    job = q.enqueue("pipeline.worker.run_pipeline", doc_id, job_timeout=600)
    # сразу инициализируем статус, чтобы фронт видел прогресс
    _r().hset(_status_key(doc_id), mapping={
        "state": "queued",
        "stage": "queued",
        "started_at": time.time()
    })
    return {"job_id": job.id, "status": job.get_status()}

@app.get("/status/{doc_id}")
def status(doc_id: str):
    r = _r()
    data = r.hgetall(_status_key(doc_id))
    if not data:
        raise HTTPException(404, "no status for doc")
    out = {k.decode(): (v.decode() if isinstance(v,(bytes,bytearray)) else v) for k,v in data.items()}
    for k in ("stages","artifacts"):
        if k in out:
            try: out[k] = json.loads(out[k])
            except: pass
    if "started_at" in out and "ended_at" in out:
        out["duration_ms"] = int((float(out["ended_at"]) - float(out["started_at"])) * 1000)
    return out

@app.get("/preview/{doc_id}/{artifact}")
def preview(doc_id: str, artifact: str):
    base = EXPORT_DIR if artifact == "graph" else DATA_DIR
    filename = "graph.json" if artifact == "graph" else "s0.json"
    path = base / doc_id / filename
    if not path.exists():
        raise HTTPException(404, "artifact not found")
    txt = path.read_text()
    if len(txt) > 50_000:
        txt = txt[:50_000] + "\n... [truncated]"
    return {"artifact": artifact, "path": str(path), "preview": txt}

@app.get("/graph/{doc_id}")
def get_graph(doc_id: str):
    gpath = EXPORT_DIR / doc_id / "graph.json"
    if not gpath.exists():
        raise HTTPException(404, f"graph not found for {doc_id}")
    return json.loads(gpath.read_text())

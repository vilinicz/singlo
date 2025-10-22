"""FastAPI gateway for Singularis: upload/queue, status, preview, graph, themes."""
import os, json, time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import redis
from uuid import uuid4
import re
from rq import Queue
from typing import Optional, List
# Import themes router from the mounted pipeline package.
# In the API container, the repository's `./pipeline` directory is mounted at `/app/pipeline`,
# and the actual Python package lives under `/app/pipeline/pipeline`.
# Hence the absolute module path here is `pipeline.pipeline.themes_router`.
from pipeline.pipeline.themes_router import preload as themes_preload
from ui.server.app.testbench import router as test_router


# -------------------- App --------------------
app = FastAPI(title="Singularis API", version="0.1")
app.include_router(test_router)

# Разрешённые источники (локально: vite/dev сервер)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # ⟵ не '*', а конкретные адреса
    allow_credentials=True,         # ⟵ нужно для cookies/Authorization
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Themes registry (preload once) ----------
RULES_BASE_DIR = os.getenv("RULES_BASE_DIR", "/app/legacy_rules")
THEMES_DIR = os.getenv("THEMES_DIR", str(Path(RULES_BASE_DIR) / "themes"))
THEME_REGISTRY = themes_preload(THEMES_DIR)  # быстрый реестр + инвертированный индекс

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# -------------------- Config --------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
DATA_DIR = Path("/app/data")
EXPORT_DIR = Path("/app/export")
RULES_PATH = os.getenv("S1_RULES_PATH", "/app/legacy_rules/common.yaml")

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
    """Liveness probe used by local dev/docker healthchecks."""
    return {"status": "ok"}


@app.post("/extract")
# Queue pipeline for an existing doc_id; see /status/{doc_id} for progress
def extract_graph(doc_id: str, theme: str = Query(default="auto")):
    """
    Кладём задачу в очередь RQ; прогресс смотри через /status/{doc_id}
    """
    q = Queue("singularis", connection=_r())
    # Передаём явные именованные аргументы, чтобы worker получил корректные параметры
    job = q.enqueue(
        "pipeline.worker.run_pipeline",
        pdf_path=str(DATA_DIR / doc_id),  # из каталога /app/data/<doc_id>
        doc_id=doc_id,
        theme=theme,
        job_timeout=600,
    )
    # сразу инициализируем статус, чтобы фронт видел прогресс
    _r().hset(_status_key(doc_id), mapping={
        "state": "queued",
        "stage": "queued",
        "started_at": time.time(),
        "theme": theme
    })
    return {"job_id": job.id, "status": job.get_status()}


@app.get("/status/{doc_id}")
def status(doc_id: str, request: Request):
    """Return current job status and parsed JSON fields.

    Adds artifacts.pdf_url pointing to `/pdf/{doc_id}` for convenience.
    """
    r = _r()
    data = r.hgetall(_status_key(doc_id))
    if not data:
        raise HTTPException(404, "no status for doc")
    out = {k.decode(): (v.decode() if isinstance(v, (bytes, bytearray)) else v) for k, v in data.items()}
    for k in ("stages", "artifacts"):
        if k in out:
            try:
                out[k] = json.loads(out[k])
            except:
                pass
    if "started_at" in out and "ended_at" in out:
        out["duration_ms"] = int((float(out["ended_at"]) - float(out["started_at"])) * 1000)
    # Ensure artifacts dict and inject a URL to the source PDF
    artifacts = out.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    artifacts["pdf_url"] = f"/pdf/{doc_id}"
    out["artifacts"] = artifacts
    return out


@app.get("/preview/{doc_id}/{artifact}")
# Return small text preview of an artifact (s0|s1|s2|graph)
def preview(doc_id: str, artifact: str):
    # куда и какой файл смотреть
    mapping = {
        "s0": (DATA_DIR, "s0.json"),
        "graph": (EXPORT_DIR, "graph.json"),
        "s1": (EXPORT_DIR, "s1_debug.json"),
        "s2": (EXPORT_DIR, "s2_debug.json"),
    }
    if artifact not in mapping:
        raise HTTPException(404, f"unknown artifact '{artifact}'")

    base, fname = mapping[artifact]
    path = base / doc_id / fname
    if not path.exists():
        raise HTTPException(404, f"artifact not found: {path}")

    txt = path.read_text()
    if len(txt) > 50_000:
        txt = txt[:50_000] + "\n... [truncated]"
    return {"artifact": artifact, "path": str(path), "preview": txt}


@app.get("/graph/{doc_id}")
# Return full graph JSON for a completed document
def get_graph(doc_id: str):
    gpath = EXPORT_DIR / doc_id / "graph.json"
    if not gpath.exists():
        raise HTTPException(404, f"graph not found for {doc_id}")
    return json.loads(gpath.read_text())


@app.get("/pdf/{doc_id}")
def get_pdf(doc_id: str):
    """Serve the uploaded source PDF for a document id."""
    pdf_path = DATA_DIR / doc_id / "input.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, f"pdf not found for {doc_id}")
    # Inline display in browser
    return FileResponse(str(pdf_path), media_type="application/pdf")


def _slugify(name: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '-', name).strip('-').lower()
    return slug or 'doc'


# Upload a PDF, store under data/<doc_id>/input.pdf and enqueue S0->S2
@app.post("/parse", response_model=ParseResp)
async def parse_pdf(
        doc_id: str | None = Query(default=None),
        file: UploadFile = File(...),
        theme: str = Query(default="auto")  # <--- добавили
):
    """
    Принимаем PDF, генерируем doc_id, сохраняем input.pdf и
    АСИНХРОННО запускаем полный пайплайн S0→S1→S2 в воркере.
    """
    # 1) doc_id
    if not doc_id or doc_id.strip() == "" or doc_id == "demo":
        stem = Path(file.filename or "doc").stem
        doc_id = f"{_slugify(stem)}-{str(uuid4())[:8]}"

    # 2) сохранить PDF
    doc_dir = DATA_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = doc_dir / "input.pdf"
    with pdf_path.open("wb") as f:
        f.write(await file.read())

    # 3) пометить статус "queued" и запустить конвейер
    r = _r()
    r.hset(_status_key(doc_id), mapping={
        "state": "queued",
        "stage": "queued",
        "started_at": time.time(),
        "stages": json.dumps([]),
        "artifacts": json.dumps({}),
        "theme": theme  # <--- пишем для прозрачности
    })
    q = Queue("singularis", connection=r)
    # Передаём явные именованные аргументы: путь к данным документа, doc_id и theme
    q.enqueue(
        "pipeline.worker.run_pipeline",
        pdf_path=str(doc_dir),
        doc_id=doc_id,
        theme=theme,
        job_timeout=900,
    )

    return {"doc_id": doc_id, "s0_path": str(doc_dir / "s0.json")}


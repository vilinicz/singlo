# pipeline/pipeline/worker.py
"""
RQ-воркер: оркестрация S0 → S1 → S2
Сохраняет прогресс и артефакты в Redis для отображения во фронте.
Добавлено:
- Тематический роутинг: приём параметра `theme` (auto | name | name1,name2)
- Предзагрузка реестра тем и прокидка в S1
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, List

from rq import Worker, Queue
from rq.connections import Connection
import redis

from pipeline.s1 import run_s1
from pipeline.s0 import build_s0
from pipeline.s2 import run_s2

# темы: реестр и роутинг
from pipeline.themes_router import preload as themes_preload


RULES_BASE_DIR = os.getenv("RULES_BASE_DIR", "/app/rules")
THEMES_DIR = os.getenv("THEMES_DIR", str(Path(RULES_BASE_DIR) / "themes"))

Path(RULES_BASE_DIR).mkdir(parents=True, exist_ok=True)
Path(THEMES_DIR).mkdir(parents=True, exist_ok=True)

try:
    THEME_REGISTRY = themes_preload(THEMES_DIR)
except Exception:
    THEME_REGISTRY = None

# -------- Redis helpers --------
def _r():
    return redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))


def _status_key(doc_id): return f"status:{doc_id}"


def _init_status(doc_id, *, theme: str = "auto"):
    r = _r()
    payload = {
        "state": "running",
        "stage": "init",
        "started_at": time.time(),
        "stages": json.dumps([]),
        "artifacts": json.dumps({}),
        "theme": theme,
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


# -------- Темы: глобальный реестр (preload once per process) --------
THEMES_DIR = os.getenv("THEMES_DIR", "/app/themes")
try:
    THEME_REGISTRY = themes_preload(THEMES_DIR)
except Exception:
    # Не валим воркер, если тем нет — S1 сможет работать только с common.yaml
    THEME_REGISTRY = None


def _parse_theme_override(theme_str: Optional[str]) -> Optional[List[str]]:
    """
    "auto" | None -> None
    "biomed" -> ["biomed"]
    "biomed,physics" -> ["biomed","physics"]
    Пустые части отбрасываются.
    """
    if not theme_str:
        return None
    ts = theme_str.strip()
    if ts.lower() == "auto":
        return None
    parts = [p.strip() for p in ts.split(",") if p.strip()]
    return parts or None


# -------- основной pipeline --------
def run_pipeline(doc_id: str, theme: str = "auto"):
    """
    Полный пайплайн: S0 → S1 → S2
    Параметры:
      - doc_id: идентификатор документа (рабочая директория /app/data/{doc_id})
      - theme : "auto" (роутер тем) или конкр. тема/список тем ("biomed" | "biomed,physics")
    """
    _init_status(doc_id, theme=theme)

    data_dir = Path("/app/data") / doc_id
    export_dir = Path("/app/export") / doc_id
    export_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = data_dir / "input.pdf"
    s0_path = data_dir / "s0.json"
    graph_path = export_dir / "graph.json"

    # Совместимость: базовый путь к правилам (common), может быть добавлен к тематическим
    rules_path = os.getenv("S1_RULES_PATH", str(Path(RULES_BASE_DIR) / "common.yaml"))

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
        # распарсить override тем из строки
        override_list = _parse_theme_override(theme)

        run_s1(
            str(s0_path),
            rules_path,
            str(graph_path),
            themes_root=THEMES_DIR,  # где лежат themes/*
            theme_override=override_list,  # None => auto
            theme_registry=THEME_REGISTRY  # заранее предзагруженный реестр
        )
        _push_stage(doc_id, "S1", t1, notes=f"regex extraction (theme={theme})")
        _set_artifact(doc_id, "s1_graph", str(export_dir / "s1_graph.json"))

        # ---------- S2 ----------
        t2 = time.time()
        run_s2(str(export_dir))
        _push_stage(doc_id, "S2", t2, notes="semantic linking")
        _set_artifact(doc_id, "graph", str(graph_path))

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
            }, indent=2), encoding="utf-8")
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

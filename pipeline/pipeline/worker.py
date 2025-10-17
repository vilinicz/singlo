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
from typing import Optional, List, Union

from rq import Worker, Queue
from rq.connections import Connection
import redis

from .s0 import build_s0
from .s1 import run_s1
from .s2 import run_s2

# темы: реестр и роутинг
from .themes_router import preload as themes_preload

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


def _resolve_pdf_like(p: str | Path, doc_id: Optional[str] = None) -> Path:
    """
    Принимает путь к .pdf ИЛИ к каталогу ИЛИ «кривой» путь.
    Возвращает существующий путь к файлу .pdf.
    Поиск:
      1) если p — существующий .pdf → его;
      2) если p — каталог → <p>/input.pdf или первый *.pdf;
      3) если данных нет, пробуем в DATA_DIR по doc_id: DATA_DIR/doc_id/(input.pdf|*.pdf|doc_id.pdf).
    """
    p = Path(str(p))
    # 1) уже .pdf
    if p.suffix.lower() == ".pdf" and p.exists():
        return p.resolve()
    # 2) каталог
    if p.exists() and p.is_dir():
        cand = p / "input.pdf"
        if cand.exists():
            return cand.resolve()
        any_pdf = list(p.glob("*.pdf"))
        if any_pdf:
            return any_pdf[0].resolve()
    # 3) DATA_DIR / doc_id
    data_dir = Path(os.getenv("DATA_DIR", "/app/data")).resolve()
    name = doc_id or p.name  # если doc_id не задан — используем имя каталога
    base = data_dir / name
    cand1 = base / "input.pdf"
    if cand1.exists():
        return cand1.resolve()
    any_pdf = list(base.glob("*.pdf"))
    if any_pdf:
        return any_pdf[0].resolve()
    cand2 = data_dir / f"{name}.pdf"
    if cand2.exists():
        return cand2.resolve()
    raise FileNotFoundError(f"PDF not found (tried): {p}, {cand1}, {base}/*.pdf, {cand2}")


def _ensure_pdf_path(p: str | Path) -> Path:
    """Принимаем либо путь к .pdf, либо каталог; для каталога ищем input.pdf или любой *.pdf."""
    if not p:
        raise TypeError("pdf_path is required")
    p = Path(p)
    if p.suffix.lower() == ".pdf":
        return p.resolve()
    if p.is_dir():
        cand = p / "input.pdf"
        if cand.exists():
            return cand.resolve()
        pdfs = list(p.glob("*.pdf"))
        if pdfs:
            return pdfs[0].resolve()
    # если это не .pdf и не каталог — пусть упадёт дальше с FileNotFoundError
    return p.resolve()


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


def _normalize_theme_override(theme: Optional[Union[str, List[str]]],
                              theme_override: Optional[List[str]]) -> Optional[List[str]]:
    """
    Принимает theme (строка 'biomed,physics' | список) и/или theme_override (список),
    возвращает нормализованный список тем или None.
    Приоритет: theme_override (если задан) > theme.
    """
    if theme_override:
        return [str(t).strip() for t in theme_override if str(t).strip()]
    if theme is None:
        return None
    if isinstance(theme, str):
        parts = [x.strip() for x in theme.replace(";", ",").split(",") if x.strip()]
        return parts or None
    if isinstance(theme, list):
        parts = [str(x).strip() for x in theme if str(x).strip()]
        return parts or None
    return None


def run_pipeline(
        pdf_path: Optional[str] = None,
        rules_path: Optional[str] = None,
        export_dir: Optional[str] = None,
        *,
        themes_root: Optional[str] = None,
        theme_override: Optional[List[str]] = None,
        theme: Optional[Union[str, List[str]]] = None,
        doc_id: Optional[str] = None,
):
    """
    Универсальный раннер S0→S1→S2.

    pdf_path: путь к PDF
    rules_path: путь к rules/common.yaml
    export_dir: базовая директория вывода артефактов (export/)
    themes_root: директория с themes/* (опционально)
    theme_override: ['biomed','physics'] — ручной выбор тем (опционально; старое имя)
    theme: 'biomed,physics' | ['biomed','physics'] — альтернативное имя параметра (опционально)
    doc_id: форсированный идентификатор для имени папки и артефактов (опционально)
    """
    # дефолты из окружения
    export_dir = export_dir or os.getenv("EXPORT_DIR", "/app/export")
    rules_path = rules_path or os.getenv("RULES_PATH", "/app/rules/common.yaml")
    themes_root = themes_root or os.getenv("THEMES_ROOT", "/app/rules/themes")

    # нормализация пути (pdf_path может быть каталогом /app/data/<doc_id> или любым «похожим» значением)
    pdf_path = _ensure_pdf_path(pdf_path)
    export_dir = Path(export_dir).resolve();
    export_dir.mkdir(parents=True, exist_ok=True)
    rules_path = str(Path(rules_path).resolve())
    themes_root = str(Path(themes_root).resolve())

    # --- S0 ---
    t0 = time.time()
    # 1) резолвим реальный PDF-файл (учитывая doc_id, если передан)
    resolved_pdf = _resolve_pdf_like(pdf_path, doc_id=doc_id)

    # 2) выбираем doc_id для папки вывода (до запуска S0)
    doc_id_eff = doc_id or resolved_pdf.stem
    _init_status(doc_id_eff, theme=str(theme) if theme is not None else "auto")
    # ВАЖНО: S0 сохраняем в DATA_DIR, т.к. фронт и другие части кода ждут s0.json там
    data_dir = Path(os.getenv("DATA_DIR", "/app/data")).resolve() / doc_id_eff
    data_dir.mkdir(parents=True, exist_ok=True)
    # S1/S2 и графы — в EXPORT_DIR
    out_dir = Path(export_dir).resolve() / doc_id_eff
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 3) S0: сохраняем артефакты в DATA_DIR/<doc_id>
        s0_stage_t = time.time()
        s0 = build_s0(str(resolved_pdf), str(data_dir))
        # при необходимости фиксируем doc_id в артефактах
        if doc_id:
            s0["doc_id"] = doc_id
        else:
            doc_id = s0.get("doc_id") or doc_id_eff
        # сохраняем s0.json
        s0_path = data_dir / "s0.json"
        s0_path.write_text(json.dumps(s0, ensure_ascii=False, indent=2), encoding="utf-8")
        _push_stage(doc_id_eff, "S0", s0_stage_t)
        _set_artifact(doc_id_eff, "s0", str(s0_path))

        # --- normalize theme params ---
        themes_sel = _normalize_theme_override(theme=theme, theme_override=theme_override)

        # --- S1 ---
        s1_stage_t = time.time()
        graph_path = out_dir / "graph.json"  # будущее место для S2-выхода
        run_s1(
            str(s0_path),
            rules_path,
            str(graph_path),
            themes_root=themes_root,
            theme_override=themes_sel
        )
        s1_graph = out_dir / "s1_graph.json"
        if not s1_graph.exists():
            raise RuntimeError(f"s1_graph.json not found at {s1_graph}")
        _push_stage(doc_id_eff, "S1", s1_stage_t)
        _set_artifact(doc_id_eff, "s1", str(s1_graph))

        # --- S2 ---
        s2_stage_t = time.time()
        run_s2(str(s1_graph), str(graph_path))  # создаст graph.json и s2_debug.json в out_dir
        _push_stage(doc_id_eff, "S2", s2_stage_t)
        _set_artifact(doc_id_eff, "graph", str(graph_path))
        s2_debug = out_dir / "s2_debug.json"
        if s2_debug.exists():
            _set_artifact(doc_id_eff, "s2", str(s2_debug))

        _finish_status(doc_id_eff, ok=True)

        return {
            "doc_id": doc_id_eff,
            "s0": str(s0_path),
            "s1": str(s1_graph),
            "graph": str(graph_path),
            "s2_debug": str(s2_debug) if s2_debug.exists() else "",
            "t_start": t0,
        }
    except Exception as e:
        _finish_status(doc_id_eff, ok=False, err_msg=str(e))
        raise


# -------- вспомогательная задача: только S0 --------
def run_s0_only(doc_id: str):
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

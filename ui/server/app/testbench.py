from fastapi import APIRouter, Query, HTTPException
from pathlib import Path
import os
import time
import json
import logging
import traceback
import json
import shutil
import hashlib
from typing import List, Dict, Any

from tools.report import generate_report  # см. ниже пункт 2 — CLI уже писали, он реиспользуем
from pipeline.pipeline.worker import run_pipeline  # если у вас есть синхронный раннер
from fastapi.responses import HTMLResponse

# единоразовая настройка логгера для этого роутера
logger = logging.getLogger("testbench.run")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
    # INFO по умолчанию; поставь DEBUG, если нужно больше деталей
    logger.setLevel(logging.INFO)

router = APIRouter(prefix="/test", tags=["testbench"])

DATASET_ROOT = Path("/app/dataset")  # пробросим в Docker
WORKDIR = Path("/app/workdir").resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data")).resolve()  # ⟵ добавь это
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "/app/export")).resolve()  # ⟵ и это (полезно для отчётов)

# Какие имена файлов считаем артефактами
S0_NAMES = ("s0.json",)
S1_NAMES = ("s1_debug.json", "s1.json", "s1_graph.json")
S2_NAMES = ("s2.json",)
G_NAMES = ("graph.json", "final_graph.json")


@router.get("/list")
def list_pdfs(subdir: str = ""):
    base = (DATASET_ROOT / subdir).resolve()
    if not base.exists() or not str(base).startswith(str(DATASET_ROOT)):
        raise HTTPException(404, f"Not found: {base}")
    pdfs = [str(p.relative_to(DATASET_ROOT)) for p in base.rglob("*.pdf")]
    return {"root": str(DATASET_ROOT), "count": len(pdfs), "items": pdfs[:2000]}


def _clean_tree(root: Path):
    """
    Безопасно чистим ВСЁ содержимое каталога (только внутри разрешённых корней).
    """
    root = root.resolve()
    allowed = {DATA_DIR.resolve(), EXPORT_DIR.resolve()}
    if root not in allowed:
        raise RuntimeError(f"Refuse to clean non-allowed path: {root}")
    for p in root.iterdir():
        try:
            if p.is_dir():
                import shutil;
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Cannot remove {p}: {e}")


def _doc_id_from_pdf(pdf: Path) -> str:
    # Локальный импорт — гарантирует доступ даже если модуль кэшировался странно
    import hashlib as _hashlib

    try:
        p = pdf if isinstance(pdf, Path) else Path(str(pdf))
        full = str(p.resolve())
    except Exception:
        p = Path(str(pdf))
        full = str(p)

    stem = p.stem.strip().replace(" ", "_")
    stem = "".join(ch for ch in stem if ch.isalnum() or ch in ("_", "-", "."))

    h = _hashlib.md5(full.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{h}"


def _prepare_data_dir(doc_id: str, src_pdf: Path) -> Path:
    """
    Создаёт /app/data/{doc_id} и кладёт туда source.pdf.
    Исходный PDF не трогаем.
    """
    out_dir = (DATA_DIR / doc_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_pdf = out_dir / "input.pdf"
    # копируем только если нет или размер отличается (чтобы не гонять лишний раз)
    if (not dst_pdf.exists()) or (dst_pdf.stat().st_size != src_pdf.stat().st_size):
        shutil.copy2(src_pdf, dst_pdf)
    return out_dir


@router.post("/run")
def run_over_dataset(
        subdir: str = "",
        limit: int = Query(50, ge=1, le=2000),
        theme: str = "auto",  # ← учитываем override от фронта
        clean: int = 1,
):
    """
    Идём по PDF в /app/dataset[/subdir], для каждого:
      - генерим doc_id
      - готовим /app/data/{doc_id}/source.pdf
      - вызываем run_pipeline(doc_id, theme)
    НИЧЕГО не передаем как путь в run_pipeline — только doc_id!
    """
    t0 = time.perf_counter()

    if clean:
        logger.info("Cleaning DATA_DIR and EXPORT_DIR before run…")
        _clean_tree(DATA_DIR)
        _clean_tree(EXPORT_DIR)

    base = (DATASET_ROOT / (subdir or "")).resolve()
    if not base.exists() or not str(base).startswith(str(DATASET_ROOT)):
        msg = f"Not found or outside DATASET_ROOT: {base}"
        logger.error(msg)
        raise HTTPException(status_code=404, detail=msg)

    processed: List[Dict[str, Any]] = []
    seen = 0
    ok_count = 0
    err_count = 0

    logger.info(f"/test/run start | base='{base}' | limit={limit} | theme='{theme}'")

    for pdf in base.rglob("*.pdf"):
        if seen >= limit:
            break
        seen += 1

        try:
            rel = str(pdf.resolve().relative_to(DATASET_ROOT))
        except Exception:
            rel = str(pdf)

        if not pdf.is_file():
            processed.append({"pdf": rel, "error": "skip: not a file"})
            continue

        doc_id = _doc_id_from_pdf(pdf)
        t_file = time.perf_counter()
        logger.info(f"[{seen}/{limit}] doc_id={doc_id} <- {rel}")

        try:
            _prepare_data_dir(doc_id, pdf)
            rv = run_pipeline(doc_id=doc_id, theme=theme)  # ← ключевой момент
            elapsed_ms = int((time.perf_counter() - t_file) * 1000)
            ok_count += 1
            processed.append({
                "pdf": rel,
                "doc_id": doc_id,
                "workdir": rv.get("workdir") or f"/app/data/{doc_id}",
                "elapsed_ms": elapsed_ms,
            })
            logger.info(f"[{seen}] OK doc_id={doc_id} | {elapsed_ms} ms")
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t_file) * 1000)
            err_count += 1
            logger.exception(f"[{seen}] FAIL doc_id={doc_id} <- {rel} | {elapsed_ms} ms | {e}")
            processed.append({
                "pdf": rel,
                "doc_id": doc_id,
                "error": f"{e.__class__.__name__}: {e}",
                "elapsed_ms": elapsed_ms,
            })

    total_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(f"/test/run done | seen={seen} | ok={ok_count} | err={err_count} | total={total_ms} ms")
    return {
        "ok": True,
        "base": str(base),
        "limit": limit,
        "seen": seen,
        "ok_count": ok_count,
        "err_count": err_count,
        "elapsed_ms": total_ms,
        "processed": processed,
    }


def _safe_load_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_first(dirpath: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        p = dirpath / name
        if p.exists():
            return p
    return None


def _extract_counts(doc_dir: Path) -> dict:
    """Возвращает словарь:
       {doc_id, s0_sections_total, s1_nodes_total, s1_edges_total, nodes_by_type, has_files}
    """
    s0j = _find_first(doc_dir, S0_NAMES)
    s1d = _find_first(doc_dir, ("s1_debug.json",))
    s1j = _find_first(doc_dir, ("s1.json", "s1_graph.json"))
    gj = _find_first(doc_dir, G_NAMES)

    doc_id = doc_dir.name
    s0_sections_total = 0
    nodes_total = 0
    edges_total = 0
    nodes_by_type: dict[str, int] = {}

    # doc_id из s0.json если есть
    if s0j:
        s0 = _safe_load_json(s0j) or {}
        doc_id = s0.get("doc_id") or doc_id
        secs = s0.get("sections") or []
        s0_sections_total = len(secs)

    # приоритетно читаем s1_debug.json (там готовые счётчики)
    if s1d:
        dbg = _safe_load_json(s1d) or {}
        s1counts = (dbg.get("s1_counts") or {})
        nodes_total = int(s1counts.get("nodes_total") or 0)
        edges_total = int(s1counts.get("edges_total") or 0)
        nodes_by_type = dict(s1counts.get("nodes_by_type") or {})
    elif s1j:
        s1 = _safe_load_json(s1j) or {}
        nodes = s1.get("nodes") or []
        edges = s1.get("edges") or []
        nodes_total = len(nodes)
        edges_total = len(edges)
        # посчитаем по типам
        for n in nodes:
            t = n.get("type") or "Unknown"
            nodes_by_type[t] = nodes_by_type.get(t, 0) + 1
    elif gj:
        g = _safe_load_json(gj) or {}
        nodes = g.get("nodes") or []
        edges = g.get("edges") or []
        nodes_total = len(nodes)
        edges_total = len(edges)
        for n in nodes:
            t = n.get("type") or "Unknown"
            nodes_by_type[t] = nodes_by_type.get(t, 0) + 1

    has_files = any([s0j, s1d, s1j, gj])

    return {
        "doc_id": doc_id,
        "s0_sections_total": s0_sections_total,
        "s1_nodes_total": nodes_total,
        "s1_edges_total": edges_total,
        "nodes_by_type": nodes_by_type,
        "has_files": has_files,
        "dir": str(doc_dir),
    }


def _extract_counts_for_doc(doc_id: str) -> dict:
    """
    Ищем:
      - s0.json в /app/data/{doc_id}
      - s1/s2/graph в /app/export/{doc_id}
    Падаем мягко, если чего-то нет.
    """
    data_dir = (DATA_DIR / doc_id)
    export_dir = (EXPORT_DIR / doc_id)

    s0 = None
    for nm in S0_NAMES:
        p = data_dir / nm
        if p.exists(): s0 = _safe_load_json(p); break

    s1 = None
    dbg = None
    for nm in S1_NAMES:
        p = export_dir / nm
        if p.exists():
            if nm == "s1_debug.json":
                dbg = _safe_load_json(p)
            else:
                s1 = _safe_load_json(p)

    s2 = None
    for nm in S2_NAMES:
        p = export_dir / nm
        if p.exists(): s2 = _safe_load_json(p); break

    g = None
    for nm in G_NAMES:
        p = export_dir / nm
        if p.exists(): g = _safe_load_json(p); break

    # секции
    s0_sections = len((s0 or {}).get("sections") or [])

    # узлы/рёбра + по типам
    nodes_total = edges_total = 0
    nodes_by_type = {}

    if dbg and isinstance(dbg.get("s1_counts"), dict):
        c = dbg["s1_counts"]
        nodes_total = int(c.get("nodes_total") or 0)
        edges_total = int(c.get("edges_total") or 0)
        nodes_by_type = dict(c.get("nodes_by_type") or {})
    elif s1 and isinstance(s1, dict):
        nodes = s1.get("nodes") or []
        edges = s1.get("edges") or []
        nodes_total = len(nodes)
        edges_total = len(edges)
        for n in nodes:
            t = n.get("type") or "Unknown"
            nodes_by_type[t] = nodes_by_type.get(t, 0) + 1
    elif g and isinstance(g, dict):
        nodes = g.get("nodes") or []
        edges = g.get("edges") or []
        nodes_total = len(nodes)
        edges_total = len(edges)
        for n in nodes:
            t = n.get("type") or "Unknown"
            nodes_by_type[t] = nodes_by_type.get(t, 0) + 1

    return {
        "doc_id": doc_id,
        "s0_sections": s0_sections,
        "s1_nodes": nodes_total,
        "s1_edges": edges_total,
        "nodes_by_type": nodes_by_type,
        "has_any": any([s0, s1, s2, g]),
    }


@router.get("/report", response_class=HTMLResponse)
def dataset_report(limit: int = 5000):
    """
    Репорт собирается по подкаталогам в /app/export (каждый — doc_id).
    s0.json читаем из /app/data/{doc_id} при наличии.
    """
    roots = [EXPORT_DIR]
    doc_ids = []
    for root in roots:
        if not root.exists(): continue
        for p in root.iterdir():
            if p.is_dir():
                doc_ids.append(p.name)
    doc_ids = doc_ids[:limit]

    rows = [_extract_counts_for_doc(did) for did in doc_ids]
    rows = [r for r in rows if r["has_any"]]

    # totals
    total_docs = len(rows)
    total_sections = sum(r["s0_sections"] for r in rows)
    total_nodes = sum(r["s1_nodes"] for r in rows)
    total_edges = sum(r["s1_edges"] for r in rows)
    # уникальные типы по всем документам
    types_all = {}
    for r in rows:
        for t, cnt in r["nodes_by_type"].items():
            types_all[t] = types_all.get(t, 0) + cnt
    unique_types_count = len(types_all)

    def _fmt_types(d: dict) -> str:
        if not d: return "—"
        return ", ".join(f"{k}:{v}" for k, v in sorted(d.items()))

    html_rows = [
        f"<tr>"
        f"<td><code>{r['doc_id']}</code></td>"
        f"<td style='text-align:right'>{r['s0_sections']}</td>"
        f"<td style='text-align:right'>{r['s1_nodes']}</td>"
        f"<td style='text-align:right'>{r['s1_edges']}</td>"
        f"<td>{_fmt_types(r['nodes_by_type'])}</td>"
        f"</tr>"
        for r in rows
    ]

    html = f"""
    <html><head><title>Dataset Report</title>
    <meta charset="utf-8"/>
    <style>
      :root {{
        --bg:#0b0f1a; --card:#111628; --line:#1c2442;
        --text:#e7e9ee; --muted:#9aa3ae; --head:#cbd3df;
        --pill:#18213e; --pill-b:#2a386c;
      }}
      body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;padding:16px;background:var(--bg);color:var(--text)}}
      .wrap{{max-width:1200px;margin:0 auto}}
      .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}}
      h2{{margin:6px 0 12px}}
      .kpis{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 14px}}
      .pill{{background:var(--pill);border:1px solid var(--pill-b);padding:6px 10px;border-radius:999px}}
      table{{border-collapse:collapse;width:100%}}
      th,td{{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
      th{{background:#111a32;color:var(--head)}}
      code{{font-family:ui-monospace,Consolas,monospace}}
    </style></head>
    <body>
      <div class="wrap">
        <div class="card">
          <h2>Dataset report</h2>
          <div class="kpis">
            <div class="pill"><b>Docs</b>&nbsp;&nbsp;{total_docs}</div>
            <div class="pill"><b>Sections</b>&nbsp;&nbsp;{total_sections}</div>
            <div class="pill"><b>Nodes</b>&nbsp;&nbsp;{total_nodes}</div>
            <div class="pill"><b>Edges</b>&nbsp;&nbsp;{total_edges}</div>
            <div class="pill"><b>Node types</b>&nbsp;&nbsp;{unique_types_count}</div>
          </div>
          <div class="muted" style="color:#9aa3ae;margin:6px 0 12px">
            DATA_DIR: <code>{DATA_DIR}</code> &nbsp;|&nbsp; EXPORT_DIR: <code>{EXPORT_DIR}</code>
          </div>
          <table>
            <thead><tr>
              <th>doc_id</th><th style="width:110px">S0 sections</th><th style="width:110px">S1 nodes</th><th style="width:110px">S1 edges</th><th>node types</th>
            </tr></thead>
            <tbody>{''.join(html_rows) if html_rows else '<tr><td colspan="5">Ничего не найдено в EXPORT_DIR.</td></tr>'}</tbody>
          </table>
        </div>
      </div>
    </body></html>
    """
    return HTMLResponse(html, status_code=200)

# llm_layer.py
# One-shot GraphRefine LLM layer for Singularis
# - Takes noisy S1 graph JSON and (optionally) light S0 context
# - Asks LLM to "clean the logic": normalize/dedup nodes, fix types/polarity,
#   add/drop edges, and return a compact, valid JSON graph.
# - Robust to provider quirks (json_object vs text), truncated outputs, empty completions
# - Safe caching (sqlite, optional); rich logging via stdout
#
# Env knobs (sane defaults):
#   LLM_MODEL=gpt-4o-mini
#   LLM_FALLBACK_MODELS=gpt-5-nano
#   LLM_TIMEOUT=60
#   LLM_TEMPERATURE=            (omit for strict models)
#   LLM_MAX_COMPLETION_TOKENS=1600
#   LLM_FORCE_JSON=1
#   LLM_PREFLIGHT_TOKENS=16
#   LLM_DEBUG=1
#   LLM_LOG_RESP=1
#   LLM_DUMP_RESP=/tmp/llm_resp
#   LLM_CACHE_DB=/mnt/data/llm_cache.sqlite
#   LLM_BUDGET_TOKENS=8000            (prompt budget target)
#   LLM_S1_TEXT_TRUNC=220             (truncate input node text)
#   LLM_OUT_TEXT_TRUNC=180            (requirement for canonical node text)
#
# Public entrypoint:
#   refine_graph(doc_id, s1_graph: dict, s0_sections: list[str] | None) -> dict

from __future__ import annotations
import os, sys, time, json, re, sqlite3, hashlib, pathlib
from typing import Any, Dict, List, Optional, Tuple

# ---------- logging ----------
def _log(ev: str, data: Any):
    if os.environ.get("LLM_DEBUG", "1") != "1":
        return
    try:
        print(f"[LLM] {ev} {data}", file=sys.stdout, flush=True)
    except Exception:
        pass

def _log_llm_response(tag: str, text: str):
    if os.environ.get("LLM_LOG_RESP", "0") != "1":
        return
    head = (text or "")[:400]
    _log("response", {"tag": tag, "chars": len(text or "")})
    _log("response.preview", {"tag": tag, "head400": head})
    dump_dir = os.environ.get("LLM_DUMP_RESP", "").strip()
    if dump_dir:
        try:
            pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(dump_dir, f"{re.sub(r'[^a-zA-Z0-9._-]', '_', tag)}.txt"), "w", encoding="utf-8") as f:
                f.write(text or "")
        except Exception as e:
            _log("dump.fail", str(e))

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

# ---------- tiny token estimator ----------
def _est_tokens_from_texts(texts: List[str]) -> Tuple[int, int]:
    chars = sum(len(t) for t in texts if t)
    # crude ~4 chars/token heuristic
    toks = max(1, int(chars / 4))
    return chars, toks

# ---------- sqlite cache (optional & safe) ----------
class _KV:
    _init_done = False
    _disabled = False
    _db = None

    @classmethod
    def _ensure(cls):
        if cls._disabled or cls._init_done:
            return
        try:
            path = os.environ.get("LLM_CACHE_DB", "/mnt/data/llm_cache.sqlite")
            cls._db = sqlite3.connect(path, timeout=2.0, check_same_thread=False)
            cur = cls._db.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, ts INTEGER)")
            cls._db.commit()
            cls._init_done = True
        except Exception as e:
            _log("cache.init.fail", str(e))
            cls._disabled = True

    @classmethod
    def get(cls, k: str) -> Optional[str]:
        try:
            cls._ensure()
            if cls._disabled: return None
            cur = cls._db.cursor()
            cur.execute("SELECT v FROM kv WHERE k=?", (k,))
            row = cur.fetchone()
            return row[0] if row else None
        except Exception as e:
            _log("cache.get.fail", str(e))
            return None

    @classmethod
    def put(cls, k: str, v: str):
        try:
            cls._ensure()
            if cls._disabled: return
            cur = cls._db.cursor()
            cur.execute("INSERT OR REPLACE INTO kv (k, v, ts) VALUES (?, ?, ?)", (k, v, int(time.time())))
            cls._db.commit()
        except Exception as e:
            _log("cache.put.fail", str(e))

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

def _cache_get(k: str) -> Optional[str]:
    return _KV.get(k)

def _cache_put(k: str, v: str):
    _KV.put(k, v)

# ---------- JSON coercion / repair ----------
_JSON_SIMPLE = re.compile(r"\{.*\}", re.S)

def _repair_truncated_json(js: str) -> str:
    """
    Fix common tail truncations: dangling commas and unbalanced ]/}.
    Not a general-purpose fixer; just balances simple endings.
    """
    s = js.strip()
    # remove trailing comma before final ] or }
    s = re.sub(r",\s*(\]|\})\s*$", r"\1", s)

    open_sq = s.count("["); close_sq = s.count("]")
    open_cu = s.count("{"); close_cu = s.count("}")

    if close_sq < open_sq:
        s += "]" * (open_sq - close_sq)
    if close_cu < open_cu:
        s += "}" * (open_cu - close_cu)

    s = re.sub(r",\s*(\]|\})\s*$", r"\1", s)
    return s

def _coerce_json_object(s: str) -> str:
    """
    Pull out a JSON object from possible Markdown/text:
    - remove ``` fences,
    - take widest {...},
    - fix trailing commas, try to balance brackets,
    - strip BOM and spaces.
    """
    if not s:
        raise ValueError("Empty LLM response")
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = _JSON_SIMPLE.search(s)
    if not m:
        raise ValueError("No JSON object found")
    js = m.group(0).strip()
    js = re.sub(r",(\s*[\]\}])", r"\1", js)
    js = _repair_truncated_json(js)
    js = js.replace("\uFEFF", "").strip()
    return js

def _parse_json_strict(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        fixed = _coerce_json_object(raw)
        return json.loads(fixed)

# ---------- OpenAI client & calling ----------
def _preflight_ok(client, model: str, timeout_s: float, tag: str) -> bool:
    """
    Quick availability check. Treat 'output limit / max_tokens' 400 as OK, since it proves liveness.
    """
    try:
        t0 = time.time()
        pre_tokens = int(os.environ.get("LLM_PREFLIGHT_TOKENS", "16"))
        _ = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ok"}],
            max_completion_tokens=pre_tokens,
            timeout=min(max(5.0, timeout_s / 6.0), 15.0),
        )
        dt = time.time() - t0
        _log("preflight.ok", {"tag": tag, "elapsed_sec": round(dt, 3)})
        return True
    except Exception as e:
        msg = (getattr(e, "message", None) or str(e)).lower()
        _log("preflight.fail", {"tag": tag, "err": str(e)})
        if "max_tokens" in msg or "output limit" in msg:
            return True
        return False

def _call_llm(messages: List[Dict[str, str]], model: str, cache_tag: str) -> str:
    """
    One robust call with:
      - json_object when supported (and disabled for *nano* by default),
      - fallback to no-json mode if empty/unsupported,
      - model fallbacks list,
      - proper caching after model/json flag resolved,
      - parsed vs content handling,
      - detailed logs and usage stats.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai client not installed; pip install openai>=1.0") from e

    timeout_s = _env_float("LLM_TIMEOUT", 60.0)
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=timeout_s,
    )

    fallbacks = [m.strip() for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
    candidates = [model] + [m for m in fallbacks if m != model]

    # static kwargs except model
    kwargs_static: Dict[str, Any] = {"messages": messages}
    t_str = os.environ.get("LLM_TEMPERATURE", "").strip()
    if t_str:
        try:
            kwargs_static["temperature"] = float(t_str)
        except Exception:
            pass
    kwargs_static["max_completion_tokens"] = int(os.environ.get("LLM_MAX_COMPLETION_TOKENS", "1600"))
    force_json = (os.environ.get("LLM_FORCE_JSON", "1") == "1")

    def _do_once(mname: str, use_json: bool) -> str:
        kw = dict(kwargs_static)
        kw["model"] = mname
        if use_json:
            kw["response_format"] = {"type": "json_object"}
        else:
            kw.pop("response_format", None)

        if not _preflight_ok(client, mname, timeout_s, tag=f"{cache_tag}/pre"):
            raise TimeoutError("preflight failed")

        _log("request", {"model": mname, "timeout": timeout_s, "has_temp": "temperature" in kw, "json": use_json})

        payload_for_hash = json.dumps({"endpoint": "chat.completions", **kw}, ensure_ascii=False, sort_keys=True)
        key = _hash(payload_for_hash + "|" + cache_tag)
        hit = _cache_get(key)
        if hit:
            _log("cache hit", cache_tag)
            _log_llm_response(f"{cache_tag}/cache", hit)
            return hit

        t0 = time.time()
        resp = client.chat.completions.create(**kw)
        dt = time.time() - t0
        try:
            usage = getattr(resp, "usage", None)
            usage_info = {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                          "completion_tokens": getattr(usage, "completion_tokens", None),
                          "total_tokens": getattr(usage, "total_tokens", None)} if usage else None
        except Exception:
            usage_info = None
        _log("response.time", {"tag": cache_tag, "elapsed_sec": round(dt, 3), "usage": usage_info})

        choice = resp.choices[0].message
        text = None
        parsed = getattr(choice, "parsed", None)
        if parsed is not None:
            try:
                text = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                text = None
        if not text:
            text = (choice.content or "").strip()

        _log_llm_response(cache_tag, text or "")
        if not text:
            raise RuntimeError("empty_completion")

        _cache_put(key, text)
        return text

    last_err: Optional[Exception] = None
    for mname in candidates:
        try:
            # Avoid forcing JSON on "nano" by default; retry without JSON when needed
            use_json = True
            try:
                return _do_once(mname, use_json=use_json)
            except Exception as e1:
                msg = (getattr(e1, "message", None) or str(e1)).lower()
                if use_json and any(k in msg for k in ["response_format", "json", "empty_completion", "output limit", "max_tokens"]):
                    _log("retry without json", {"model": mname})
                    return _do_once(mname, use_json=False)
                raise
        except Exception as e:
            last_err = e
            _log("model.fail", {"model": mname, "err": str(e)})
            continue
    raise last_err or RuntimeError("all LLM candidates failed")

# ---------- GraphRefine prompt ----------
SYSTEM_PROMPT = """You are a precise graph refiner for scientific papers.
You receive a noisy candidate graph produced by regex/rules (S1).
Your job: (1) validate and normalize nodes, (2) deduplicate near-duplicates,
(3) fix types and polarity, (4) create or prune edges, and (5) return a clean graph
that preserves the paper’s meaning with no hallucinations.

Constraints:
- DO NOT invent facts. Only use provided S1 nodes/edges and optional S0 context.
- If information is insufficient, drop or mark with low confidence; do NOT fill gaps.
- Keep each node text concise ≤ {OUT_TRUNC} characters, but never drop numbers/units.
- Prefer full sentences (one sentence per node).
- Use only these node types: InputFact, Hypothesis, Experiment, Technique, Result, Dataset, Analysis, Conclusion.
- Use only these edge types: uses, produces, supports, refutes.
- Keep JSON strictly valid. No comments, no trailing commas, no extra keys.
- If you merge nodes, keep one with merged provenance array.
- Output minimal JSON, no explanations outside the JSON.
""".replace("{OUT_TRUNC}", os.environ.get("LLM_OUT_TEXT_TRUNC", "180"))

USER_TEMPLATE = """Paper ID: {doc_id}

S1_GRAPH_JSON (compacted):
{S1_JSON}

OPTIONAL_CONTEXT:
- Sections present: {SECTIONS}
- Notes: {NOTES}

Task:
1) DROP clearly spurious nodes and edges (prefer keep if conf ≥ {min_node_conf}; edges conf ≥ {min_edge_conf}).
2) DEDUP nodes that have same meaning. Merge provenance (section, spans, labels).
   Keep strongest type and polarity if consistent; else choose most specific type and set polarity="neutral".
3) NORMALIZE node types to: InputFact, Hypothesis, Experiment, Technique, Result, Dataset, Analysis, Conclusion.
4) CANONICALIZE text: ≤{OUT_TRUNC} chars, single sentence, keep key numbers/units/qualifiers.
   Add a `merge_key` (lowercased, alphanumeric) for duplicates you merged.
5) EDGES:
   - Keep only uses/produces/supports/refutes.
   - Add missing edges based on proximity and semantics:
     Technique→(Experiment|Result): uses
     Experiment→Result: produces
     Result→Hypothesis: supports/refutes (by polarity)
   - Remove edges that contradict node types or meaning.
6) CONFIDENCE:
   - Start from S1 `conf`. If you promote a node/edge (clear evidence), raise up to 0.70–0.85.
   - If only weak evidence, keep 0.45–0.60.
7) POSITIONING HINTS (optional): assign grid `col` by type ordering:
   [0:InputFact, 1:Hypothesis, 2:Experiment, 3:Technique, 4:Result, 5:Dataset, 6:Analysis, 7:Conclusion].
   `row` may be sequential per column (best effort).

Return JSON ONLY with this schema:
{{
  "doc_id": "{doc_id}",
  "nodes": [
    {{
      "id": "<keep original id or a merged canonical id>",
      "type": "InputFact|Hypothesis|Experiment|Technique|Result|Dataset|Analysis|Conclusion",
      "text": "<≤{OUT_TRUNC} chars canonical sentence>",
      "polarity": "positive|negative|neutral",
      "conf": 0.00,
      "merge_key": "<short key; optional>",
      "prov": [
        {{"section": "<S0 section>", "span": [0, 0], "labels": ["..."]}}
      ],
      "data": {{"col": 0, "row": 0}}
    }}
  ],
  "edges": [
    {{
      "from": "<node_id>",
      "to": "<node_id>",
      "type": "uses|produces|supports|refutes",
      "conf": 0.00,
      "prov": {{"hint": "llm", "reason": "<short>"}}
    }}
  ]
}}
""".replace("{OUT_TRUNC}", os.environ.get("LLM_OUT_TEXT_TRUNC", "180"))

# ---------- S1 compaction ----------
_ALLOWED_NODE_TYPES = {"InputFact","Hypothesis","Experiment","Technique","Result","Dataset","Analysis","Conclusion"}
_ALLOWED_EDGE_TYPES = {"uses","produces","supports","refutes"}

def _truncate_text(s: str, lim: int) -> str:
    if not s:
        return ""
    if len(s) <= lim:
        return s
    return (s[:lim-1] + "…")

def _compact_s1_graph(s1: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only necessary fields and truncate node text so the prompt fits budget.
    """
    text_lim = int(os.environ.get("LLM_S1_TEXT_TRUNC", "220"))
    doc_id = s1.get("doc_id") or s1.get("id") or "doc"
    nodes_in = s1.get("nodes", [])
    edges_in = s1.get("edges", [])
    nodes_out = []
    for n in nodes_in:
        nodes_out.append({
            "id": n.get("id"),
            "type": n.get("type"),
            "text": _truncate_text(n.get("text","").strip(), text_lim),
            "polarity": n.get("polarity","neutral"),
            "conf": float(n.get("conf", 0.0)),
            "label": n.get("label", ""),
            "prov": n.get("prov", {}),
            "data": n.get("data", {}),
        })
    edges_out = []
    for e in edges_in:
        edges_out.append({
            "from": e.get("from"),
            "to": e.get("to"),
            "type": e.get("type"),
            "conf": float(e.get("conf", 0.0)),
            "prov": e.get("prov", {}),
        })
    return {"doc_id": doc_id, "nodes": nodes_out, "edges": edges_out}

# ---------- Public: one-shot refine ----------
def refine_graph(doc_id: str, s1_graph: Dict[str, Any], s0_sections: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Build a single LLM request to refine S1 graph into a clean S2/Sfinal graph.
    Returns dict with keys: doc_id, nodes[], edges[].
    This function is robust and never raises on LLM quirks; on fatal errors it returns original S1.
    """
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    # 1) compact S1 and compute sizes
    s1c = _compact_s1_graph(s1_graph)
    s1c_json = json.dumps(s1c, ensure_ascii=False)
    sects = ",".join(s0_sections or [])
    notes = "—"
    min_node_conf = "0.40"
    min_edge_conf = "0.55"

    # token/char budgeting log
    n_texts = [n["text"] for n in s1c.get("nodes", [])]
    e_descs = [f'{e.get("from","")}->{e.get("to","")}:{e.get("type","")}' for e in s1c.get("edges", [])]
    node_chars, node_toks = _est_tokens_from_texts(n_texts)
    edge_chars, edge_toks = _est_tokens_from_texts(e_descs)
    budget = int(os.environ.get("LLM_BUDGET_TOKENS", "8000"))
    _log("pre-chunk sizes", {"nodes": len(n_texts), "edges": len(e_descs),
                             "node_chars": node_chars, "node_est_tokens": node_toks,
                             "edge_chars": edge_chars, "edge_est_tokens": edge_toks,
                             "budget_tokens": budget})

    # 2) build messages
    user_text = USER_TEMPLATE.format(
        doc_id=doc_id,
        S1_JSON=s1c_json,
        SECTIONS=sects,
        NOTES=notes,
        min_node_conf=min_node_conf,
        min_edge_conf=min_edge_conf
    )
    _log("payload_size", {"chars": len(user_text)})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]

    # 3) call LLM
    raw = None
    try:
        raw = _call_llm(messages, model=model, cache_tag=f"{doc_id}/graphrefine")
        data = _parse_json_strict(raw)

        # light sanity checks
        out = {
            "doc_id": data.get("doc_id", doc_id),
            "nodes": [],
            "edges": []
        }
        # keep only allowed node/edge fields & types
        for n in data.get("nodes", []):
            t = (n.get("type") or "").strip()
            if t and t not in _ALLOWED_NODE_TYPES:
                continue
            out["nodes"].append({
                "id": n.get("id"),
                "type": t or "Result",
                "text": (n.get("text") or "").strip(),
                "polarity": (n.get("polarity") or "neutral"),
                "conf": float(n.get("conf", 0.0)),
                "merge_key": n.get("merge_key", None),
                "prov": n.get("prov", []),
                "data": n.get("data", {}),
            })
        for e in data.get("edges", []):
            et = (e.get("type") or "").strip()
            if et not in _ALLOWED_EDGE_TYPES:
                continue
            out["edges"].append({
                "from": e.get("from"),
                "to": e.get("to"),
                "type": et,
                "conf": float(e.get("conf", 0.0)),
                "prov": e.get("prov", {"hint": "llm"}),
            })

        # final trims: drop empties
        out["nodes"] = [n for n in out["nodes"] if n.get("id") and n.get("text")]
        out["edges"] = [e for e in out["edges"] if e.get("from") and e.get("to") and e.get("type")]
        return out

    except Exception as e:
        _log("graphrefine.fail", {"err": str(e)})
        # fallback: return original S1 (so pipeline keeps running)
        return {
            "doc_id": doc_id,
            "nodes": s1_graph.get("nodes", []),
            "edges": s1_graph.get("edges", []),
            "prov": {"hint": "fallback_s1"}
        }

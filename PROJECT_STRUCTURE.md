**Project Structure**

- Purpose: End-to-end pipeline to parse PDFs and extract a scientific knowledge graph, with a web UI and FastAPI gateway. Heuristic rules live under `rules/` (S1), data parsing in `pipeline/`, and the UI stack in `ui/`.

**Top-Level Layout**

- `pipeline/` — Python 3.11 processing pipeline
  - `pipeline/pipeline/s0.py` — document ingest (PDF → text/sections)
  - `pipeline/pipeline/s1.py` — heuristic extraction (rules/regex)
  - `pipeline/pipeline/s2.py` — linking, post-processing
  - `pipeline/pipeline/worker.py` — queue worker entrypoint
  - `pipeline/pipeline/themes_router.py` — themes (FOS) routing logic
  - `pipeline/pipeline/parsers/` — parsers (`pdf_parser.py`, `latex_parser.py`)
  - `pipeline/requirements.txt`, `pipeline/Dockerfile`

- `ui/` — UI stack
  - `ui/server/` — FastAPI gateway (queues worker jobs)
    - `ui/server/app/main.py` — API endpoints (parse/extract/graph)
    - `ui/server/Dockerfile`, `ui/server/requirements.txt`
  - `ui/web/` — Vite + React client (Cytoscape graph)
    - `ui/web/src/App.jsx`, `ui/web/src/main.jsx`, `ui/web/Dockerfile`

- `rules/` — S1 rules and theme taxonomy (FOS-2021)
  - `rules/common.yaml` — global S1 config (section weights, boosts, link patterns)
  - `rules/themes/` — 6 fields and 42 subfields
    - `<field>/lexicon.yaml`, `<field>/rules.yaml`, `<field>/triggers.yaml`
    - `<field>/<subfield>/lexicon.yaml|rules.yaml|triggers.yaml`
    - `_schemas/` — YAML schemas for lints (`lexicon.schema.yaml`, `rules.schema.yaml`, `triggers.schema.yaml`)
    - `shared-lexicon.yaml` — shared synonyms/abbr/hedges
  - `rules/tools/` — lints and helpers (`rules_linter.py`)

- Data & Working Dirs
  - `dataset/` — sample corpora (PDFs)
  - `data/`, `export/`, `workdir/` — intermediate artifacts, cache, exports

- Utilities
  - `scripts/` — maintenance tooling (`fos_generate.py`, YAML migrations)
  - `tools/` — reporting utilities (`report.py`)

- Orchestration
  - `docker-compose.yml` — multi-container stack (API, worker, grobid, redis, neo4j)
  - `makefile` — tasks: `make run`, `make rebuild-all`, `make rebuild-app`
  - `.env`, `.env.example` — environment variables

**Rules (S1) Overview**

- `triggers.yaml` — route PDFs to fields/subfields
  - Keys: `id`, `version`, `threshold`, `must[]`, `should[[token,weight]]`, `negative[[token,weight]]`
- `lexicon.yaml` — abbreviations, synonyms, hedging extras
  - Keys: `abbr[[short,long]]`, `synonyms[[a,b]]`, `hedging_extra[]`
- `rules.yaml` — regex-based elements for node types
  - Element keys: `id`, `type` (Result|Hypothesis|Experiment|Technique|Dataset|Analysis|Conclusion|Input Fact), `weight`, `sections[]`, `pattern`, `negatives[]`, `captures[]`

**Development Quickstart**

- Run full stack: `make run` then open `http://localhost:{3000,8000,8070,7474}`
- UI-only dev: `cd ui/web && npm install && npm run dev` (expects API running)
- Worker debug: `docker compose exec worker python -m pipeline.worker`
- Local Python (no Docker): create Python 3.11 venv in `/.venv` and install `pipeline/requirements.txt`

**Services & Ports**

- `grobid` — 8070 (REST), 8071 (Admin)
- `api` — 8000 (FastAPI)
- `web` — 3000 (Vite/React dev)
- `neo4j` — 7474 (HTTP UI), 7687 (Bolt)
- `redis` — 6379

**API Endpoints**

- `GET /health` — service status
- `POST /parse` — upload PDF, returns `doc_id` and schedules pipeline (S0→S2)
- `POST /extract?doc_id=...&theme=auto|<theme>` — enqueue extraction by `doc_id`
- `GET /status/{doc_id}` — pipeline status and timings
- `GET /preview/{doc_id}/{artifact}` — preview `s0|s1|s2|graph` artifacts
- `GET /graph/{doc_id}` — graph JSON
- `GET /themes` — available themes (FOS registry)

**Data Flow & Artifacts**

- Input: uploaded PDF stored at `data/<doc_id>/input.pdf`
- S0 output: `data/<doc_id>/s0.json` (sections, text)
- S1/S2 debug: `export/<doc_id>/{s1_debug.json,s2_debug.json}`
- Graph: `export/<doc_id>/graph.json`
- Routing: `rules/themes/**/triggers.yaml` selects field/subfield; S1 applies `rules.yaml`; S2 links nodes/edges using patterns from `rules/common.yaml`.

**Key Environment Variables**

- `GROBID_URL` (API, worker) — Grobid base URL
- `REDIS_URL` (API, worker) — Redis connection
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — Neo4j access
- `S1_RULES_PATH` — path to `rules/common.yaml`
- `RULES_BASE_DIR`, `THEMES_DIR` (API) — theme registry discovery

**Directory Snapshot (abridged)**

```
.
├─ pipeline/
│  ├─ pipeline/
│  │  ├─ s0.py  s1.py  s2.py  worker.py  themes_router.py
│  │  └─ parsers/ (pdf_parser.py, latex_parser.py)
│  ├─ requirements.txt
│  └─ Dockerfile
├─ ui/
│  ├─ server/ (FastAPI)
│  └─ web/ (Vite+React)
├─ rules/
│  ├─ common.yaml
│  ├─ themes/
│  │  ├─ 1-natural-sciences/ ... 6-humanities-and-the-arts/
│  │  └─ _schemas/ (lexicon|rules|triggers.schema.yaml)
│  └─ tools/ (rules_linter.py)
├─ dataset/  data/  export/  workdir/
├─ docker-compose.yml  makefile  AGENTS.md  Project_Description.md
```

For goals and context see `Project_Description.md`. For coding and workflow conventions see `AGENTS.md`.

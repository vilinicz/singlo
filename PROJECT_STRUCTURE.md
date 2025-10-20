# Project Structure

This document reflects the current repository layout, services, and key flows. It replaces an older corrupted version.

## Stack Overview
- grobid - PDF to TEI service used by S0
- api - FastAPI gateway (queues jobs, exposes preview/graph endpoints)
- worker - executes S0 -> S1 -> S2 pipeline and writes outputs
- web - React + Vite client with Cytoscape graph view
- neo4j - optional graph database (mounted export/ for import)
- redis - queue and status store for jobs (RQ)

## Repository Layout
- pipeline/ - Python 3.11 processing pipeline
  - pipeline/pipeline/s0_grobid.py - GROBID fulltext to s0.json
  - pipeline/pipeline/s1.py - spaCy-based node/edge extraction using theme patterns
  - pipeline/pipeline/s2.py - graph assembly and debug artifacts
  - pipeline/pipeline/worker.py - run_pipeline orchestrator, Redis job status
  - pipeline/pipeline/themes_router.py - theme registry preload (FOS routing)
  - pipeline/pipeline/spacy_loader.py - spaCy model and pattern loading
  - pipeline/pipeline/parsers/{pdf_parser.py,latex_parser.py} - parsers (auxiliary)
  - pipeline/pipeline/{s0_legacy.py,s1_legacy.py} - legacy stages (kept for reference)
  - pipeline/requirements.txt, pipeline/Dockerfile

- ui/server/ - FastAPI API
  - ui/server/app/main.py - routes: /parse, /extract, /status/{doc_id}, /preview/{doc_id}/{artifact}, /graph/{doc_id}, /themes, /health
  - ui/server/app/testbench.py - misc test/demo routes
  - ui/server/{Dockerfile,requirements.txt}

- ui/web/ - Vite + React client (Cytoscape)
  - ui/web/src/{App.jsx,main.jsx}; served on :3000 in Docker; npm run dev uses :5173
  - ui/web/Dockerfile, ui/web/package.json

- rules/ - data-driven themes (current)
  - rules/themes/<theme>/{lexicon.json,patterns/{matcher.json,depmatcher.json}}
  - rules/themes/common/{lexicon.json,patterns/...}

- legacy_rules/ - YAML rule set (schemas + legacy themes)
  - legacy_rules/themes/<field>/{lexicon.yaml,rules.yaml,triggers.yaml}
  - legacy_rules/themes/_schemas/{lexicon.schema.yaml,rules.schema.yaml,triggers.schema.yaml}
  - legacy_rules/themes/shared-lexicon.yaml

- Data and artifacts
  - dataset/ - example/input PDFs (read-only samples)
  - data/<doc_id>/ - inputs and early artifacts (input.pdf, s0.json)
  - export/<doc_id>/ - outputs (graph.json, s1_debug.json, s2_debug.json)
  - workdir/ - scratch directory used by tools/pipeline

- Utilities
  - scripts/ - maintenance and migration scripts (FOS generation, YAML migrations)
  - tools/ - small utilities (e.g., tools/report.py)

- Top-level
  - docker-compose.yml - services: api, worker, grobid, redis, neo4j, web
  - makefile - make run, make rebuild-all, make rebuild-app
  - .env, .env.example - container configuration (if present)
  - AGENTS.md, Project_Description.md

## Build & Run
- Full stack: make run -> http://localhost:{3000,8000,8070,7474}
- Rebuild all (drop volumes): make rebuild-all
- Rebuild app-only (worker + web): make rebuild-app
- UI dev only: cd ui/web && npm install && npm run dev (serves on :5173; API still required)
- Worker debug: docker compose exec worker python -m pipeline.worker
- Local single-file runs: use Python 3.11 venv at /.venv and install pipeline/requirements.txt

## Ports
- grobid 8070 (REST), 8071 (Admin)
- api 8000 (FastAPI)
- web 3000 (Vite/React Docker), 5173 (Vite dev)
- neo4j 7474 (HTTP UI), 7687 (Bolt)
- redis 6379

## API Endpoints (ui/server/app/main.py)
- GET /health - basic liveness
- POST /parse - upload PDF -> stores data/<doc_id>/input.pdf, queues pipeline, returns doc_id
- POST /extract?doc_id=...&theme=auto|<name>[,<name>] - queue pipeline for an existing doc_id
- GET /status/{doc_id} - current job status and stage timings
- GET /preview/{doc_id}/{artifact} - preview s0|s1|s2|graph files
- GET /graph/{doc_id} - full graph JSON
- GET /themes - available themes (preloaded registry)

## Data Flow & Artifacts
- Input: data/<doc_id>/input.pdf
- S0 (GROBID): data/<doc_id>/s0.json
- S1/S2: export/<doc_id>/{s1_debug.json,s2_debug.json}
- Graph: export/<doc_id>/graph.json
- Rules and themes influence: rules/themes/** (current) and legacy_rules/themes/** (legacy)

## Environment Variables
- GROBID_URL - Grobid endpoint (api, worker)
- REDIS_URL - Redis connection (api, worker)
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD - Neo4j (optional)
- RULES_BASE_DIR, THEMES_DIR - rules root and themes directory used by the pipeline
- S1_RULES_PATH - legacy YAML entry (API defaults to legacy_rules/common.yaml)

## Runbook (typical flow)
1) POST /parse with a PDF; receive doc_id and queued status
2) Poll GET /status/{doc_id} until state=done
3) Inspect GET /preview/{doc_id}/s1 or GET /graph/{doc_id}
4) Re-run with explicit theme via POST /extract?doc_id=...&theme=biomed (or multiple biomed,physics)

## Quick Tree
`
.
|-- pipeline/
|   |-- pipeline/ (s0_grobid.py, s1.py, s2.py, worker.py, themes_router.py, spacy_loader.py, parsers/)
|   |-- requirements.txt
|   -- Dockerfile
|-- ui/
|   |-- server/ (FastAPI)
|   -- web/ (Vite + React)
|-- rules/ (JSON themes: common + specific)
|-- legacy_rules/ (YAML themes + schemas)
|-- dataset/  data/  export/  workdir/
-- docker-compose.yml  makefile  AGENTS.md  Project_Description.md
`

> For project goals and high-level context, see Project_Description.md. Operational notes for agents live in AGENTS.md.

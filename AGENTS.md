# Repository Guidelines

## Project Structure & Module Organization
- `pipeline/` hosts the Python 3.11 processing pipeline; stage modules live in `pipeline/pipeline/{s0,s1,s2}.py`.
- `ui/` contains the UI stack: `ui/web/` is a Vite + React client (Cytoscape graph view) and `ui/server/` is a FastAPI gateway that queues worker jobs.
- Root-level `docker-compose.yml`, service Dockerfiles, and `makefile` define the multi-container setup (API, worker, grobid, redis, neo4j).

## Build, Test, and Development Commands
- `make run` builds and starts the full stack; hit http://localhost:{3000,8000,8070,7474} once containers report healthy.
- `make rebuild-all` tears down volumes and rebuilds everything; use after dependency updates.
- `make rebuild-app` rebuilds only the worker and web containers for quicker iteration.
- UI-only work: `cd ui/web && npm install && npm run dev` to serve the React client on port 5173; it still expects the API stack to be running.
- Worker debugging: `docker compose exec worker python -m pipeline.worker` restarts the worker in the running container.
- to run single files locally without Docker, set up a Python 3.11 virtualenv installed in root folder /.venv

## Coding Style & Naming Conventions
- Python modules follow PEP 8 (4-space indent, snake_case module/function names); keep pipeline stages pure functions where possible.
- Prefer pydantic models for structured payloads and annotate functions with types to keep queue payloads explicit.

## Testing Guidelines
- Record manual QA steps for graph interactions in PR descriptions until automated checks are in place.

## Project Goal
- read in Project_Description.md
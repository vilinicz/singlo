# Singularis 
## A deterministic knowledge-graph extractor for scientific PDFs

### Overview

Singularis turns scientific PDFs into graphs of atomic research elements—Input Fact, Hypothesis, Experiment, Technique, Result, Dataset, Analysis, Conclusion—linked with typed edges. The goal is to reach 80–90% of an LLM read-out’s quality at 10–100× lower cost and latency; LLMs are used sparingly (refine/QA/auto-rule suggestion), while the core extraction is deterministic and cheap.

### Architecture (services & code map)

* grobid — PDF→TEI for S0
* api — FastAPI gateway (queues, preview/graph endpoints)
* worker — executes the S0→S1→S2 pipeline and writes artifacts
* web — React + Cytoscape viewer
* redis — RQ queue + job status store. 

#### Repository highlights:

* `pipeline/pipeline/{s0_grobid.py, s1.py, s2.py, worker.py,…}` — stages & orchestration

* `ui/` — web client

* `rules/` — data-driven themes (spaCy matcher/dep-matcher + lexicon)

* `legacy_rules/` — historical YAML rule set (kept for reference). 

### Quick Start

    Spin up the full stack with Docker, run the pipeline on a PDF, and fetch the graph.

#### Prereqs

* Docker + Docker Compose
* A PDF to test with

1. Create a file `.env` based on the `.env.example` file (requires no changes)
2. Launch the stack:

        `docker compose up -d`

        # Services:
        # - GROBID on 8070/8071
        # - API on 8000
        # - Web UI on 3000 
        # - Redis on 6379


3. Open a web page: http://localhost:3000/
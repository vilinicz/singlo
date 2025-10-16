**Структура Проекта**

- Назначение: конвейер от PDF к графу знаний (S0→S1→S2) с FastAPI‑шлюзом и веб‑клиентом. Правила (регексы/лексикон/триггеры) живут в `rules/`, парсинг и бизнес‑логика — в `pipeline/`, UI — в `ui/`.

**Верхнеуровневые Каталоги**

- `pipeline/` — Python 3.11 pipeline
  - `pipeline/pipeline/s0.py` — разбор PDF в текст и секции
  - `pipeline/pipeline/s1.py` — извлечение сущностей по правилам (regex/heuristics)
  - `pipeline/pipeline/s2.py` — линковка узлов/рёбер, пост‑обработка
  - `pipeline/pipeline/worker.py` — entrypoint воркера очереди
  - `pipeline/pipeline/themes_router.py` — маршрутизация по темам (FOS)
  - `pipeline/pipeline/parsers/` — `pdf_parser.py`, `latex_parser.py`, утилиты
  - `pipeline/requirements.txt`, `pipeline/Dockerfile`

- `ui/` — UI‑стек
  - `ui/server/` — FastAPI (постановка задач, предпросмотр артефактов)
    - `ui/server/app/main.py` — эндпоинты: parse/extract/status/preview/graph/themes
    - `ui/server/Dockerfile`, `ui/server/requirements.txt`
  - `ui/web/` — Vite + React (Cytoscape‑граф)
    - `ui/web/src/App.jsx`, `ui/web/src/main.jsx`, `ui/web/Dockerfile`

- `rules/` — конфигурация S1 и таксономия тем (FOS‑2021)
  - `rules/common.yaml` — глобальные веса секций, бусты, шаблоны связей
  - `rules/themes/` — 6 направлений, 42 поднаправления
    - `<field>/lexicon.yaml`, `<field>/rules.yaml`, `<field>/triggers.yaml`
    - `<field>/<subfield>/{lexicon,rules,triggers}.yaml`
    - `_schemas/` — схемы YAML (`lexicon.schema.yaml`, `rules.schema.yaml`, `triggers.schema.yaml`)
    - `shared-lexicon.yaml` — общий лексикон (abbr/synonyms/hedges)
  - `rules/tools/` — инструменты (`rules_linter.py`)

- Данные и рабочие каталоги
  - `dataset/` — примеры корпусов (PDF)
  - `data/` — входные и результаты S0 (`data/<doc_id>/input.pdf`, `s0.json`)
  - `export/` — отладка/итоги: `graph.json`, `s1_debug.json`, `s2_debug.json`
  - `workdir/` — временные файлы

- Утилиты
  - `scripts/` — генерация FOS, миграции YAML, фиксы (`fos_generate.py`, `migrate_lexicon_shape.py`, `wrap_rules_elements.py`, ...)
  - `tools/` — отчёты (`report.py`)

- Оркестрация
  - `docker-compose.yml` — стек: API, worker, grobid, redis, neo4j
  - `makefile` — `make run`, `make rebuild-all`, `make rebuild-app`
  - `.env`, `.env.example` — переменные окружения

**Правила (S1)**

- `triggers.yaml` — роутинг PDF в тему/подтему
  - Ключи: `id`, `version`, `threshold`, `must[]`, `should[[token,weight]]`, `negative[[token,weight]]`
- `lexicon.yaml` — сокращения, синонимы, hedging
  - Ключи: `abbr[[short,long]]`, `synonyms[[a,b]]`, `hedging_extra[]`
- `rules.yaml` — элементы графа (regex)
  - Ключи: `id`, `type` (Result|Hypothesis|Experiment|Technique|Dataset|Analysis|Conclusion|Input Fact), `weight`, `sections[]`, `pattern`, `negatives[]`, `captures[]`

**Быстрый Старт**

- Полный стек: `make run` → открыть `http://localhost:{3000,8000,8070,7474}`
- Только UI: `cd ui/web && npm install && npm run dev` (API должен работать)
- Отладка воркера: `docker compose exec worker python -m pipeline.worker`
- Локально (без Docker): Python 3.11 venv в `/.venv`, установить `pipeline/requirements.txt`

**Сервисы и Порты**

- `grobid` — 8070 (REST), 8071 (Admin)
- `api` — 8000 (FastAPI)
- `web` — 3000 (Vite/React)
- `neo4j` — 7474 (HTTP UI), 7687 (Bolt)
- `redis` — 6379

**API (основные эндпоинты)**

- `GET /health` — проверка статуса
- `POST /parse` — загрузка PDF, создание `doc_id`, постановка S0→S2
- `POST /extract?doc_id=...&theme=auto|<theme>` — явный запуск извлечения
- `GET /status/{doc_id}` — состояние пайплайна, тайминги
- `GET /preview/{doc_id}/{artifact}` — предпросмотр `s0|s1|s2|graph`
- `GET /graph/{doc_id}` — итоговый граф JSON
- `GET /themes` — список тем (FOS‑реестр)

**Поток Данных и Артефакты**

- Input: `data/<doc_id>/input.pdf`
- S0: `data/<doc_id>/s0.json`
- S1/S2: `export/<doc_id>/s1_debug.json`, `export/<doc_id>/s2_debug.json`
- Graph: `export/<doc_id>/graph.json`
- Роутинг: `rules/themes/**/triggers.yaml` → S1 по `rules.yaml` → S2 связи по `rules/common.yaml`

**Переменные Окружения**

- `GROBID_URL` — адрес Grobid (API, worker)
- `REDIS_URL` — Redis (API, worker)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — доступ к Neo4j
- `S1_RULES_PATH` — путь к `rules/common.yaml`
- `RULES_BASE_DIR`, `THEMES_DIR` (API) — автопоиск тем

**Runbook (типовой сценарий)**

- `POST /parse` → получить `doc_id`
- `GET /status/{doc_id}` — следить за прогрессом
- `GET /preview/{doc_id}/s1` и `.../graph` — проверка артефактов
- Открыть веб‑клиент и загрузить граф по `doc_id`

**Линтер Правил**

- `python rules/tools/rules_linter.py --json rules_lint_report.json` — проверки YAML/regex
- Схемы: `rules/themes/_schemas/*.yaml`

**Снимок Дерева (кратко)**

```
.
├─ pipeline/
│  ├─ pipeline/ (s0.py, s1.py, s2.py, worker.py, themes_router.py, parsers/)
│  ├─ requirements.txt
│  └─ Dockerfile
├─ ui/
│  ├─ server/ (FastAPI)
│  └─ web/ (Vite+React)
├─ rules/
│  ├─ common.yaml
│  ├─ themes/ (6 полей, 42 подтемы, _schemas/, shared-lexicon.yaml)
│  └─ tools/ (rules_linter.py)
├─ dataset/  data/  export/  workdir/
├─ docker-compose.yml  makefile  AGENTS.md  Project_Description.md
```

Для целей и контекста см. `Project_Description.md`. Для конвенций по коду и запуску — `AGENTS.md`.


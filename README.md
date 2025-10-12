 

* grobid (S0-разметка секций/библиографии),
* api (FastAPI: эндпоинты /parse, /extract, /graph),
* worker (S1/S2 пайплайн + очереди),
* web (React UI с Cytoscape),
* neo4j (граф для демо),
* redis (очередь задач).

Пояснения:

    api отдаёт эндпоинты: /parse (кинуть PDF → S0), /extract (→ S1/S2), /graph/{doc_id}, /neo4j/import.
    worker можно заменить на Celery/RQ/Arq — очередь в Redis уже готова.
    grobid healthcheck ждёт полной инициализации сервера перед запуском API.
    neo4j включает APOC, чтобы было удобно импортировать NDJSON/CSV.
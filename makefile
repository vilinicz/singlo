run:
	docker compose up -d --build

# UI: http://localhost:3000
# API: http://localhost:8000/docs
# GROBID: http://localhost:8070
# Neo4j: http://localhost:7474  (логин: neo4j / testtest)

rebuild-all:
	docker compose down -v
	docker compose up -d --build

rebuild-app:
	docker compose build worker web
	docker compose up -d worker web
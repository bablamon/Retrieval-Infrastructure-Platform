.PHONY: help build up down logs init-db test lint fmt clean

help:
	@echo "Targets:"
	@echo "  build     Build the api + worker images"
	@echo "  up        Start the full stack (detached)"
	@echo "  down      Stop the stack"
	@echo "  logs      Tail api + worker logs"
	@echo "  init-db   Create Postgres tables"
	@echo "  test      Run the test suite (in the api container)"
	@echo "  lint      Run ruff lint"
	@echo "  fmt       Auto-format with ruff"
	@echo "  clean     Stop the stack and remove volumes (wipes data)"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f api worker

init-db:
	docker compose exec api python scripts/init_db.py

test:
	docker compose exec api pytest -q

lint:
	ruff check app tests

fmt:
	ruff check --fix app tests

clean:
	docker compose down -v

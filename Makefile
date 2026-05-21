.PHONY: setup lint format test docs report reproduce
.PHONY: docker-setup docker-lint docker-test docker-report docker-reproduce docker-check

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check . --fix
	uv run ruff format .

test:
	uv run pytest

docs:
	uv run sphinx-build -b html docs docs/_build/html

report:
	uv run quarto render report/analysis.qmd

reproduce: lint test docs report

docker-setup:
	docker compose build
	docker compose run --rm dev uv sync --frozen
	docker compose run --rm dev make reproduce

docker-lint:
	docker compose run --rm dev make lint

docker-test:
	docker compose run --rm dev make test

docker-report:
	docker compose run --rm dev make report

docker-reproduce:
	docker compose run --rm dev make reproduce

docker-check:
	docker compose run --rm dev make reproduce
	docker compose build
	docker compose run --rm analysis

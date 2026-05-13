.PHONY: setup lint format test docs report reproduce

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

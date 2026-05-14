# Reproducible Financial Time Series Analysis

This repository is a reproducible Python rewrite of the original R project
[Financial Time Series Analysis and Forecasting GARCH models](https://github.com/DavidAlexanderMoe/Financial-Time-Series-Analysis-and-Forecasting-GARCH-models).

## Group Members

- Sati Ter-Harutyunyan
- Anna Perekhodko
- Markiian Strohyi
- Mykola Bolokh

## Project Structure

```text
data/raw/              Source dataset used by the analysis
data/processed/        Generated intermediate data
src/                   Python modules for the reproducible workflow
tests/                 Automated tests
report/                Quarto report
docs/                  Sphinx documentation
outputs/               Generated reports, plots, and other outputs
```

## Local Setup

Install dependencies with uv:

```bash
uv sync
```

Quarto is required for rendering the report. On macOS it can be installed with:

```bash
brew install --cask quarto
```

## Common Commands

```bash
make lint       # Run Ruff lint and format checks
make test       # Run pytest
make docs       # Build Sphinx HTML documentation
make report     # Render the Quarto HTML report
make reproduce  # Run lint, tests, docs, and report rendering
```

Regenerate the processed ATVI dataset only when intentionally updating data:

```bash
uv run python -m src.data
```

## Docker

The project has two Docker Compose services:

- `dev` mounts your local project files into the container and is intended for
  day-to-day checks while developing.
- `analysis` uses the code copied into the Docker image and is intended as the
  final reproducibility runner before a pull request or presentation.

Initial Docker setup:

```bash
docker compose build
docker compose run --rm dev uv sync --frozen
docker compose run --rm dev make reproduce
```

Daily Docker checks:

```bash
docker compose run --rm dev make lint
docker compose run --rm dev make test
docker compose run --rm dev make report
```

Before opening a pull request for code, data, report, dependency, or pipeline
changes:

```bash
docker compose run --rm dev make reproduce
docker compose build
docker compose run --rm analysis
```

Rendered outputs are written to `outputs/`.

## Collaboration Workflow

All work should be done on feature branches and merged into `main` through pull
requests. Do not push directly to `main`.

Recommended workflow:

```bash
git checkout main
git pull --ff-only
git checkout -b feature/short-description
```

Edit files locally in VS Code.

Run quick checks while working:

```bash
docker compose run --rm dev make lint
docker compose run --rm dev make test
```

If the task changes reports, plots, data processing, dependencies, Docker, or
pipeline logic, run final Docker checks before committing:

```bash
docker compose run --rm dev make reproduce
docker compose build
docker compose run --rm analysis
```

Then inspect and stage only the files related to your task:

```bash
git status
git diff
git add src/
git add tests/
git add report/analysis.qmd
```

Commit with a short subject and a wrapped body:

```bash
git commit \
  -m "feat: add diagnostics" \
  -m "Add plotting utilities for model volatility diagnostics." \
  -m "Save figures to outputs/figures for reuse in the Quarto report."
```

Check the branch and push:

```bash
git status --short --branch
git log --oneline origin/main..HEAD
git push -u origin feature/short-description
```

Each pull request should explain what changed and how it was verified.

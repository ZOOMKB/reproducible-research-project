# Reproducible Financial Time Series Analysis

This repository is a reproducible Python rewrite of the original R project
[Financial Time Series Analysis and Forecasting GARCH models](https://github.com/DavidAlexanderMoe/Financial-Time-Series-Analysis-and-Forecasting-GARCH-models).

The goal of this course project is not to improve the econometric methodology.
The goal is to make the research workflow reproducible, documented, automated,
and suitable for collaborative development through GitHub pull requests.

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

## Docker

Build and run the reproducible environment:

```bash
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
git pull
git checkout -b feature/short-description

# Make changes, then verify them
make lint
make test

git add .
git commit -m "type: describe the change"
git push -u origin feature/short-description
```

Each pull request should explain what changed and how it was verified.

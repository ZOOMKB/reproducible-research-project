# Reproducible Financial Time Series Analysis

This repository is a reproducible Python rewrite of the original R project
[Financial Time Series Analysis and Forecasting GARCH models](https://github.com/DavidAlexanderMoe/Financial-Time-Series-Analysis-and-Forecasting-GARCH-models).

The project focuses on reproducibility. The workflow separates reusable Python
modules, automated checks, HTML documentation, a Quarto report, and a Docker
environment that can reproduce the report from a clean container.

## Group Members

- Sati Ter-Harutyunyan
- Anna Perekhodko
- Markiian Strohyi
- Mykola Bolokh

## Project Structure

```text
data/raw/              Original ATVI source dataset
data/processed/        Processed dataset used by the analysis
src/                   Python package with data processing, EDA, ARMA, GARCH,
                       forecasting, diagnostics, and evaluation modules
tests/                 Lightweight automated tests
report/                Quarto report source
docs/                  Sphinx documentation source
outputs/               Generated reports, figures, and documentation outputs

Makefile               Reproducible command interface for checks and reports
Dockerfile             Docker image definition for the reproducible environment
docker-compose.yml     Docker services for development and final verification
pyproject.toml         Python project metadata, dependencies, and tool settings
uv.lock                Locked Python dependency versions
.pre-commit-config.yaml
                       Pre-commit hooks for formatting and code quality
README.md              Project overview and reproduction instructions
```

## Reproduce From Docker Hub

The public Docker image is:

```bash
mykolanocap/reproducible-research-project:latest
```

To generate the report from the Docker image:

```bash
docker pull mykolanocap/reproducible-research-project:latest
mkdir -p outputs
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  mykolanocap/reproducible-research-project:latest
```

On Windows PowerShell, use:

```powershell
docker pull mykolanocap/reproducible-research-project:latest
mkdir outputs
docker run --rm `
  -v "${PWD}/outputs:/app/outputs" `
  mykolanocap/reproducible-research-project:latest
```

The rendered report is written to:

```text
outputs/report/analysis.html
```

## What The Container Runs

The Docker image runs:

```bash
make reproduce
```

This command executes the full reproducibility workflow:

```text
lint -> tests -> Sphinx documentation -> Quarto report
```

The Quarto report is the main project artifact. It imports the Python modules in
`src/`, regenerates figures and tables, and writes a self-contained HTML report
to `outputs/report/analysis.html`.

## Local Reproduction

For local development without Docker, install dependencies with uv:

```bash
uv sync
```

Quarto is required for local report rendering. On macOS:

```bash
brew install --cask quarto
```

Run the full local workflow:

```bash
make reproduce
```

Useful local commands:

```bash
make lint       # Run Ruff lint and format checks
make test       # Run pytest
make docs       # Build Sphinx HTML documentation
make report     # Render the Quarto HTML report
```

Regenerate the processed ATVI dataset only when intentionally updating data:

```bash
uv run python -m src.data
```

## Docker Development

The repository includes two Docker Compose services:

- `dev`: mounts the local project into the container for development checks.
- `analysis`: runs the code copied into the built image, matching final
  reproducibility verification more closely.

Initial Docker setup:

```bash
make docker-setup
```

Daily Docker checks:

```bash
make docker-lint
make docker-test
make docker-report
```

Final check before a pull request or release:

```bash
make docker-check
```

## Documentation

Sphinx API documentation can be generated with:

```bash
make docs
```

The generated documentation is written to:

```text
docs/_build/html/index.html
```

## Build And Push Docker Image

After changes to code, data, report, dependencies, or Docker configuration are
merged into `main`, rebuild and push the public multi-platform image:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t mykolanocap/reproducible-research-project:latest \
  --push .
```

Verify the pushed image:

```bash
docker pull mykolanocap/reproducible-research-project:latest
mkdir -p outputs
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  mykolanocap/reproducible-research-project:latest
```

## Collaboration

All project work was developed through feature branches and pull requests. For
future changes, start from an up-to-date `main`:

```bash
git checkout main
git pull --ff-only
git checkout -b feature/short-description
```

Before opening a pull request for code, data, report, dependency, Docker, or
pipeline changes, run:

```bash
make docker-check
```

Each pull request should describe what changed and how it was verified.

## AI Usage Disclosure

LLM tools were used to support coding, debugging, documentation, and report
writing. The models used were ChatGPT 5.5, Claude Opus 4.7, and Claude Sonnet
4.6. AI assistance was used to improve reproducible project structure, Docker
workflow, tests, and explanatory text. Externally sourced references and the
original project are cited where relevant; no external code snippets were
included without review and adaptation.

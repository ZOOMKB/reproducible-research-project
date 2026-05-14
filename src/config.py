"""Shared project paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_ATVI_PATH = RAW_DATA_DIR / "ATVI.csv"
PROCESSED_ATVI_PATH = PROCESSED_DATA_DIR / "atvi_processed.csv"

"""Tests for ATVI data processing."""

import numpy as np
import pandas as pd
from src.config import PROCESSED_ATVI_PATH, RAW_ATVI_PATH
from src.data import ATVIDataProcessor


def test_raw_atvi_file_exists() -> None:
    """Raw source data should be available in the repository."""
    assert RAW_ATVI_PATH.exists()


def test_processed_atvi_dataset_shape_and_columns() -> None:
    """Processed dataset should expose the columns used downstream."""
    data = pd.read_csv(PROCESSED_ATVI_PATH)

    expected_columns = {
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adjusted",
        "Volume",
        "log_adjusted",
        "log_close",
        "cc_ret",
        "return_pct",
        "gk_vol",
    }

    assert len(data) == 1762
    assert expected_columns.issubset(data.columns)
    assert data["cc_ret"].isna().sum() == 0
    assert data["gk_vol"].isna().sum() == 0


def test_return_calculation_on_toy_data(tmp_path) -> None:
    """Close-to-close returns should use log differences of Adjusted prices."""
    raw_path = tmp_path / "toy.csv"
    processed_path = tmp_path / "processed.csv"
    raw = pd.DataFrame(
        {
            "Date": ["2015-01-02", "2015-01-05", "2015-01-06"],
            "Open": [10.0, 11.0, 12.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.0, 11.0, 12.0],
            "Adj Close": [10.0, 11.0, 12.0],
            "Volume": [100, 110, 120],
        }
    )
    raw.to_csv(raw_path, index=False)

    processor = ATVIDataProcessor(
        raw_path=raw_path,
        processed_path=processed_path,
        start_date="2015-01-02",
    )
    processed = processor.save_processed()

    expected_cc_ret = np.log(processed["Adjusted"]).diff()
    expected_return_pct = 100 * expected_cc_ret

    pd.testing.assert_series_equal(
        processed["cc_ret"],
        expected_cc_ret,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        processed["return_pct"],
        expected_return_pct,
        check_names=False,
    )

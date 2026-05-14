"""Data loading and processing for the ATVI dataset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_ATVI_PATH, RAW_ATVI_PATH


@dataclass(frozen=True)
class ATVIDataProcessor:
    """Prepare the ATVI source dataset for reproducible analysis.

    The processor reads the raw ATVI CSV file, standardizes the adjusted close
    column name, computes log-price, return, and Garman-Klass volatility
    columns, filters the study period, and writes a stable processed dataset.
    """

    raw_path: Path = RAW_ATVI_PATH
    processed_path: Path = PROCESSED_ATVI_PATH
    start_date: str = "2015-01-03"

    def load_raw(self) -> pd.DataFrame:
        """Load the raw ATVI CSV file.

        Returns:
            Raw ATVI data with ``Date`` parsed as datetime and ``Adj Close``
            renamed to ``Adjusted``.

        Raises:
            FileNotFoundError: If the raw CSV file does not exist.
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_path}")

        data = pd.read_csv(self.raw_path, parse_dates=["Date"])
        return data.rename(columns={"Adj Close": "Adjusted"})

    def process(self) -> pd.DataFrame:
        """Create the processed ATVI dataset.

        Returns:
            Filtered dataset with log-price and return columns.
        """
        data = self.load_raw().copy()

        data["log_adjusted"] = np.log(data["Adjusted"])
        data["log_close"] = np.log(data["Close"])
        data["cc_ret"] = data["log_adjusted"].diff()
        data["return_pct"] = 100 * data["cc_ret"]
        data["gk_vol"] = self.garman_klass(data)

        processed = data.loc[data["Date"] >= self.start_date].reset_index(drop=True)
        return processed

    @staticmethod
    def garman_klass(data: pd.DataFrame, sd: bool = True) -> np.ndarray:
        """Compute the Garman-Klass volatility estimator.

        This is a Python port of the ``.garmanklass`` helper in the original
        R project. It adjusts OHLC prices by the ``Adjusted / Close`` ratio and
        includes the overnight return correction used by the source analysis.

        Args:
            data: DataFrame with ``Open``, ``High``, ``Low``, ``Close``, and
                ``Adjusted`` columns.
            sd: If ``True``, return volatility. If ``False``, return variance.

        Returns:
            Array of Garman-Klass volatility estimates.
        """
        coef = data["Adjusted"].to_numpy() / data["Close"].to_numpy()
        high = np.log(data["High"].to_numpy() * coef)
        low = np.log(data["Low"].to_numpy() * coef)
        open_ = np.log(data["Open"].to_numpy() * coef)
        close = np.log(data["Close"].to_numpy() * coef)

        u = high - open_
        d = low - open_
        c = close - open_

        intraday = (
            0.511 * (u - d) ** 2 - 0.019 * (c * (u + d) - 2 * u * d) - 0.383 * c**2
        )

        raw_open = data["Open"].to_numpy()
        raw_close = data["Close"].to_numpy()

        overnight = np.empty(len(data))
        overnight[0] = np.nan
        overnight[1:] = np.log(raw_open[1:] / raw_close[:-1])

        open_to_close = np.log(raw_close / raw_open)
        overnight_sum = np.nansum(overnight**2)
        open_to_close_sum = np.nansum(open_to_close**2)
        overnight_fraction = np.clip(
            overnight_sum / (overnight_sum + open_to_close_sum),
            0.01,
            0.99,
        )

        adjusted_variance = (
            0.12 * overnight**2 / overnight_fraction
            + (0.88 / (1 - overnight_fraction)) * intraday
        )

        return 1.034 * np.sqrt(adjusted_variance) if sd else adjusted_variance

    def save_processed(self) -> pd.DataFrame:
        """Process raw data and write the processed CSV file.

        Returns:
            Processed dataset that was written to disk.
        """
        processed = self.process()
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(self.processed_path, index=False)
        return processed


def main() -> None:
    """Generate the processed ATVI dataset from raw data."""
    processor = ATVIDataProcessor()
    processed = processor.save_processed()
    print(f"Saved {len(processed):,} rows to {processor.processed_path}")


if __name__ == "__main__":
    main()

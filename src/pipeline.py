"""Small orchestration helpers for reproducible analysis steps."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import ATVIDataProcessor
from src.diagnostics import PriceEDA


@dataclass(frozen=True)
class DataEDAResult:
    """Artifacts produced by the data and price EDA pipeline."""

    processed_data: pd.DataFrame
    adf_summary: pd.DataFrame
    price_plot_path: Path
    acf_pacf_plot_path: Path


def run_data_eda(save_figures: bool = True) -> DataEDAResult:
    """Run the data processing and price-level EDA steps.

    Args:
        save_figures: Whether to write EDA figures to ``outputs/figures``.

    Returns:
        Data and EDA artifacts used by the Quarto report.
    """
    processor = ATVIDataProcessor()
    processed_data = processor.save_processed()

    eda = PriceEDA(data=processed_data)
    price_figure = eda.price_plot(save=save_figures)
    acf_pacf_figure = eda.acf_pacf_plot(save=save_figures)
    adf_summary = eda.adf_unit_root_summary()

    plt.close(price_figure)
    plt.close(acf_pacf_figure)

    return DataEDAResult(
        processed_data=processed_data,
        adf_summary=adf_summary,
        price_plot_path=eda.figures_dir / "01_price_panels.png",
        acf_pacf_plot_path=eda.figures_dir / "02_price_acf_pacf.png",
    )


def main() -> None:
    """Run data processing and EDA from the command line."""
    result = run_data_eda()
    print(f"Processed rows: {len(result.processed_data):,}")
    print(f"Price plot: {result.price_plot_path}")
    print(f"ACF/PACF plot: {result.acf_pacf_plot_path}")
    print(result.adf_summary.to_string(index=False))


if __name__ == "__main__":
    main()

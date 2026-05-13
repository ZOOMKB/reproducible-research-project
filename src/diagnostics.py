"""Exploratory diagnostics for ATVI price series."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from src.config import OUTPUTS_DIR, PROCESSED_ATVI_PATH


@dataclass(frozen=True)
class PriceEDA:
    """Run exploratory diagnostics for ATVI price levels."""

    data: pd.DataFrame
    figures_dir: Path = OUTPUTS_DIR / "figures"

    @classmethod
    def from_processed_csv(cls, path: Path = PROCESSED_ATVI_PATH) -> "PriceEDA":
        """Create an EDA object from the processed ATVI dataset."""
        data = pd.read_csv(path, parse_dates=["Date"])
        return cls(data=data)

    def price_plot(self, save: bool = True) -> Figure:
        """Plot Close, log(Close), Adjusted, and log(Adjusted) prices.

        Args:
            save: Whether to save the figure to ``outputs/figures``.

        Returns:
            Matplotlib figure with the price panels.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        panels = [
            ("Close", "Close", "Price"),
            ("log_close", "Log(Close)", "Log price"),
            ("Adjusted", "Adjusted Close", "Price"),
            ("log_adjusted", "Log(Adjusted)", "Log price"),
        ]

        for ax, (column, title, ylabel) in zip(axes.ravel(), panels, strict=True):
            ax.plot(self.data["Date"], self.data[column], linewidth=0.9)
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel(ylabel)

        fig.tight_layout()
        if save:
            self._save_figure(fig, "01_price_panels.png")
        return fig

    def acf_pacf_plot(self, save: bool = True) -> Figure:
        """Plot ACF and PACF of log adjusted prices up to lag 150."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))

        plot_acf(
            self.data["log_adjusted"],
            lags=150,
            ax=axes[0],
            title="ACF of log adjusted prices",
            zero=False,
        )
        plot_pacf(
            self.data["log_adjusted"],
            lags=150,
            ax=axes[1],
            title="PACF of log adjusted prices",
            zero=False,
            method="ywm",
        )

        for ax in axes:
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")

        fig.tight_layout()
        if save:
            self._save_figure(fig, "02_price_acf_pacf.png")
        return fig

    def adf_unit_root_summary(self) -> pd.DataFrame:
        """Run the two-step ADF unit-root check used in the original analysis."""
        series = self.data["log_adjusted"].dropna()
        trend_result = adfuller(series, regression="ct", autolag="AIC", maxlag=20)
        drift_result = adfuller(series, regression="c", autolag="AIC", maxlag=20)

        rows = [
            self._adf_row("Trend", "tau3", trend_result),
            self._adf_row("Drift", "tau2", drift_result),
        ]
        return pd.DataFrame(rows)

    def _save_figure(self, fig: Figure, filename: str) -> Path:
        """Save a figure under the configured figures directory."""
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.figures_dir / filename
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return output_path

    @staticmethod
    def _adf_row(model: str, statistic_name: str, result: tuple) -> dict[str, object]:
        """Format a statsmodels ADF result for reporting."""
        statistic = result[0]
        p_value = result[1]
        used_lags = result[2]
        critical_values = result[4]
        critical_5 = critical_values["5%"]

        return {
            "model": model,
            "statistic": statistic_name,
            "value": statistic,
            "p_value": p_value,
            "used_lags": used_lags,
            "critical_5pct": critical_5,
            "decision_5pct": "reject unit root"
            if statistic < critical_5
            else "fail to reject unit root",
        }

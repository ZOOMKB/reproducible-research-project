"""Returns analysis for the ATVI reproducible research project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, bds, het_arch
from statsmodels.tsa.stattools import adfuller

from src.config import OUTPUTS_DIR, PROCESSED_ATVI_PATH
from src.data import ATVIDataProcessor


@dataclass(frozen=True)
class ReturnsAnalyzer:
    """Analyze ATVI percentage log-returns."""

    data: pd.DataFrame
    figures_dir: Path = OUTPUTS_DIR / "figures"

    @classmethod
    def from_processed_csv(cls, path: Path = PROCESSED_ATVI_PATH) -> ReturnsAnalyzer:
        """Create a returns analyzer from the processed ATVI dataset."""
        data = pd.read_csv(path, parse_dates=["Date"])
        return cls(data=data)

    @classmethod
    def from_raw_processor(cls) -> ReturnsAnalyzer:
        """Create a returns analyzer by processing the raw ATVI dataset."""
        data = ATVIDataProcessor().process()
        return cls(data=data)

    def compute_yret(self) -> pd.Series:
        """Compute percentage close-to-close log returns."""
        yret = self.data["return_pct"].copy()
        yret.index = pd.DatetimeIndex(self.data["Date"])
        return yret.dropna()

    def returns_plot(self, save: bool = True) -> Figure:
        """Plot percentage returns through time."""
        yret = self.compute_yret()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(yret.index, yret.to_numpy(), linewidth=0.9)
        ax.axhline(y=0.0, linestyle="--", linewidth=1)
        ax.set_title("Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("%")

        fig.tight_layout()
        if save:
            self._save_figure(fig, "03_returns.png")
        return fig

    def acf_pacf_plot(self, save: bool = True) -> Figure:
        """Plot ACF and PACF of percentage returns up to lag 150."""
        yret = self.compute_yret().to_numpy()

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        plot_acf(yret, lags=150, ax=axes[0], title="ACF of returns", zero=False)
        plot_pacf(
            yret,
            lags=150,
            ax=axes[1],
            title="PACF of returns",
            zero=False,
            method="ywm",
        )

        for ax in axes:
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")

        fig.tight_layout()
        if save:
            self._save_figure(fig, "04_returns_acf_pacf.png")
        return fig

    def ljung_box_summary(self) -> pd.DataFrame:
        """Run Ljung-Box tests on returns."""
        yret = self.compute_yret().to_numpy()
        result = acorr_ljungbox(
            yret,
            lags=[2, 5, 10, 15, 20, 30, 50],
            return_df=True,
        )
        print("\nLjung-Box statistics on log-returns")
        print(result)
        return result

    def adf_returns_summary(self) -> pd.DataFrame:
        """Run ADF unit-root checks on returns."""
        yret = self.compute_yret().to_numpy()

        trend_result = adfuller(yret, regression="ct", autolag="AIC", maxlag=20)
        drift_result = adfuller(yret, regression="c", autolag="AIC", maxlag=20)

        self._print_adf_full_result("Test1: ADF with trend", trend_result)
        self._print_adf_full_result("Test2: ADF with drift", drift_result)

        rows = [
            self._adf_row("Trend", "tau3", trend_result),
            self._adf_row("Drift", "tau2", drift_result),
        ]
        result = pd.DataFrame(rows)

        print("\nADF unit-root check on returns")
        print(result)
        return result

    def bds_summary(self) -> Any:
        """Run BDS independence test on returns."""
        yret = self.compute_yret().to_numpy()
        epsilons = np.std(yret, ddof=1) * np.array([0.5, 1.0, 1.5, 2.0])
        result = bds(yret, max_dim=4, epsilon=epsilons)
        self._print_bds_result(result, epsilons)
        return result

    def arch_summary(self) -> pd.DataFrame:
        """Run ARCH LM tests on demeaned returns."""
        yret = self.compute_yret().to_numpy()
        demeaned = yret - np.mean(yret)

        rows: list[dict[str, float]] = []
        for lag in [4, 8, 12, 16]:
            lm_stat, lm_pvalue, _f_stat, _f_pvalue = het_arch(demeaned, nlags=lag)
            rows.append({"lag": lag, "LM-stat": lm_stat, "p-value": lm_pvalue})

        result = pd.DataFrame(rows)
        print("\nARCH test on demeaned log-returns")
        print(result)
        return result

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
        critical_5 = result[4]["5%"]

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

    @staticmethod
    def _print_adf_full_result(name: str, result: tuple) -> None:
        """Print a fuller ADF result similar to R's summary output."""
        statistic, p_value, used_lags, nobs, critical_values, icbest = result[:6]

        print(f"\n-----\n{name}")
        print(f"ADF statistic: {statistic:.6f}")
        print(f"p-value: {p_value:.6g}")
        print(f"used lags: {used_lags}")
        print(f"number of observations: {nobs}")
        print(f"information criterion: {icbest:.6f}")
        print("critical values:")
        for level, value in critical_values.items():
            print(f"  {level}: {value:.6f}")

    @staticmethod
    def _print_bds_result(result: Any, epsilons: np.ndarray) -> None:
        """Print BDS statistics and p-values in readable tables."""
        stat, pvalue = result
        dimensions = [2, 3, 4]
        columns = [f"eps={eps:.4f}" for eps in epsilons]

        print("\nBDS test on returns")
        print("\nStandard Normal statistics")
        print(pd.DataFrame(np.asarray(stat), index=dimensions, columns=columns))
        print("\np-values")
        print(pd.DataFrame(np.asarray(pvalue), index=dimensions, columns=columns))


def compute_yret(data: pd.DataFrame) -> pd.Series:
    """Compute percentage close-to-close log returns."""
    return ReturnsAnalyzer(data=data).compute_yret()

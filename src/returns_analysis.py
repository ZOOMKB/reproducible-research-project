"""Returns analysis for the ATVI reproducible research project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import jarque_bera as scipy_jarque_bera
from scipy.stats import norm, probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, bds

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
            ax.set_ylim(-0.1, 0.1)

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

    def bds_summary(self) -> pd.DataFrame:
        """Run BDS independence test on returns."""
        yret = self.compute_yret().to_numpy()
        epsilons = np.std(yret, ddof=1) * np.array([0.5, 1.0, 1.5, 2.0])

        rows: list[dict[str, float]] = []
        for epsilon in epsilons:
            stat, pvalue = bds(yret, max_dim=4, epsilon=float(epsilon))
            for dimension, dim_stat, dim_pvalue in zip(
                [2, 3, 4],
                np.asarray(stat),
                np.asarray(pvalue),
                strict=True,
            ):
                rows.append(
                    {
                        "dimension": dimension,
                        "epsilon": float(epsilon),
                        "statistic": float(dim_stat),
                        "p_value": float(dim_pvalue),
                    }
                )

        result = pd.DataFrame(rows)
        print("\nBDS test on returns")
        print(result)
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

    def acf_triple_plot(self, save: bool = True) -> Figure:
        """Plot ACFs of returns, absolute returns, and squared returns."""
        yret = self.compute_yret().to_numpy()

        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        plot_acf(yret, lags=100, ax=axes[0], title="Returns", zero=False)
        plot_acf(np.abs(yret), lags=100, ax=axes[1], title="|Returns|", zero=False)
        plot_acf(yret**2, lags=100, ax=axes[2], title="Returns²", zero=False)

        for ax in axes:
            ax.set_ylim(-0.25, 0.25)

        fig.tight_layout()
        if save:
            self._save_figure(fig, "05_returns_acf_triple.png")
        return fig

    def distribution_plot(self, save: bool = True) -> Figure:
        """Plot histogram with normal density and normal QQ plot."""
        yret = self.compute_yret().to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(yret, bins=200, density=True)
        x_grid = np.linspace(-10, 10, 200)
        pdf = norm.pdf(x_grid, loc=np.mean(yret), scale=np.std(yret, ddof=1))
        axes[0].plot(x_grid, pdf)
        axes[0].set_xlim(-10, 10)
        axes[0].set_title("Returns")

        scaled_yret = (yret - np.mean(yret)) / np.std(yret, ddof=1)
        probplot(scaled_yret, dist="norm", plot=axes[1])
        axes[1].set_title("Normal Q-Q")

        fig.tight_layout()
        if save:
            self._save_figure(fig, "06_returns_distribution.png")
        return fig

    def jarque_bera_summary(self) -> tuple[float, float]:
        """Run Jarque-Bera normality test on returns."""
        yret = self.compute_yret().to_numpy()
        test_result = scipy_jarque_bera(yret)
        statistic = float(test_result.statistic)
        pvalue = float(test_result.pvalue)

        print(f"\nJarque-Bera statistic: {statistic:.4f}, p-value: {pvalue:.4e}")
        return statistic, pvalue

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


def compute_yret(data: pd.DataFrame) -> pd.Series:
    """Compute percentage close-to-close log returns."""
    return ReturnsAnalyzer(data=data).compute_yret()


def main() -> None:
    """Run the Section 3 returns analysis."""
    analyzer = ReturnsAnalyzer.from_processed_csv()

    analyzer.returns_plot()
    analyzer.acf_pacf_plot()
    analyzer.ljung_box_summary()
    analyzer.adf_returns_summary()
    analyzer.bds_summary()
    analyzer.arch_summary()
    analyzer.acf_triple_plot()
    analyzer.distribution_plot()
    analyzer.jarque_bera_summary()


if __name__ == "__main__":
    main()

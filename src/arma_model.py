"""ARMA baseline model for the ATVI reproducible research project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.univariate import ARX, StudentsT
from arch.univariate.base import ARCHModelResult
from matplotlib.figure import Figure
from statsmodels.graphics.tsaplots import plot_acf

from src.config import OUTPUTS_DIR, PROCESSED_ATVI_PATH
from src.data import ATVIDataProcessor
from src.returns_analysis import compute_yret


@dataclass
class ARMAModel:
    """Fit and diagnose an ARMA(1,0)-Student-t model for returns."""

    yret: pd.Series | np.ndarray
    figures_dir: Path = OUTPUTS_DIR / "figures"

    def __post_init__(self) -> None:
        """Clean returns and create output directory."""
        self.yret = self._clean_yret(self.yret)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.result: ARCHModelResult | None = None

    @classmethod
    def from_processed_csv(cls, path: Path = PROCESSED_ATVI_PATH) -> ARMAModel:
        """Create an ARMA model object from processed ATVI data."""
        data = pd.read_csv(path, parse_dates=["Date"])
        return cls(yret=compute_yret(data))

    @classmethod
    def from_raw_processor(cls) -> ARMAModel:
        """Create an ARMA model object by processing raw ATVI data."""
        data = ATVIDataProcessor().process()
        return cls(yret=compute_yret(data))

    def fit(self, cov_type: str = "robust") -> ARCHModelResult:
        """Fit ARMA(1,0) with Student-t errors."""
        model = ARX(self.yret, lags=1, constant=True)
        model.distribution = StudentsT()
        self.result = model.fit(disp="off", cov_type=cov_type)
        self.print_summary_tables(self.result)
        return self.result

    def print_summary_tables(self, result: ARCHModelResult | None = None) -> None:
        """Print information criteria and coefficient table."""
        fitted = self._require_result(result)

        nobs = fitted.nobs
        nparams = len(fitted.params)
        loglikelihood = fitted.loglikelihood

        information_criteria = pd.Series(
            {
                "Akaike": fitted.aic,
                "Bayes": fitted.bic,
                "Shibata": -2 * loglikelihood
                + nobs * np.log((nobs + 2 * nparams) / nobs),
                "Hannan-Quinn": -2 * loglikelihood + 2 * nparams * np.log(np.log(nobs)),
            },
            name="value",
        )

        print("\nInformation Criteria")
        print(information_criteria)

        table = pd.DataFrame(
            {
                "Estimate": fitted.params,
                "Std. Error": fitted.std_err,
                "t value": fitted.tvalues,
                "Pr(>|t|)": fitted.pvalues,
            }
        )

        print("\nCoefficient table")
        print(table)

        if getattr(fitted, "cov_type", "").lower() == "robust":
            print("\nRobust coefficient table")
            print(table)

    def residual_acf_plot(
        self,
        result: ARCHModelResult | None = None,
        save: bool = True,
    ) -> Figure:
        """Plot ACF diagnostics for ARMA residuals."""
        fitted = self._require_result(result)
        residuals = self._extract_residuals(fitted)

        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        plot_acf(residuals, lags=100, ax=axes[0], title="Residuals", zero=False)
        plot_acf(np.abs(residuals), lags=100, ax=axes[1], title="|res|", zero=False)
        plot_acf(residuals**2, lags=100, ax=axes[2], title="res²", zero=False)

        for ax in axes:
            ax.set_ylim(-0.25, 0.25)

        fig.tight_layout()
        if save:
            self._save_figure(fig, "07_arma_residuals.png")
        return fig

    def _save_figure(self, fig: Figure, filename: str) -> Path:
        """Save a figure under the configured figures directory."""
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.figures_dir / filename
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return output_path

    @staticmethod
    def _clean_yret(yret: pd.Series | np.ndarray) -> pd.Series:
        """Convert returns to a finite pandas Series."""
        if isinstance(yret, pd.Series):
            series = yret.astype(float)
        else:
            series = pd.Series(np.asarray(yret, dtype=float))
        return series.replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _extract_residuals(result: ARCHModelResult) -> np.ndarray:
        """Extract finite residuals from a fitted model."""
        residuals = result.resid
        if isinstance(residuals, pd.Series):
            values = residuals.dropna().to_numpy(dtype=float)
        else:
            values = pd.Series(np.asarray(residuals, dtype=float)).dropna().to_numpy()
        return values[np.isfinite(values)]

    def _require_result(self, result: ARCHModelResult | None = None) -> ARCHModelResult:
        """Return a fitted result or raise an informative error."""
        fitted = result if result is not None else self.result
        if fitted is None:
            raise ValueError("No fitted ARMA model is available. Call fit() first.")
        return fitted


def fit_arma(yret: pd.Series | np.ndarray) -> ARCHModelResult:
    """Fit ARMA(1,0) with Student-t errors."""
    return ARMAModel(yret=yret).fit()


def plot_arma_residuals(arma_result: ARCHModelResult) -> Figure:
    """Plot ACF diagnostics for ARMA residuals."""
    model = ARMAModel(yret=pd.Series(arma_result.model.y))
    return model.residual_acf_plot(arma_result)


def main() -> None:
    """Run the Section 4 ARMA analysis."""
    arma = ARMAModel.from_processed_csv()
    result = arma.fit()
    arma.residual_acf_plot(result)


if __name__ == "__main__":
    main()

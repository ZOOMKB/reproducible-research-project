"""ARMA baseline model for the ATVI reproducible research project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from arch.univariate import ARX, StudentsT
from arch.univariate.base import ARCHModelResult

from src.config import OUTPUTS_DIR, PROCESSED_ATVI_PATH
from src.data import ATVIDataProcessor
from src.returns_analysis import compute_yret


@dataclass
class ARMAModel:
    """Fit an ARMA(1,0)-Student-t model for returns."""

    yret: pd.Series | np.ndarray
    figures_dir: Path = OUTPUTS_DIR / "figures"

    def __post_init__(self) -> None:
        """Clean returns and initialize model result."""
        self.yret = self._clean_yret(self.yret)
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

    @staticmethod
    def _clean_yret(yret: pd.Series | np.ndarray) -> pd.Series:
        """Convert returns to a finite pandas Series."""
        if isinstance(yret, pd.Series):
            series = yret.astype(float)
        else:
            series = pd.Series(np.asarray(yret, dtype=float))
        return series.replace([np.inf, -np.inf], np.nan).dropna()

    def _require_result(self, result: ARCHModelResult | None = None) -> ARCHModelResult:
        """Return a fitted result or raise an informative error."""
        fitted = result if result is not None else self.result
        if fitted is None:
            raise ValueError("No fitted ARMA model is available. Call fit() first.")
        return fitted


def fit_arma(yret: pd.Series | np.ndarray) -> ARCHModelResult:
    """Fit ARMA(1,0) with Student-t errors."""
    return ARMAModel(yret=yret).fit()

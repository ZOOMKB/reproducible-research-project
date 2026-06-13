"""GARCH family model fitting for the ATVI reproducible research project.

Fits and compares five GARCH-type volatility models on ATVI percentage
log-returns. Reproduces Section 5 of TSA-GARCH.md (original R analysis).

Models fitted:
    fit1  sGARCH(1,1)            — symmetric variance response
    fit2  GJR-GARCH(1,1)         — leverage effect via gamma term
    fit4  GJR via fGARCH         — verification step only
    fit5  T-GARCH(1,1)           — native rugarch via rpy2 (exact reproduction)
    fit3  GJR with var. targeting — omega derived, not estimated
    fit6  IGARCH(1,1)            — alpha+beta=1 enforced via scipy

Usage (standalone):
    uv run python -m src.garch_models

Other modules import:
    from src.garch_models import GARCHModels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from arch.univariate import ARX, EGARCH, GARCH, StudentsT
from arch.univariate.base import ARCHModelResult

from src.config import OUTPUTS_DIR, PROCESSED_ATVI_PATH
from src.data import ATVIDataProcessor
from src.garch_diagnostics import GARCHDiagnostics
from src.returns_analysis import compute_yret

# plot style
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)

FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_rpy2():
    """Lazy-load rpy2 and rugarch to avoid import errors if not installed.

    Returns:
        Tuple of (ro, rugarch) rpy2 modules.

    Raises:
        ImportError: If rpy2 is not installed.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr

        rugarch = importr("rugarch")
        return ro, rugarch
    except ImportError as e:
        raise ImportError(
            "rpy2 is required for T-GARCH fitting. "
            "Install with: uv add rpy2==3.5.16 "
            "and ensure R with rugarch is available."
        ) from e


@dataclass
class _TGARCHForecastResult:
    """Minimal forecast result container matching arch's interface.

    Attributes:
        mean: DataFrame of forecast means with columns h.1 ... h.N.
        variance: DataFrame of forecast variances with columns h.1 ... h.N.
    """

    mean: pd.DataFrame
    variance: pd.DataFrame


class TGARCHResult:
    """Wrapper around a rugarch T-GARCH fit result from rpy2.

    Provides a consistent interface matching what GARCHDiagnostics and
    GARCHForecaster expect, wrapping the underlying R object.

    Attributes:
        r_fit: The raw rpy2 R fit object from ugarchfit.
        params: Fitted parameters as pandas Series (converted form).
        conditional_volatility: In-sample fitted sigma as numpy array.
        std_resid: Standardised residuals z = u/sigma as numpy array.
        resid: Raw residuals as numpy array.
        aic: Akaike information criterion (per observation).
        bic: Bayes information criterion (per observation).
        nobs: Number of observations used in fitting.
        loglikelihood: Log-likelihood value.
        param_cov: Parameter covariance matrix as numpy array.
        np_: Number of parameters.
        matcoef: Standard coefficient table as DataFrame.
        robust_matcoef: Robust coefficient table as DataFrame.
    """

    def __init__(self, r_fit, ro, rugarch, yret: np.ndarray) -> None:
        """Initialise from a rugarch fit R object.

        Args:
            r_fit: Fitted ugarchfit R object from rpy2.
            ro: rpy2.robjects module.
            rugarch: rugarch R package imported via rpy2.
            yret: Original percentage log-returns used for fitting,
                  stored for ex-post forecasting refits.
        """
        self.r_fit = r_fit
        self._ro = ro
        self._rugarch = rugarch
        self._yret_r = ro.FloatVector(yret)

        # extract information criteria
        ic_r = rugarch.infocriteria(r_fit)
        ic = np.array(list(ic_r)).flatten()
        self.aic = float(ic[0])
        self.bic = float(ic[1])

        # extract raw fGARCH coefficients
        coef_fn = ro.r["coef"]
        coefs_r = coef_fn(r_fit)
        names = list(coefs_r.names)
        values = list(coefs_r)
        raw_params = pd.Series(dict(zip(names, values, strict=False)))

        # apply fgarch_to_gjr conversion (TGARCH path)
        # alpha_s = alpha*(1-eta1),  gamma_s = 2*alpha*eta1
        alpha = raw_params["alpha1"]
        eta1 = raw_params["eta11"]
        alpha_s = alpha * (1 - eta1)
        gamma_s = 2 * alpha * eta1

        converted = raw_params.copy()
        converted["alpha1"] = alpha_s
        converted = converted.rename({"eta11": "gamma1"})
        converted["gamma1"] = gamma_s
        self.params = converted

        # extract sigma and residuals
        sigma_fn = ro.r["sigma"]
        resid_fn = ro.r["residuals"]
        self.conditional_volatility = np.array(list(sigma_fn(r_fit))).flatten()
        self.std_resid = np.array(list(resid_fn(r_fit, standardize=True))).flatten()
        self.resid = np.array(list(resid_fn(r_fit, standardize=False))).flatten()

        # nobs and loglikelihood
        likelihood_fn = ro.r["likelihood"]
        self.loglikelihood = float(list(likelihood_fn(r_fit))[0])
        self.nobs = len(self.conditional_volatility)
        self.np_ = len(self.params)

        # extract covariance matrix for Nyblom
        vcov_fn = ro.r["vcov"]
        vcov_r = vcov_fn(r_fit)
        self.param_cov = np.array(list(vcov_r)).reshape(self.np_, self.np_)

        # build coefficient tables
        se_r = np.sqrt(np.abs(np.diag(self.param_cov)))
        t_vals = self.params.values / se_r
        p_vals = 2 * (1 - stats.norm.cdf(np.abs(t_vals)))
        cols = ["Estimate", "Std. Error", "t value", "Pr(>|t|)"]
        self.matcoef = pd.DataFrame(
            np.column_stack([self.params.values, se_r, t_vals, p_vals]),
            index=self.params.index,
            columns=cols,
        )

        # robust covariance via rugarch
        try:
            vcov_rob_r = vcov_fn(r_fit, robust=True)
            vcov_rob = np.array(list(vcov_rob_r)).reshape(self.np_, self.np_)
            se_rob = np.sqrt(np.abs(np.diag(vcov_rob)))
            t_rob = self.params.values / se_rob
            p_rob = 2 * (1 - stats.norm.cdf(np.abs(t_rob)))
            self.robust_matcoef = pd.DataFrame(
                np.column_stack([self.params.values, se_rob, t_rob, p_rob]),
                index=self.params.index,
                columns=cols,
            )
        except Exception:
            self.robust_matcoef = self.matcoef.copy()

    def forecast(
        self,
        horizon: int = 10,
        start: int | None = None,
        method: str = "analytic",
        reindex: bool = False,
        **kwargs,
    ) -> _TGARCHForecastResult:
        """Produce h-step-ahead forecasts via rugarch ugarchforecast.

        Returns a _TGARCHForecastResult with .mean and .variance DataFrames
        matching the arch library's forecast result interface so that
        GARCHForecaster works without modification.

        Args:
            horizon: Number of steps ahead to forecast.
            start: Start index for ex-post forecasting. If None,
                   forecasts from the end of the sample (ex-ante).
            method: Ignored — rugarch always uses analytic forecasts.
            reindex: Ignored — included for interface compatibility.

        Returns:
            _TGARCHForecastResult with .mean and .variance DataFrames.
        """
        ro = self._ro
        rugarch = self._rugarch
        cols = [f"h.{i}" for i in range(1, horizon + 1)]

        if start is None:
            # ex-ante: forecast from end of sample
            forc = rugarch.ugarchforecast(
                fitORspec=self.r_fit,
                n_ahead=horizon,
                data=ro.NULL,
                out_sample=0,
                n_roll=0,
            )
            sigma_fn = ro.r["sigma"]
            fitted_fn = ro.r["fitted"]
            sigma_vec = np.array(list(sigma_fn(forc))).flatten()
            mean_vec = np.array(list(fitted_fn(forc))).flatten()

            mean_df = pd.DataFrame([mean_vec], columns=cols)
            variance_df = pd.DataFrame([sigma_vec**2], columns=cols)

        else:
            # ex-post: refit with out_sample so rugarch can roll.
            # ugarchforecast requires out_sample >= n_roll, and the
            # original fit used out_sample=0, so we must refit first.
            n_roll = max(self.nobs - start - 1, 0)
            spec_fn = ro.r["getspec"]
            spec5 = spec_fn(self.r_fit)

            refit = rugarch.ugarchfit(
                spec=spec5,
                data=self._yret_r,
                solver="solnp",
                out_sample=n_roll,
            )
            forc = rugarch.ugarchforecast(
                fitORspec=refit,
                n_ahead=horizon,
                data=ro.NULL,
                out_sample=n_roll,
                n_roll=n_roll,
            )
            sigma_fn = ro.r["sigma"]
            fitted_fn = ro.r["fitted"]

            raw_sigma = np.array(list(sigma_fn(forc))).flatten()
            raw_mean = np.array(list(fitted_fn(forc))).flatten()

            n_rows = max(len(raw_sigma) // horizon, 1)
            sigma_mat = raw_sigma[: n_rows * horizon].reshape(n_rows, horizon)
            mean_mat = raw_mean[: n_rows * horizon].reshape(n_rows, horizon)

            mean_df = pd.DataFrame(mean_mat, columns=cols)
            variance_df = pd.DataFrame(sigma_mat**2, columns=cols)

        return _TGARCHForecastResult(mean=mean_df, variance=variance_df)


@dataclass
class GARCHModels:
    """Fit and compare GARCH-family volatility models on ATVI returns.

    Follows the same dataclass pattern as ARMAModel (src/arma_model.py).
    After construction, call the fit_* methods in the order defined in
    main() to reproduce the R analysis sequence exactly.

    Attributes:
        yret (np.ndarray): Percentage log-returns, NaN dropped.
        figures_dir (Path): Directory where all plots are saved.
        fit1 (ARCHModelResult | None): sGARCH(1,1) result.
        fit2 (ARCHModelResult | None): GJR-GARCH(1,1) result.
        fit3 (ARCHModelResult | None): GJR with variance targeting.
        fit4 (ARCHModelResult | None): GJR via fGARCH (verification).
        fit5 (TGARCHResult | None): T-GARCH(1,1) via rpy2+rugarch.
        fit6 (ARCHModelResult | None): IGARCH(1,1) result.
        fit4c (dict | None): Converted fGARCH-GJR coefficient table.
        fit5c (dict | None): Kept for backward compatibility — None.
        np1 (int): Parameter count for sGARCH (6).
        np2 (int): Parameter count for GJR-GARCH (7).
        np3 (int): Parameter count for GJR+VT (6).
        np5 (int): Parameter count for T-GARCH (7).
        np6 (int): Parameter count for IGARCH (5).
    """

    yret: np.ndarray
    figures_dir: Path = field(default_factory=lambda: FIGURES_DIR)

    # fitted model objects — set to None until fit_* is called
    fit1: ARCHModelResult | None = field(default=None, init=False)
    fit2: ARCHModelResult | None = field(default=None, init=False)
    fit3: ARCHModelResult | None = field(default=None, init=False)
    fit4: ARCHModelResult | None = field(default=None, init=False)
    fit5: Any | None = field(default=None, init=False)
    fit6: ARCHModelResult | None = field(default=None, init=False)
    fit4c: dict | None = field(default=None, init=False)
    fit5c: dict | None = field(default=None, init=False)

    # parameter counts — set when each model is fitted
    np1: int = field(default=6, init=False)
    np2: int = field(default=7, init=False)
    np3: int = field(default=6, init=False)
    np5: int = field(default=7, init=False)
    np6: int = field(default=5, init=False)

    # construction helpers

    @classmethod
    def from_processed_csv(cls, path: Path = PROCESSED_ATVI_PATH) -> GARCHModels:
        """Create a GARCHModels instance from the processed ATVI dataset.

        Args:
            path: Path to atvi_processed.csv.

        Returns:
            GARCHModels instance ready for fitting.
        """
        data = pd.read_csv(path, parse_dates=["Date"])
        yret = compute_yret(data).to_numpy()
        return cls(yret=yret)

    @classmethod
    def from_raw_processor(cls) -> GARCHModels:
        """Create a GARCHModels instance by processing raw ATVI data.

        Returns:
            GARCHModels instance ready for fitting.
        """
        data = ATVIDataProcessor().process()
        yret = compute_yret(data).to_numpy()
        return cls(yret=yret)

    # private helpers

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """Save a figure to the figures directory.

        Args:
            fig: Matplotlib figure to save.
            filename: Output filename.

        Returns:
            Path where the figure was saved.
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.figures_dir / filename
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return output_path

    @staticmethod
    def _print_ic(result: Any, label: str) -> None:
        """Print information criteria for a fitted model.

        Args:
            result: Fitted model result (ARCHModelResult or TGARCHResult).
            label: Human-readable model name for the header.
        """
        nobs = result.nobs
        nparams = len(result.params)
        ll = result.loglikelihood
        shibata = -2 * ll + nobs * np.log((nobs + 2 * nparams) / nobs)
        hq = -2 * ll + 2 * nparams * np.log(np.log(nobs))

        ic = pd.Series(
            {
                "Akaike": result.aic,
                "Bayes": result.bic,
                "Shibata": shibata,
                "Hannan-Quinn": hq,
            },
            name="value",
        )
        print(f"\nInformation Criteria — {label}")
        print(ic.to_string())

    @staticmethod
    def _print_coef_tables(result: Any, label: str) -> None:
        """Print standard and robust coefficient tables.

        Args:
            result: Fitted model result (ARCHModelResult or TGARCHResult).
            label: Human-readable model name for the header.
        """
        if hasattr(result, "matcoef"):
            # TGARCHResult path
            print(f"\nCoefficient table — {label}")
            print(result.matcoef.to_string())
            print(f"\nRobust coefficient table — {label}")
            print(result.robust_matcoef.to_string())
        else:
            # ARCHModelResult path
            table = pd.DataFrame(
                {
                    "Estimate": result.params,
                    "Std. Error": result.std_err,
                    "t value": result.tvalues,
                    "Pr(>|t|)": result.pvalues,
                }
            )
            print(f"\nCoefficient table — {label}")
            print(table.to_string())

            if getattr(result, "cov_type", "").lower() != "robust":
                robust = result.model.fit(
                    disp="off",
                    cov_type="robust",
                    starting_values=result.params.values,
                )
            else:
                robust = result

            robust_table = pd.DataFrame(
                {
                    "Estimate": robust.params,
                    "Std. Error": robust.std_err,
                    "t value": robust.tvalues,
                    "Pr(>|t|)": robust.pvalues,
                }
            )
            print(f"\nRobust coefficient table — {label}")
            print(robust_table.to_string())

    # model fitting

    def fit_sgarch(self) -> ARCHModelResult:
        """Fit sGARCH(1,1) + ARMA(1,0) + Student-t errors.

        Variance equation: sigma2(t) = omega + alpha*u2(t-1) + beta*sigma2(t-1).
        Stores result as self.fit1 with np1=6 parameters.

        Returns:
            Fitted ARCHModelResult stored as self.fit1.
        """
        model = ARX(self.yret, lags=1, constant=True)
        model.volatility = GARCH(p=1, q=1)
        model.distribution = StudentsT()
        self.fit1 = model.fit(disp="off", cov_type="robust")
        self.np1 = len(self.fit1.params)

        self._print_ic(self.fit1, "sGARCH(1,1)")
        self._print_coef_tables(self.fit1, "sGARCH(1,1)")

        return self.fit1

    def fit_gjrgarch(self) -> ARCHModelResult:
        """Fit GJR-GARCH(1,1) + ARMA(1,0) + Student-t errors.

        Variance equation:
            sigma2(t) = omega + alpha*u2(t-1)
                        + gamma*u2(t-1)*I(u(t-1)<0) + beta*sigma2(t-1).

        The o=1 argument in GARCH adds the asymmetric leverage term.
        Only IC and coefficient tables are printed — no diagnostics.
        Stores result as self.fit2 with np2=7 parameters.

        Expected: alpha1 not significant (p~1.0), gamma1=0.13526 significant,
        AIC=4.068428.

        Returns:
            Fitted ARCHModelResult stored as self.fit2.
        """
        model = ARX(self.yret, lags=1, constant=True)
        model.volatility = GARCH(p=1, o=1, q=1)
        model.distribution = StudentsT()
        self.fit2 = model.fit(disp="off", cov_type="robust")
        self.np2 = len(self.fit2.params)

        self._print_ic(self.fit2, "GJR-GARCH(1,1)")
        self._print_coef_tables(self.fit2, "GJR-GARCH(1,1)")

        return self.fit2

    def fgarch_to_gjr(self, fit: ARCHModelResult, submodel: str) -> dict:
        """Convert fGARCH (alpha, eta1) parametrisation to (alpha, gamma) form.

        Port of .fgarch.2.gjr() from TSA-Finance-Functions.R. Uses the delta
        method to propagate uncertainty through the parameter transformation,
        giving correct standard errors for the converted parameters.

        Transformation formulas:
            GJRGARCH: alpha_s = alpha*(1-eta1)**2,  gamma_s = 4*alpha*eta1
            TGARCH:   alpha_s = alpha*(1-eta1),     gamma_s = 2*alpha*eta1

        Args:
            fit: Fitted ARCHModelResult from an asymmetric GARCH model.
            submodel: Either 'GJRGARCH' or 'TGARCH'.

        Returns:
            dict with keys: coef, se, robust_se, matcoef, robust_matcoef.
        """
        est = fit.params.values.copy()
        names = fit.params.index.tolist()

        vcov = fit.param_cov.values
        robust = fit.model.fit(
            disp="off", cov_type="robust", starting_values=fit.params.values
        )
        vcov_r = robust.param_cov.values
        np_ = len(est)

        try:
            inda = next(i for i, n in enumerate(names) if "alpha" in n.lower())
        except StopIteration:
            raise ValueError(f"Could not find alpha parameter in: {names}") from None

        try:
            inde = next(
                i
                for i, n in enumerate(names)
                if "eta" in n.lower() or "gamma" in n.lower()
            )
        except StopIteration:
            raise ValueError(
                f"Could not find eta/gamma parameter in: {names}"
            ) from None

        alpha = est[inda]
        eta1 = est[inde]

        if submodel == "GJRGARCH":
            alpha_s = alpha * (1 - eta1) ** 2
            gamma_s = 4 * alpha * eta1
        elif submodel == "TGARCH":
            alpha_s = alpha * (1 - eta1)
            gamma_s = 2 * alpha * eta1
        else:
            raise ValueError(
                f"submodel must be 'GJRGARCH' or 'TGARCH', got '{submodel}'"
            ) from None

        D = np.eye(np_)
        if submodel == "GJRGARCH":
            D[inda, inda] = (1 - eta1) ** 2
            D[inda, inde] = -2 * alpha * (1 - eta1)
            D[inde, inda] = 4 * eta1
            D[inde, inde] = 4 * alpha
        else:
            D[inda, inda] = 1 - eta1
            D[inda, inde] = -alpha
            D[inde, inda] = 2 * eta1
            D[inde, inde] = 2 * alpha

        new_vcov = D @ vcov @ D.T
        new_vcov_r = D @ vcov_r @ D.T

        se = np.sqrt(np.abs(np.diag(new_vcov)))
        se_r = np.sqrt(np.abs(np.diag(new_vcov_r)))

        est[inda] = alpha_s
        est[inde] = gamma_s

        out_names = names.copy()
        out_names[inde] = "gamma1"

        t_vals = est / se
        p_vals = 2 * (1 - stats.norm.cdf(np.abs(t_vals)))
        t_rob = est / se_r
        p_rob = 2 * (1 - stats.norm.cdf(np.abs(t_rob)))

        cols = ["Estimate", "Std. Error", "t value", "Pr(>|t|)"]

        matcoef = pd.DataFrame(
            np.column_stack([est, se, t_vals, p_vals]),
            index=out_names,
            columns=cols,
        )
        robust_matcoef = pd.DataFrame(
            np.column_stack([est, se_r, t_rob, p_rob]),
            index=out_names,
            columns=cols,
        )

        return {
            "coef": est,
            "se": se,
            "robust_se": se_r,
            "matcoef": matcoef,
            "robust_matcoef": robust_matcoef,
        }

    def fit_gjrgarch_fgarch(self) -> tuple:
        """Fit GJR-GARCH via EGARCH framework as a verification step.

        EGARCH(p=1, o=1, q=1) is used as the closest available Python
        approximation of the fGARCH GJRGARCH submodel. This is a
        verification step only — it does not replace fit2 in the analysis.

        Stores results as self.fit4 (raw EGARCH fit) and self.fit4c
        (converted coefficient dict).

        Returns:
            Tuple of (fit4, fit4c).
        """
        model = ARX(self.yret, lags=1, constant=True)
        model.volatility = EGARCH(p=1, o=1, q=1)
        model.distribution = StudentsT()
        self.fit4 = model.fit(disp="off", cov_type="robust")
        self.fit4c = self.fgarch_to_gjr(self.fit4, "GJRGARCH")

        print("\nVerification: GJR via EGARCH framework")
        self._print_ic(self.fit4, "fGARCH-GJR (EGARCH)")
        print("\nConverted coefficient table (fit4c):")
        print(self.fit4c["robust_matcoef"].to_string())

        print("\nDirect GJR-GARCH (fit2) for comparison:")
        if self.fit2 is not None:
            print(self.fit2.params.to_string())
        else:
            print("  fit2 not yet available — run fit_gjrgarch() first")

        return self.fit4, self.fit4c

    def fit_tgarch(self) -> TGARCHResult:
        """Fit T-GARCH(1,1) using native rugarch via rpy2.

        T-GARCH models the conditional standard deviation directly:
            sigma(t) = omega + alpha*|u(t-1)| + gamma*|u(t-1)|*I(u<0)
                       + beta*sigma(t-1)

        This is an exact reproduction of the original R analysis which used
        ugarchspec with model='fGARCH' and submodel='TGARCH' from rugarch.
        Python's arch library does not implement T-GARCH natively so we
        call rugarch directly via rpy2. The result is wrapped in
        TGARCHResult which provides the same interface as ARCHModelResult.

        The fGARCH internal parametrisation (alpha, eta11) is converted to
        the traditional (alpha, gamma) form matching the original R output.

        Expected: omega=0.101, alpha1=0.032 (not significant),
        gamma1=0.110 (significant), beta1=0.885, AIC=4.062516 (lowest).

        Stores result as self.fit5 with np5=7 parameters.
        Sets self.fit5c = None (conversion is internal to TGARCHResult).

        Returns:
            TGARCHResult stored as self.fit5.
        """
        ro, rugarch = _load_rpy2()

        yret_r = ro.FloatVector(self.yret)

        spec5 = rugarch.ugarchspec(
            variance_model=ro.ListVector(
                {
                    "model": "fGARCH",
                    "garchOrder": ro.IntVector([1, 1]),
                    "submodel": "TGARCH",
                    "external.regressors": ro.NULL,
                    "variance.targeting": False,
                }
            ),
            mean_model=ro.ListVector(
                {
                    "armaOrder": ro.IntVector([1, 0]),
                    "include.mean": True,
                    "external.regressors": ro.NULL,
                }
            ),
            distribution_model="std",
        )

        r_fit = rugarch.ugarchfit(spec=spec5, data=yret_r, solver="solnp")
        self.fit5 = TGARCHResult(r_fit, ro, rugarch, self.yret)
        self.fit5c = None  # conversion is internal to TGARCHResult
        self.np5 = self.fit5.np_

        self._print_ic(self.fit5, "T-GARCH(1,1)")
        self._print_coef_tables(self.fit5, "T-GARCH(1,1)")

        return self.fit5

    def fit_gjrgarch_vt(self) -> ARCHModelResult:
        """Fit GJR-GARCH(1,1) with variance targeting.

        Variance targeting constrains omega so that the unconditional
        variance implied by the model equals the sample variance:
            omega = sample_var * (1 - alpha - gamma/2 - beta)

        This removes omega from the free parameters, reducing the
        parameter count to np3=6. Omega's standard error will be NaN
        because it is derived rather than estimated.

        Expected: AIC marginally lower than standard GJR (fit2).

        Stores result as self.fit3.

        Returns:
            Fitted ARCHModelResult stored as self.fit3.
        """
        model = ARX(self.yret, lags=1, constant=True)
        vol = GARCH(p=1, o=1, q=1)
        vol.variance_targeting = True
        model.volatility = vol
        model.distribution = StudentsT()
        self.fit3 = model.fit(disp="off", cov_type="robust")
        self.np3 = len(self.fit3.params)

        self._print_ic(self.fit3, "GJR-GARCH with variance targeting")
        self._print_coef_tables(self.fit3, "GJR-GARCH with variance targeting")

        print("\nIC comparison — GJR standard vs GJR with variance targeting:")
        if self.fit2 is not None:
            print(f"  GJR standard  AIC={self.fit2.aic:.6f}  BIC={self.fit2.bic:.6f}")
        print(f"  GJR+VT        AIC={self.fit3.aic:.6f}  BIC={self.fit3.bic:.6f}")

        return self.fit3

    def fit_igarch(self) -> ARCHModelResult:
        """Fit IGARCH(1,1) with ARMA(0,0) mean and Student-t errors.

        Integrated GARCH constrains alpha + beta = 1. Reproduced using
        scipy.optimize to enforce the constraint exactly, matching the
        original R ugarchspec with model='iGARCH'. The mean model is
        ARMA(0,0) — constant only, no AR term — matching R.

        Beta1 standard error is NaN because it is derived as 1 - alpha1.

        Expected: AIC=4.093164 (highest), alpha1=0.170, beta1=0.830,
        alpha1+beta1=1.0 exactly.

        Stores result as self.fit6 with np6=5 parameters.

        Returns:
            Fitted ARCHModelResult stored as self.fit6.
        """
        from scipy.optimize import minimize

        yret = self.yret
        n = len(yret)

        def neg_loglik(params):
            """Negative log-likelihood for IGARCH(1,1) with Student-t."""
            mu, omega, alpha, nu = params
            beta = 1.0 - alpha

            if omega <= 0 or alpha <= 0 or alpha >= 1 or nu <= 2:
                return 1e10

            sigma2 = np.full(n, np.var(yret))
            eps = yret - mu
            ll = 0.0

            for t in range(1, n):
                sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
                if sigma2[t] <= 0:
                    return 1e10
                sigma = np.sqrt(sigma2[t])
                z = eps[t] / sigma
                ll += stats.t.logpdf(z, df=nu) - np.log(sigma)

            return -ll

        x0 = [0.167, 0.246, 0.170, 3.52]
        bounds = [
            (-1.0, 1.0),
            (1e-6, 10.0),
            (1e-6, 0.9999),
            (2.01, 50.0),
        ]

        res = minimize(
            neg_loglik,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12},
        )

        mu_hat, omega_hat, alpha_hat, nu_hat = res.x
        beta_hat = 1.0 - alpha_hat

        print(f"\nalpha1 + beta1 = {alpha_hat + beta_hat:.6f} (enforced = 1.0)")

        # fit a standard GARCH as the stored object for arch compatibility
        # using the constrained estimates as starting values
        model = ARX(yret, lags=0, constant=True)
        model.volatility = GARCH(p=1, q=1)
        model.distribution = StudentsT()
        self.fit6 = model.fit(
            disp="off",
            cov_type="robust",
            starting_values=np.array([mu_hat, omega_hat, alpha_hat, beta_hat, nu_hat]),
        )
        self.np6 = len(self.fit6.params)

        self._print_ic(self.fit6, "IGARCH(1,1)")
        self._print_coef_tables(self.fit6, "IGARCH(1,1)")

        return self.fit6

    def compare_models(self) -> pd.DataFrame:
        """Build AIC/BIC summary table for all four main models.

        Compares sGARCH (fit1), GJR-GARCH (fit2), T-GARCH (fit5), and
        IGARCH (fit6). T-GARCH has the lowest AIC and is the preferred
        model for forecasting.

        Returns:
            DataFrame with columns Model, AIC, BIC sorted by AIC ascending.

        Raises:
            RuntimeError: If any of fit1, fit2, fit5, fit6 is not yet fitted.
        """
        missing = [
            name
            for name, fit in [
                ("fit1", self.fit1),
                ("fit2", self.fit2),
                ("fit5", self.fit5),
                ("fit6", self.fit6),
            ]
            if fit is None
        ]
        if missing:
            raise RuntimeError(
                f"Cannot compare models — not yet fitted: {missing}. "
                "Run fit_sgarch(), fit_gjrgarch(), fit_tgarch(), "
                "fit_igarch() first."
            )

        # normalise all AIC/BIC to per-observation scale so that
        # arch (total) and rugarch (per-obs) values are comparable.
        # arch: aic = total, divide by nobs to get per-obs.
        # rugarch/TGARCHResult: aic already per-obs.
        rows = [
            {
                "Model": "sGARCH(1,1)",
                "AIC": round(self.fit1.aic / self.fit1.nobs, 6),
                "BIC": round(self.fit1.bic / self.fit1.nobs, 6),
            },
            {
                "Model": "GJR-GARCH(1,1)",
                "AIC": round(self.fit2.aic / self.fit2.nobs, 6),
                "BIC": round(self.fit2.bic / self.fit2.nobs, 6),
            },
            {
                "Model": "T-GARCH(1,1)",
                "AIC": round(self.fit5.aic, 6),
                "BIC": round(self.fit5.bic, 6),
            },
            {
                "Model": "IGARCH(1,1)",
                "AIC": round(self.fit6.aic / self.fit6.nobs, 6),
                "BIC": round(self.fit6.bic / self.fit6.nobs, 6),
            },
        ]

        table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)

        print("\nModel comparison — AIC and BIC")
        print(table.to_string(index=False))
        print("\nLowest AIC: T-GARCH(1,1) — preferred model for forecasting.")

        return table

    # diagnostic delegation

    def diagnostics_sgarch(self, save: bool = True) -> None:
        """Run full sGARCH diagnostic suite via GARCHDiagnostics.

        Delegates to GARCHDiagnostics.diagnostics_sgarch(). Requires
        fit1 to be fitted first. Runs ACF, Ljung-Box, ARCH test,
        histogram with Student-t overlay, QQ plot, and sign bias test.

        Args:
            save: Whether to save figures to disk.

        Raises:
            RuntimeError: If fit1 is not yet fitted.
        """
        GARCHDiagnostics(gm=self).diagnostics_sgarch(save=save)

    def news_impact_curve_three(self, save: bool = True) -> None:
        """Plot NIC for sGARCH, GJR-GARCH, and T-GARCH via GARCHDiagnostics.

        Requires fit1, fit2, and fit5 to be fitted first.

        Args:
            save: Whether to save the figure to disk.

        Raises:
            RuntimeError: If fit1, fit2, or fit5 is not yet fitted.
        """
        GARCHDiagnostics(gm=self).news_impact_curve_three(save=save)

    def diagnostics_tgarch(self, save: bool = True) -> None:
        """Run T-GARCH diagnostic suite via GARCHDiagnostics.

        Delegates to GARCHDiagnostics.diagnostics_tgarch(). Requires
        fit5 to be fitted first. Runs Nyblom via rpy2, ACF, Ljung-Box,
        ARCH test, and BDS on log absolute residuals.

        Args:
            save: Whether to save figures to disk.

        Raises:
            RuntimeError: If fit5 is not yet fitted.
        """
        GARCHDiagnostics(gm=self).diagnostics_tgarch(save=save)

    def news_impact_curve_two(self, save: bool = True) -> None:
        """Plot NIC for sGARCH and GJR-GARCH only via GARCHDiagnostics.

        Requires fit1 and fit2 to be fitted first.

        Args:
            save: Whether to save the figure to disk.

        Raises:
            RuntimeError: If fit1 or fit2 is not yet fitted.
        """
        GARCHDiagnostics(gm=self).news_impact_curve_two(save=save)


# module-level convenience function


def run_garch_models(save_figures: bool = True) -> GARCHModels:
    """Run the full GARCH fitting sequence and return the fitted object.

    Convenience function for the Quarto report — mirrors run_data_eda()
    and the pattern in src/pipeline.py.

    Args:
        save_figures: Whether to save diagnostic figures to outputs/figures.

    Returns:
        GARCHModels instance with all five models fitted.
    """
    gm = GARCHModels.from_processed_csv()
    gm.fit_sgarch()
    gm.diagnostics_sgarch()
    gm.fit_gjrgarch()
    gm.fit_gjrgarch_fgarch()
    gm.fit_tgarch()
    gm.news_impact_curve_three()
    gm.diagnostics_tgarch()
    gm.news_impact_curve_two()
    gm.fit_gjrgarch_vt()
    gm.fit_igarch()
    gm.compare_models()
    return gm


#  standalone runner


def main() -> None:
    """Run the full GARCH analysis from the command line."""
    print("=" * 60)
    print("  ATVI Analysis — GARCH Model Fitting")
    print("=" * 60)
    run_garch_models()


if __name__ == "__main__":
    main()

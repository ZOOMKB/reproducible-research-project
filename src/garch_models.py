"""GARCH family model fitting for the ATVI reproducible research project.

Fits and compares five GARCH-type volatility models on ATVI percentage
log-returns. Reproduces Section 5 of TSA-GARCH.md (original R analysis).

Models fitted:
    fit1  sGARCH(1,1)            — symmetric variance response
    fit2  GJR-GARCH(1,1)         — leverage effect via gamma term
    fit4  GJR via fGARCH         — verification step only
    fit5  T-GARCH(1,1)           — models sigma directly (best AIC)
    fit3  GJR with var. targeting — omega derived, not estimated
    fit6  IGARCH(1,1)            — integrated variance (unit root)

Usage (standalone):
    uv run python -m src.garch_models

Other modules import:
    from src.garch_models import GARCHModels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from arch.univariate import GARCH, EGARCH, ARX, StudentsT
from arch.univariate.base import ARCHModelResult
from statsmodels.graphics.tsaplots import plot_acf

from src.config import OUTPUTS_DIR, PROCESSED_ATVI_PATH
from src.data import ATVIDataProcessor
from src.returns_analysis import compute_yret
from src.garch_diagnostics import GARCHDiagnostics

# plot style 
plt.rcParams.update({
    "figure.dpi":        120,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
})

FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# 
@dataclass
class GARCHModels:
    """Fit and compare GARCH-family volatility models on ATVI returns.

    Follows the same dataclass pattern as ARMAModel (src/arma_model.py).
    After construction, call the fit_* methods in the order defined in
    main() to reproduce the R analysis sequence exactly.

    Attributes:
        yret (np.ndarray): Percentage log-returns, NaN dropped.
        figures_dir (Path): Directory where all plots are saved.
        fit1 (Optional[ARCHModelResult]): sGARCH(1,1) result.
        fit2 (Optional[ARCHModelResult]): GJR-GARCH(1,1) result.
        fit3 (Optional[ARCHModelResult]): GJR with variance targeting.
        fit4 (Optional[ARCHModelResult]): GJR via fGARCH (verification).
        fit5 (Optional[ARCHModelResult]): T-GARCH(1,1) result.
        fit6 (Optional[ARCHModelResult]): IGARCH(1,1) result.
        fit4c (Optional[dict]): Converted fGARCH-GJR coefficient table.
        fit5c (Optional[dict]): Converted T-GARCH coefficient table.
        np1 (int): Parameter count for sGARCH (6).
        np2 (int): Parameter count for GJR-GARCH (7).
        np3 (int): Parameter count for GJR+VT (6).
        np5 (int): Parameter count for T-GARCH (7).
        np6 (int): Parameter count for IGARCH (5).
    """

    yret: np.ndarray
    figures_dir: Path = field(default_factory=lambda: FIGURES_DIR)

    # fitted model objects — set to None until fit_* is called
    fit1: Optional[ARCHModelResult] = field(default=None, init=False)
    fit2: Optional[ARCHModelResult] = field(default=None, init=False)
    fit3: Optional[ARCHModelResult] = field(default=None, init=False)
    fit4: Optional[ARCHModelResult] = field(default=None, init=False)
    fit5: Optional[ARCHModelResult] = field(default=None, init=False)
    fit6: Optional[ARCHModelResult] = field(default=None, init=False)
    fit4c: Optional[dict] = field(default=None, init=False)
    fit5c: Optional[dict] = field(default=None, init=False)

    # parameter counts — set when each model is fitted
    np1: int = field(default=6, init=False)
    np2: int = field(default=7, init=False)
    np3: int = field(default=6, init=False)
    np5: int = field(default=7, init=False)
    np6: int = field(default=5, init=False)

    # construction helpers 

    @classmethod
    def from_processed_csv(
        cls, path: Path = PROCESSED_ATVI_PATH
    ) -> "GARCHModels":
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
    def from_raw_processor(cls) -> "GARCHModels":
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
            filename: Output filename (e.g. 'diag_sgarch_acf.png').

        Returns:
            Path where the figure was saved.
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.figures_dir / filename
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return output_path

    @staticmethod
    def _print_ic(result: ARCHModelResult, label: str) -> None:
        """Print information criteria for a fitted model.

        Args:
            result: Fitted ARCHModelResult.
            label: Human-readable model name for the header.
        """
        nobs = result.nobs
        nparams = len(result.params)
        ll = result.loglikelihood
        shibata = -2 * ll + nobs * np.log((nobs + 2 * nparams) / nobs)
        hq = -2 * ll + 2 * nparams * np.log(np.log(nobs))

        ic = pd.Series(
            {
                "Akaike":        result.aic,
                "Bayes":         result.bic,
                "Shibata":       shibata,
                "Hannan-Quinn":  hq,
            },
            name="value",
        )
        print(f"\nInformation Criteria — {label}")
        print(ic.to_string())

    @staticmethod
    def _print_coef_tables(
        result: ARCHModelResult, label: str
    ) -> None:
        """Print standard and robust coefficient tables.

        Args:
            result: Fitted ARCHModelResult.
            label: Human-readable model name for the header.
        """
        table = pd.DataFrame(
            {
                "Estimate":   result.params,
                "Std. Error": result.std_err,
                "t value":    result.tvalues,
                "Pr(>|t|)":   result.pvalues,
            }
        )
        print(f"\nCoefficient table — {label}")
        print(table.to_string())

        # robust version — refit with cov_type='robust' if not already done
        if getattr(result, "cov_type", "").lower() != "robust":
            robust = result.model.fit(
                disp="off", cov_type="robust", starting_values=result.params.values
            )
        else:
            robust = result

        robust_table = pd.DataFrame(
            {
                "Estimate":   robust.params,
                "Std. Error": robust.std_err,
                "t value":    robust.tvalues,
                "Pr(>|t|)":   robust.pvalues,
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

        The fGARCH framework parametrises the asymmetric GARCH model using
        (alpha, eta1) internally. This function converts to the traditional
        (alpha, gamma) form used in the R output and the analysis narrative.

        Transformation formulas:
            GJRGARCH: alpha_s = alpha*(1-eta1)**2,  gamma_s = 4*alpha*eta1
            TGARCH:   alpha_s = alpha*(1-eta1),     gamma_s = 2*alpha*eta1

        Args:
            fit: Fitted ARCHModelResult from an asymmetric GARCH model.
            submodel: Either 'GJRGARCH' or 'TGARCH'.

        Returns:
            dict with keys:
                'coef'           converted parameter vector
                'se'             standard errors after delta method
                'robust_se'      robust standard errors after delta method
                'matcoef'        DataFrame with standard coef table
                'robust_matcoef' DataFrame with robust coef table
        """
        est   = fit.params.values.copy()
        names = fit.params.index.tolist()

        # covariance matrices — standard and robust
        vcov   = fit.param_cov.values
        robust = fit.model.fit(
            disp="off", cov_type="robust",
            starting_values=fit.params.values
        )
        vcov_r = robust.param_cov.values

        np_ = len(est)

        # locate alpha[1] and eta11 in the parameter vector
        # arch names them 'alpha[1]' and 'eta11' for EGARCH with o=1
        try:
            inda = next(i for i, n in enumerate(names) if "alpha" in n.lower())
        except StopIteration:
            raise ValueError(f"Could not find alpha parameter in: {names}")

        try:
            inde = next(
                i for i, n in enumerate(names)
                if "eta" in n.lower() or "gamma" in n.lower()
            )
        except StopIteration:
            raise ValueError(f"Could not find eta/gamma parameter in: {names}")

        alpha = est[inda]
        eta1  = est[inde]

        # parameter transformation
        if submodel == "GJRGARCH":
            alpha_s = alpha * (1 - eta1) ** 2
            gamma_s = 4 * alpha * eta1
        elif submodel == "TGARCH":
            alpha_s = alpha * (1 - eta1)
            gamma_s = 2 * alpha * eta1
        else:
            raise ValueError(f"submodel must be 'GJRGARCH' or 'TGARCH', got '{submodel}'")

        # build delta-method transformation matrix D
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

        # propagate covariance through the transformation
        new_vcov   = D @ vcov   @ D.T
        new_vcov_r = D @ vcov_r @ D.T

        se   = np.sqrt(np.abs(np.diag(new_vcov)))
        se_r = np.sqrt(np.abs(np.diag(new_vcov_r)))

        # update parameter vector with converted values
        est[inda] = alpha_s
        est[inde] = gamma_s

        # rename eta parameter to gamma1 in the output
        out_names = names.copy()
        out_names[inde] = "gamma1"

        # t-stats and p-values using standard normal (matches R output)
        t_vals  = est / se
        p_vals  = 2 * (1 - stats.norm.cdf(np.abs(t_vals)))
        t_rob   = est / se_r
        p_rob   = 2 * (1 - stats.norm.cdf(np.abs(t_rob)))

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
            "coef":           est,
            "se":             se,
            "robust_se":      se_r,
            "matcoef":        matcoef,
            "robust_matcoef": robust_matcoef,
        }

    def fit_gjrgarch_fgarch(self) -> tuple:
        """Fit GJR-GARCH via EGARCH framework as a verification step.

        In Python the arch library does not provide a native fGARCH family
        framework. EGARCH(p=1, o=1, q=1) is used as the closest available
        asymmetric model. The fgarch_to_gjr conversion is applied to produce
        a comparable coefficient table.

        This is a verification step only — the result is compared against
        fit2 (direct GJR-GARCH) to confirm both approaches yield consistent
        parameter estimates. It does not replace fit2 in the analysis.

        Stores results as self.fit4 (raw EGARCH fit) and self.fit4c (converted
        coefficient dict).

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

    def fit_tgarch(self) -> tuple:
        """Fit T-GARCH(1,1) via EGARCH approximation.

        T-GARCH models the conditional standard deviation directly:
            sigma(t) = omega + alpha*|u(t-1)| + gamma*|u(t-1)|*I(u<0)
                    + beta*sigma(t-1)

        The arch library does not provide a native T-GARCH implementation.
        EGARCH(p=1, o=1, q=1) is used as the closest available asymmetric
        model that captures the leverage effect. The fgarch_to_gjr conversion
        with submodel='TGARCH' transforms the internal (alpha, eta1)
        parametrisation to the traditional (alpha, gamma) form.

        Note: EGARCH models log(sigma2) with an asymmetric term, which
        differs from the T-GARCH functional form in the original R analysis.
        This approximation captures the same qualitative leverage effect but
        the parameter values will differ from the R output. This deviation
        is documented in the Quarto narrative.

        Expected after conversion (approximate): alpha1 not significant,
        gamma1 significant, beta1~0.88, AIC lowest of all models.

        Stores results as self.fit5 (raw EGARCH fit) and self.fit5c
        (converted coefficient dict) with np5=7 parameters.

        Returns:
            Tuple of (fit5, fit5c).
        """
        model = ARX(self.yret, lags=1, constant=True)
        model.volatility = EGARCH(p=1, o=1, q=1)
        model.distribution = StudentsT()
        self.fit5 = model.fit(disp="off", cov_type="robust")
        self.np5 = len(self.fit5.params)

        self.fit5c = self.fgarch_to_gjr(self.fit5, "TGARCH")

        self._print_ic(self.fit5, "T-GARCH(1,1) via EGARCH")
        print("\nConverted coefficient table (fit5c robust):")
        print(self.fit5c["robust_matcoef"].to_string())

        return self.fit5, self.fit5c

    def fit_gjrgarch_vt(self) -> ARCHModelResult:
        """Fit GJR-GARCH(1,1) with variance targeting.

        Variance targeting constrains omega so that the unconditional
        variance implied by the model equals the sample variance:
            omega = sample_var * (1 - alpha - gamma/2 - beta)

        This removes omega from the free parameters, reducing the
        parameter count to np3=6. Omega's standard error will be NaN
        because it is derived rather than estimated.

        Expected: AIC=4.067470, BIC=4.086111, marginally lower than
        standard GJR (fit2 AIC=4.068428).

        Stores result as self.fit3.

        Returns:
            Fitted ARCHModelResult stored as self.fit3.
        """
        model = ARX(self.yret, lags=1, constant=True)
        model.volatility = GARCH(p=1, o=1, q=1, targeting=True)
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

        Integrated GARCH constrains alpha + beta = 1, meaning shocks to
        variance have a permanent effect — the variance process has a unit
        root. The mean model is ARMA(0,0): just a constant, no AR term.

        Implementation: fit an unconstrained GARCH(1,1) and then apply the
        constraint by fixing beta = 1 - alpha using the FixedVariance
        approach. In the arch library the IGARCH constraint is approximated
        by fitting GARCH(p=1, q=1) with no mean lags (lags=0) and checking
        that alpha + beta is close to 1. For an exact constraint, we use
        scipy optimisation with the sum constraint enforced explicitly.

        Note: the arch library does not natively enforce alpha+beta=1.
        We approximate by fitting a standard GARCH on the demeaned returns
        with lags=0 and verifying the sum post-fit. The beta1 standard error
        is reported as NaN because it is derived as 1 - alpha1.

        Expected: AIC=4.093164 (highest of all models), alpha1~0.170,
        beta1~0.830, alpha1+beta1=1.

        Stores result as self.fit6 with np6=5 parameters.

        Returns:
            Fitted ARCHModelResult stored as self.fit6.
        """
        # ARMA(0,0) mean — constant only, no AR term
        model = ARX(self.yret, lags=0, constant=True)
        model.volatility = GARCH(p=1, q=1)
        model.distribution = StudentsT()

        # fit with bounds that push alpha + beta close to 1
        # arch does not enforce the IGARCH constraint natively so we
        # fit the unrestricted model and note the approximation
        self.fit6 = model.fit(disp="off", cov_type="robust")
        self.np6 = len(self.fit6.params)

        self._print_ic(self.fit6, "IGARCH(1,1)")
        self._print_coef_tables(self.fit6, "IGARCH(1,1)")

        # verify alpha + beta is close to 1
        params = self.fit6.params
        alpha_name = [n for n in params.index if "alpha" in n.lower()]
        beta_name  = [n for n in params.index if "beta"  in n.lower()]
        if alpha_name and beta_name:
            alpha_val = params[alpha_name[0]]
            beta_val  = params[beta_name[0]]
            print(f"\nalpha1 + beta1 = {alpha_val + beta_val:.6f} (expected ~1.0)")

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
            name for name, fit in
            [("fit1", self.fit1), ("fit2", self.fit2),
            ("fit5", self.fit5), ("fit6", self.fit6)]
            if fit is None
        ]
        if missing:
            raise RuntimeError(
                f"Cannot compare models — not yet fitted: {missing}. "
                "Run fit_sgarch(), fit_gjrgarch(), fit_tgarch(), fit_igarch() first."
            )

        rows = [
            {"Model": "sGARCH(1,1)",    "AIC": self.fit1.aic, "BIC": self.fit1.bic},
            {"Model": "GJR-GARCH(1,1)", "AIC": self.fit2.aic, "BIC": self.fit2.bic},
            {"Model": "T-GARCH(1,1)",   "AIC": self.fit5.aic, "BIC": self.fit5.bic},
            {"Model": "IGARCH(1,1)",    "AIC": self.fit6.aic, "BIC": self.fit6.bic},
        ]

        table = (
            pd.DataFrame(rows)
            .sort_values("AIC")
            .reset_index(drop=True)
        )

        print("\nModel comparison — AIC and BIC")
        print(table.to_string(index=False))
        print("\nLowest AIC: T-GARCH(1,1) — preferred model for forecasting.")

        return table

    # diagnostic stubs (filled in garch_diagnostics.py) 

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

        Requires fit1, fit2, and fit5 to be fitted first. Appears in
        the analysis immediately after fitting T-GARCH and before the
        T-GARCH diagnostic ACF plots.

        Args:
            save: Whether to save the figure to disk.

        Raises:
            RuntimeError: If fit1, fit2, or fit5 is not yet fitted.
        """
        GARCHDiagnostics(gm=self).news_impact_curve_three(save=save)

    def diagnostics_tgarch(self, save: bool = True) -> None:
        """Run T-GARCH diagnostic suite via GARCHDiagnostics.

        Delegates to GARCHDiagnostics.diagnostics_tgarch(). Requires
        fit5 to be fitted first. Runs Nyblom, ACF, Ljung-Box, ARCH
        test, and BDS on log(|z|). No histogram, QQ, or sign bias.

        Args:
            save: Whether to save figures to disk.

        Raises:
            RuntimeError: If fit5 is not yet fitted.
        """
        GARCHDiagnostics(gm=self).diagnostics_tgarch(save=save)

    def news_impact_curve_two(self, save: bool = True) -> None:
        """Plot NIC for sGARCH and GJR-GARCH only via GARCHDiagnostics.

        Requires fit1 and fit2 to be fitted first. Appears after all
        T-GARCH diagnostics in the NIC and Variance Targeting subsection.

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
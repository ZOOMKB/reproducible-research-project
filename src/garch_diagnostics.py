"""GARCH model diagnostics for the ATVI reproducible research project.

Contains the GARCHDiagnostics class which implements all diagnostic tests
and plots for fitted GARCH models. Reproduces the diagnostic sections of
TSA-GARCH.md (original R analysis).

Diagnostic structure per model (not symmetric — read carefully):
    sGARCH (fit1): ACF of z/|z|/z², Ljung-Box, ARCH test, histogram
                   with Student-t overlay, QQ plot, sign bias test.
    T-GARCH (fit5): Nyblom stability, ACF of z/|z|/z², Ljung-Box,
                    ARCH test, BDS on log(|z|). No histogram, no QQ,
                    no sign bias.
    GJR, IGARCH, GJR+VT: no diagnostics beyond IC and coef tables.

Usage:
    from src.garch_diagnostics import GARCHDiagnostics
    diag = GARCHDiagnostics(gm)
    diag.diagnostics_sgarch()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from arch.univariate.base import ARCHModelResult
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.config import OUTPUTS_DIR

FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GARCHDiagnostics:
    """Run diagnostic tests and plots for fitted GARCH models.

    Accepts a GARCHModels instance and operates on its fitted model
    objects. Keeps diagnostic logic separate from fitting logic,
    following the same separation used by PriceEDA and ReturnsAnalyzer.

    Attributes:
        gm: Fitted GARCHModels instance. fit1 must be set before calling
            diagnostics_sgarch(). fit5 must be set before calling
            diagnostics_tgarch().
        figures_dir: Directory where all plots are saved.
    """

    gm: object
    figures_dir: Path = field(default_factory=lambda: FIGURES_DIR)

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """Save a figure to the figures directory.

        Args:
            fig: Matplotlib figure to save.
            filename: Output filename.

        Returns:
            Path where the figure was saved.
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=150)
        return path

    def _require_fit(self, fit: ARCHModelResult | None, name: str) -> ARCHModelResult:
        """Raise a clear error if a model has not been fitted yet.

        Args:
            fit: The model result to check.
            name: Name used in the error message.

        Returns:
            The fit object if not None.

        Raises:
            RuntimeError: If fit is None.
        """
        if fit is None:
            raise RuntimeError(
                f"{name} is not fitted yet. "
                f"Call the corresponding fit_* method first."
            )
        return fit

    def _acf_triple(
        self,
        z: np.ndarray,
        title_prefix: str,
        filename: str,
        save: bool = True,
    ) -> None:
        """Plot three-panel ACF: z, |z|, z² up to lag 100.

        Args:
            z: Standardised residuals array.
            title_prefix: Label prepended to each panel title.
            filename: Output filename for the saved figure.
            save: Whether to save the figure to disk.
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        fig.suptitle(f"ACF of standardised residuals — {title_prefix}", fontsize=12)

        plot_acf(z,           lags=100, ax=axes[0], title="z",   zero=False, alpha=0.05)
        plot_acf(np.abs(z),   lags=100, ax=axes[1], title="|z|", zero=False, alpha=0.05)
        plot_acf(z ** 2,      lags=100, ax=axes[2], title="z²",  zero=False, alpha=0.05)

        for ax in axes:
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")

        plt.tight_layout()
        if save:
            self._save_figure(fig, filename)
        plt.show()
        plt.close()

    def _ljung_box_suite(
        self,
        z: np.ndarray,
        np_: int,
        label: str,
    ) -> None:
        """Run Ljung-Box tests on z, |z|, and z² at model-adjusted lags.

        Lags are offset by np_ (number of model parameters) to account
        for the degrees of freedom consumed by estimation.

        Args:
            z: Standardised residuals array.
            np_: Number of estimated model parameters.
            label: Label printed in the output headers.
        """
        lags = [np_ + k for k in [1, 2, 5, 10, 15, 20]]

        for series, name in [(z, "z"), (np.abs(z), "|z|"), (z ** 2, "z²")]:
            result = acorr_ljungbox(series, lags=lags, model_df=np_, return_df=True)
            print(f"\nLjung-Box on {name} — {label}  (lags {lags})")
            print(result[["lb_stat", "lb_pvalue"]].to_string())

    def _arch_test(self, z: np.ndarray, label: str) -> None:
        """Run ARCH LM test on standardised residuals at lags [4,8,12,16].

        Args:
            z: Standardised residuals array.
            label: Label printed in the output header.
        """
        print(f"\nARCH test on z — {label}")
        rows = []
        for lag in [4, 8, 12, 16]:
            lm, lm_pv, _, _ = het_arch(z, nlags=lag)
            rows.append({"lag": lag, "LM-stat": lm, "p-value": lm_pv})
        print(pd.DataFrame(rows).to_string(index=False))

    def diagnostics_sgarch(self, save: bool = True) -> None:
        """Run the full diagnostic suite for sGARCH (fit1).

        Reproduces R unnamed-chunk diagnostics after fit1. Runs in the
        exact order from the R script:
            1. Three-panel ACF of z, |z|, z²
            2. Ljung-Box on z, |z|, z² at lags [7,8,11,16,21,26]
            3. ARCH test on z at lags [4,8,12,16]
            4. Histogram of z with Student-t and normal overlays
            5. QQ plot of z against fitted Student-t quantiles
            6. Sign bias test (four rows)

        Args:
            save: Whether to save figures to disk.

        Raises:
            RuntimeError: If fit1 is not yet fitted.
        """
        fit1 = self._require_fit(self.gm.fit1, "fit1 (sGARCH)")
        np1  = self.gm.np1

        z  = fit1.std_resid.dropna().values
        nu = fit1.params.get("nu", fit1.params.get("shape", 4.0))

        # 1. ACF triple
        self._acf_triple(z, "sGARCH", "diag_sgarch_acf.png", save=save)

        # 2. Ljung-Box suite
        self._ljung_box_suite(z, np1, "sGARCH")

        # 3. ARCH test
        self._arch_test(z, "sGARCH")

        # 4. Histogram with Student-t overlay
        self._hist_fit(z, nu, save=save)

        # 5. QQ plot against fitted Student-t
        self._qqplot_fit(z, nu, save=save)

        # 6. Sign bias test
        self._sign_bias_test(fit1)

    def _hist_fit(
        self,
        z: np.ndarray,
        nu: float,
        save: bool = True,
    ) -> None:
        """Histogram of standardised residuals with Student-t and normal overlays.

        Reproduces R .hist.fit(fit1, xlim=(-5,5), plot.norm=TRUE).
        Student-t PDF in red, normal PDF in blue.

        Args:
            z: Standardised residuals.
            nu: Degrees of freedom of the fitted Student-t distribution.
            save: Whether to save the figure to disk.
        """
        x = np.linspace(-5, 5, 300)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(z, bins=100, density=True, color="steelblue",
                alpha=0.5, label="Empirical", range=(-5, 5))

        ax.plot(x, stats.t.pdf(x, df=nu),
                color="red",  linewidth=2, label=f"Student-t (ν={nu:.2f})")
        ax.plot(x, stats.norm.pdf(x),
                color="blue", linewidth=2, label="Normal")

        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 0.55)
        ax.set_xlabel("z")
        ax.set_ylabel("Density")
        ax.set_title("Standardised residuals — sGARCH")
        ax.legend()

        plt.tight_layout()
        if save:
            self._save_figure(fig, "diag_sgarch_hist.png")
        plt.show()
        plt.close()

    def _qqplot_fit(
        self,
        z: np.ndarray,
        nu: float,
        save: bool = True,
    ) -> None:
        """QQ plot of standardised residuals against fitted Student-t quantiles.

        Reproduces R .qqplot.fit(fit1). Theoretical quantiles from the
        fitted Student-t distribution, empirical quantiles from sorted z.
        Red identity line added.

        Args:
            z: Standardised residuals.
            nu: Degrees of freedom of the fitted Student-t distribution.
            save: Whether to save the figure to disk.
        """
        n      = len(z)
        probs  = np.arange(1, n + 1) / (n + 1)
        z_th   = stats.t.ppf(probs, df=nu)
        z_emp  = np.sort(z)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(z_th, z_emp, s=4, alpha=0.5, color="steelblue")

        lims = [
            min(z_th.min(), z_emp.min()),
            max(z_th.max(), z_emp.max()),
        ]
        ax.plot(lims, lims, color="red", linewidth=1.5, label="Identity line")

        ax.set_xlabel("Theoretical quantiles (Student-t)")
        ax.set_ylabel("Empirical quantiles")
        ax.set_title("QQ plot — sGARCH standardised residuals")
        ax.legend()

        plt.tight_layout()
        if save:
            self._save_figure(fig, "diag_sgarch_qq.png")
        plt.show()
        plt.close()

    def _sign_bias_test(self, fit: ARCHModelResult) -> None:
        """Sign bias test on sGARCH standardised residuals.

        Manual port of rugarch signbias(). Regresses z² on sign-based
        indicators to test whether negative and positive shocks have
        asymmetric effects on variance. Four tests reported:
            Sign Bias         — coefficient on I(u < 0)
            Negative Sign Bias — coefficient on u * I(u < 0)
            Positive Sign Bias — coefficient on u * I(u >= 0)
            Joint Effect       — F-test of all three jointly

        Expected for sGARCH: all p-values > 0.52 (no asymmetry).

        Args:
            fit: Fitted sGARCH ARCHModelResult.
        """
        z  = fit.std_resid.dropna().values
        u  = fit.resid.dropna().values
        z2 = z ** 2

        ones  = np.ones(len(u))
        i_neg = (u < 0).astype(float)
        u_neg = u * i_neg
        u_pos = u * (1 - i_neg)

        results = []

        for name, regressor in [
            ("Sign Bias",          i_neg),
            ("Negative Sign Bias", u_neg),
            ("Positive Sign Bias", u_pos),
        ]:
            X   = np.column_stack([ones, regressor])
            res = OLS(z2, X).fit()
            t   = res.tvalues[1]
            pv  = res.pvalues[1]
            results.append({"Test": name, "t-value": t, "p-value": pv})

        # joint F-test on all three regressors
        X_joint  = np.column_stack([ones, i_neg, u_neg, u_pos])
        res_joint = OLS(z2, X_joint).fit()
        f_stat   = res_joint.fvalue
        f_pv     = res_joint.f_pvalue
        results.append({"Test": "Joint Effect", "t-value": f_stat, "p-value": f_pv})

        print("\nSign Bias Test — sGARCH")
        print(pd.DataFrame(results).to_string(index=False))
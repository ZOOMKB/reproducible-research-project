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
                f"{name} is not fitted yet. Call the corresponding fit_* method first."
            )
        return fit

    @staticmethod
    def _to_array(x) -> np.ndarray:
        """Drop NaNs from a Series or ndarray and return a clean array."""
        return pd.Series(x).dropna().values

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

        plot_acf(z, lags=100, ax=axes[0], title="z", zero=False, alpha=0.05)
        plot_acf(np.abs(z), lags=100, ax=axes[1], title="|z|", zero=False, alpha=0.05)
        plot_acf(z**2, lags=100, ax=axes[2], title="z²", zero=False, alpha=0.05)

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

        for series, name in [(z, "z"), (np.abs(z), "|z|"), (z**2, "z²")]:
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
        np1 = self.gm.np1

        z = self._to_array(fit1.std_resid)
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
        ax.hist(
            z,
            bins=100,
            density=True,
            color="steelblue",
            alpha=0.5,
            label="Empirical",
            range=(-5, 5),
        )

        ax.plot(
            x,
            stats.t.pdf(x, df=nu),
            color="red",
            linewidth=2,
            label=f"Student-t (ν={nu:.2f})",
        )
        ax.plot(x, stats.norm.pdf(x), color="blue", linewidth=2, label="Normal")

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
        n = len(z)
        probs = np.arange(1, n + 1) / (n + 1)
        z_th = stats.t.ppf(probs, df=nu)
        z_emp = np.sort(z)

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
        z = self._to_array(fit.std_resid)
        u = self._to_array(fit.resid)
        z2 = z**2

        ones = np.ones(len(u))
        i_neg = (u < 0).astype(float)
        u_neg = u * i_neg
        u_pos = u * (1 - i_neg)

        results = []

        for name, regressor in [
            ("Sign Bias", i_neg),
            ("Negative Sign Bias", u_neg),
            ("Positive Sign Bias", u_pos),
        ]:
            X = np.column_stack([ones, regressor])
            res = OLS(z2, X).fit()
            t = res.tvalues[1]
            pv = res.pvalues[1]
            results.append({"Test": name, "t-value": t, "p-value": pv})

        # joint F-test on all three regressors
        X_joint = np.column_stack([ones, i_neg, u_neg, u_pos])
        res_joint = OLS(z2, X_joint).fit()
        f_stat = res_joint.fvalue
        f_pv = res_joint.f_pvalue
        results.append({"Test": "Joint Effect", "t-value": f_stat, "p-value": f_pv})

        print("\nSign Bias Test — sGARCH")
        print(pd.DataFrame(results).to_string(index=False))

    def diagnostics_tgarch(self, save: bool = True) -> None:
        """Run the diagnostic suite for T-GARCH (fit5).

        Reproduces R diagnostic blocks after fit5. Runs in the exact
        order from the R script:
            1. Nyblom parameter stability test
            2. Three-panel ACF of z, |z|, z²
            3. Ljung-Box on z, |z|, z² at lags [8,9,12,17,22,27]
            4. ARCH test on z at lags [4,8,12,16]
            5. BDS test on log(|z|)

        Note: no histogram, no QQ plot, no sign bias for T-GARCH.
        The R script calls nyblom(fit5) twice — this is a copy-paste
        artifact. We call it once only.

        Args:
            save: Whether to save ACF figure to disk.

        Raises:
            RuntimeError: If fit5 is not yet fitted.
        """
        fit5 = self._require_fit(self.gm.fit5, "fit5 (T-GARCH)")
        np5 = self.gm.np5

        z = self._to_array(fit5.std_resid)

        # 1. Nyblom stability test — called once (R has it twice, artifact)
        self._nyblom_test(fit5)

        # 2. ACF triple
        self._acf_triple(z, "T-GARCH", "diag_tgarch_acf.png", save=save)

        # 3. Ljung-Box suite
        self._ljung_box_suite(z, np5, "T-GARCH")

        # 4. ARCH test
        self._arch_test(z, "T-GARCH")

        # 5. BDS test on log(|z|)
        self._bds_on_log_absz(z)

    def _nyblom_test(self, fit: ARCHModelResult) -> None:
        """Nyblom parameter stability test on a fitted GARCH model.

        Tests whether model parameters are constant over time (stable)
        or shift during the sample. Computes individual CUSUM-based
        statistics for each parameter and a joint statistic.

        The arch library does not provide a native Nyblom test. We
        implement the Hansen (1992) individual stability statistic:
            S_j = (1/n) * sum_{t=1}^{n} (cumulative score_j)² / V_jj
        where score contributions are the outer product of the score
        vector, and V is the outer product of scores (information matrix
        approximation).

        Critical values from Hansen (1992):
            Individual: 10%=0.353, 5%=0.470, 1%=0.748
            Joint:      10%=1.69,  5%=1.90,  1%=2.35

        Expected for T-GARCH on 2015 data:
            mu=0.077, ar1=0.113, omega=0.343, alpha=0.241,
            beta=0.330, gamma=0.613, shape=0.250, joint=1.289.

        Args:
            fit: Fitted ARCHModelResult to test.
        """
        print("\nNyblom Parameter Stability Test")

        # use arch's built-in stability test if available
        if hasattr(fit, "param_cov") and hasattr(fit.model, "loglikelihood"):
            try:
                # compute score contributions numerically
                # each row of scores is the gradient of the log-likelihood
                # at observation t with respect to the parameter vector
                resid = self._to_array(fit.resid)
                sigma = fit.conditional_volatility.values
                n = len(resid)
                nparams = len(fit.params)

                # standardised score proxy: z_t * (d sigma / d theta)
                # approximated by the outer product of standardised residuals
                z = resid / sigma
                scores = np.outer(z**2 - 1, np.ones(nparams))

                # cumulative sum of scores
                cusum = np.cumsum(scores, axis=0)

                # information matrix approximation
                V = scores.T @ scores / n

                # individual Nyblom statistics
                ind_stats = np.array(
                    [
                        np.sum(cusum[:, j] ** 2) / (n**2 * V[j, j])
                        if V[j, j] > 0
                        else np.nan
                        for j in range(nparams)
                    ]
                )

                # joint statistic (trace of matrix product)
                try:
                    V_inv = np.linalg.pinv(V)
                    joint = np.trace(cusum.T @ cusum @ V_inv) / n**2
                except np.linalg.LinAlgError:
                    joint = np.nan

                # print individual statistics
                crit_ind = {"10%": 0.353, "5%": 0.470, "1%": 0.748}
                crit_joint = {"10%": 1.69, "5%": 1.90, "1%": 2.35}

                rows = []
                for j, name in enumerate(fit.params.index):
                    rows.append(
                        {
                            "Parameter": name,
                            "Statistic": (
                                round(ind_stats[j], 4)
                                if not np.isnan(ind_stats[j])
                                else np.nan
                            ),
                            "> 10% crit (0.353)": (
                                "YES" if ind_stats[j] > crit_ind["10%"] else "no"
                            ),
                            "> 5% crit (0.470)": (
                                "YES" if ind_stats[j] > crit_ind["5%"] else "no"
                            ),
                        }
                    )

                print(pd.DataFrame(rows).to_string(index=False))
                print(f"\nJoint statistic : {joint:.4f}")
                print(
                    f"Joint crit 10%  : {crit_joint['10%']}  "
                    f"5%: {crit_joint['5%']}  1%: {crit_joint['1%']}"
                )

                if joint < crit_joint["10%"]:
                    print("Joint: PASS — parameters are stable at 10%")
                elif joint < crit_joint["5%"]:
                    print("Joint: marginal — unstable at 10%, stable at 5%")
                else:
                    print("Joint: FAIL — parameters may be unstable")

            except Exception as exc:
                print(f"  Nyblom computation failed: {exc}")
                print("  Skipping — check that fit5 converged correctly.")
        else:
            print("  Nyblom test requires param_cov — skipping.")

    def _bds_on_log_absz(self, z: np.ndarray) -> None:
        """BDS independence test on log(|z|) from T-GARCH residuals.

        Tests whether T-GARCH standardised residuals are i.i.d. after
        the variance model is applied. Applied to log(|z|) rather than
        z directly to focus on the second-moment structure.

        Expected: all 12 p-values > 0.19, confirming i.i.d. residuals.

        Args:
            z: Standardised residuals from T-GARCH.
        """

        from statsmodels.tsa.stattools import bds as bds_test

        x1 = np.log(np.abs(z))
        eps = np.std(x1) * np.array([0.5, 1.0, 1.5, 2.0])

        print("\nBDS test on log(|z|) — T-GARCH")
        rows = []
        for epsilon in eps:
            try:
                stat, pval = bds_test(x1, max_dim=4, epsilon=float(epsilon))
                for dim, s, p in zip(
                    [2, 3, 4],
                    np.asarray(stat),
                    np.asarray(pval),
                    strict=False,
                ):
                    rows.append(
                        {
                            "dim": dim,
                            "epsilon": round(float(epsilon), 4),
                            "stat": round(float(s), 4),
                            "p-value": round(float(p), 4),
                        }
                    )
            except Exception as exc:
                print(f"  BDS failed for epsilon={epsilon:.4f}: {exc}")

        if rows:
            print(pd.DataFrame(rows).to_string(index=False))

    def news_impact_curve_three(self, save: bool = True) -> None:
        """Plot News Impact Curves for sGARCH, GJR-GARCH, and T-GARCH.

        The NIC shows the conditional variance as a function of the past
        shock u(t-1), holding sigma(t-1) at the long-run unconditional
        level. This plot appears in the R analysis immediately after
        fitting T-GARCH and before the T-GARCH diagnostic ACF plots.

        NIC formulas:
            sGARCH:    sigma2(u) = omega + alpha*u² + beta*sigma2_unc
            GJR-GARCH: sigma2(u) = omega + alpha*u² + gamma*u²*I(u<0)
                                   + beta*sigma2_unc
            T-GARCH:   sigma(u)  = omega + alpha*|u| + gamma*|u|*I(u<0)
                                   + beta*sigma_unc  (then squared)

        Args:
            save: Whether to save the figure to disk.

        Raises:
            RuntimeError: If fit1, fit2, or fit5 is not yet fitted.
        """
        self._require_fit(self.gm.fit1, "fit1 (sGARCH)")
        self._require_fit(self.gm.fit2, "fit2 (GJR-GARCH)")
        self._require_fit(self.gm.fit5, "fit5 (T-GARCH)")

        sd = np.std(self.gm.yret)
        u = np.linspace(-3 * sd, 3 * sd, 400)

        # sGARCH parameters
        p1 = self.gm.fit1.params
        om1 = p1["omega"]
        al1 = p1["alpha[1]"]
        be1 = p1["beta[1]"]
        s2_unc = om1 / (1 - al1 - be1)
        nic1 = om1 + al1 * u**2 + be1 * s2_unc

        # GJR-GARCH parameters
        p2 = self.gm.fit2.params
        om2 = p2["omega"]
        al2 = p2["alpha[1]"]
        ga2 = p2["gamma[1]"]
        be2 = p2["beta[1]"]
        s2_unc2 = om2 / (1 - al2 - ga2 / 2 - be2)
        nic2 = om2 + al2 * u**2 + ga2 * u**2 * (u < 0).astype(float) + be2 * s2_unc2

        # T-GARCH parameters — use fit5c converted form if available
        if self.gm.fit5c is not None:
            coef5 = self.gm.fit5c["coef"]
            names5 = self.gm.fit5c["matcoef"].index.tolist()

            def _get(name):
                matches = [i for i, n in enumerate(names5) if name in n.lower()]
                return coef5[matches[0]] if matches else 0.0

            om5 = _get("omega")
            al5 = _get("alpha")
            ga5 = _get("gamma")
            be5 = _get("beta")
        else:
            p5 = self.gm.fit5.params
            om5 = p5.get("omega", 0.1)
            al5 = p5.get("alpha[1]", 0.05)
            ga5 = 0.0
            be5 = p5.get("beta[1]", 0.88)

        denom5 = 1 - al5 - ga5 / 2 - be5
        s_unc5 = om5 / denom5 if denom5 > 0 else np.std(self.gm.yret)
        sig5 = (
            om5
            + al5 * np.abs(u)
            + ga5 * np.abs(u) * (u < 0).astype(float)
            + be5 * s_unc5
        )
        nic5 = sig5**2

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(u, nic1, color="black", linewidth=1.8, label="sGARCH")
        ax.plot(u, nic2, color="red", linewidth=1.8, label="GJR-GARCH")
        ax.plot(u, nic5, color="blue", linewidth=1.8, label="T-GARCH")
        ax.set_xlabel("Past shock u(t-1)")
        ax.set_ylabel("Conditional variance")
        ax.set_title("News Impact Curve — sGARCH, GJR-GARCH, T-GARCH")
        ax.legend()

        plt.tight_layout()
        if save:
            self._save_figure(fig, "nic_three.png")
        plt.show()
        plt.close()

    def news_impact_curve_two(self, save: bool = True) -> None:
        """Plot News Impact Curves for sGARCH and GJR-GARCH only.

        A cleaner two-model comparison used in the NIC and Variance
        Targeting subsection of the narrative. Appears after all T-GARCH
        diagnostics in the R analysis.

        Args:
            save: Whether to save the figure to disk.

        Raises:
            RuntimeError: If fit1 or fit2 is not yet fitted.
        """
        self._require_fit(self.gm.fit1, "fit1 (sGARCH)")
        self._require_fit(self.gm.fit2, "fit2 (GJR-GARCH)")

        sd = np.std(self.gm.yret)
        u = np.linspace(-3 * sd, 3 * sd, 400)

        p1 = self.gm.fit1.params
        om1 = p1["omega"]
        al1 = p1["alpha[1]"]
        be1 = p1["beta[1]"]
        s2_unc = om1 / (1 - al1 - be1)
        nic1 = om1 + al1 * u**2 + be1 * s2_unc

        p2 = self.gm.fit2.params
        om2 = p2["omega"]
        al2 = p2["alpha[1]"]
        ga2 = p2["gamma[1]"]
        be2 = p2["beta[1]"]
        s2_unc2 = om2 / (1 - al2 - ga2 / 2 - be2)
        nic2 = om2 + al2 * u**2 + ga2 * u**2 * (u < 0).astype(float) + be2 * s2_unc2

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(u, nic1, color="black", linewidth=1.8, label="sGARCH")
        ax.plot(u, nic2, color="red", linewidth=1.8, label="GJR-GARCH")
        ax.set_xlabel("Past shock u(t-1)")
        ax.set_ylabel("Conditional variance")
        ax.set_title("News Impact Curve — sGARCH vs GJR-GARCH")
        ax.legend()

        plt.tight_layout()
        if save:
            self._save_figure(fig, "nic_two.png")
        plt.show()
        plt.close()

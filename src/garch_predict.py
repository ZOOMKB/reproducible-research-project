"""GARCH volatility forecasting for the ATVI reproducible research project.

Usage:
    from src.garch_predict import GARCHForecaster

    gm = GARCHModels.from_processed_csv()
    gm.fit_sgarch()
    fc = GARCHForecaster(gm)
    table = fc.forecast_fixed(gm.fit1, horizon=1, n_expost=100)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from arch.univariate import EGARCH

from src.config import OUTPUTS_DIR

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

DEFAULT_N_SIMS = 10_000


@dataclass
class GARCHForecaster:
    gm: object
    figures_dir: Path = field(default_factory=lambda: FIGURES_DIR)

    @classmethod
    def from_garch_models(cls, gm: object) -> GARCHForecaster:
        return cls(gm=gm)

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=150)
        return path

    @staticmethod
    def _require_fit(fit, name: str):
        if fit is None:
            raise RuntimeError(
                f"{name} is not fitted yet. Call the corresponding fit_* method first."
            )
        return fit

    @staticmethod
    def _is_simulation_model(fit) -> bool:
        """Return True if the model requires simulation-based forecasting.

        TGARCHResult has no .model attribute and handles forecasting via
        ugarchforecast internally — always returns False for it.
        EGARCH models require simulation in the arch library.

        Args:
            fit: Fitted model result (ARCHModelResult or TGARCHResult).

        Returns:
            True if simulation forecasting should be used, False otherwise.
        """
        if not hasattr(fit, "model"):
            return False
        return isinstance(fit.model.volatility, EGARCH)

    def _nobs(self) -> int:
        return int(np.asarray(self.gm.yret).shape[0])

    def expost_start(self, nobs: int, n_expost: int, horizon: int) -> int:
        start = nobs - n_expost - horizon
        if start < 0:
            raise ValueError(
                f"Cannot place {n_expost} ex-post forecasts at horizon "
                f"{horizon} in a sample of {nobs} observations."
            )
        return start

    def _bands(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        lower = mean - z * sigma
        upper = mean + z * sigma
        return lower, upper

    def forecast_fixed(
        self,
        fit,
        horizon: int = 1,
        n_expost: int = 100,
        alpha: float = 0.05,
        seed: int | None = None,
        n_sims: int = DEFAULT_N_SIMS,
    ) -> pd.DataFrame:

        fit = self._require_fit(fit, "model")
        if horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        if n_expost < 1:
            raise ValueError("n_expost must be a positive integer.")

        nobs = self._nobs()
        start = self.expost_start(nobs, n_expost, horizon)

        forecast = self._run_forecast(fit, horizon, start, seed, n_sims)

        col = f"h.{horizon}"
        mean_all = forecast.mean[col].to_numpy()
        variance_all = forecast.variance[col].to_numpy()

        mean = mean_all[:n_expost]
        sigma = np.sqrt(variance_all[:n_expost])

        origins = np.arange(start, start + n_expost)
        target = origins + horizon

        yret = np.asarray(self.gm.yret)
        actual = yret[target]

        lower, upper = self._bands(mean, sigma, alpha)

        return pd.DataFrame(
            {
                "t": target,
                "actual": actual,
                "mean": mean,
                "sigma": sigma,
                "lower": lower,
                "upper": upper,
            }
        )

    def forecast_rolling(
        self,
        fit,
        horizon: int = 10,
        alpha: float = 0.05,
        seed: int | None = None,
        n_sims: int = DEFAULT_N_SIMS,
    ) -> pd.DataFrame:

        fit = self._require_fit(fit, "model")
        if horizon < 1:
            raise ValueError("horizon must be a positive integer.")

        nobs = self._nobs()
        start = nobs - 1

        forecast = self._run_forecast(fit, horizon, start, seed, n_sims)

        mean = forecast.mean.to_numpy()[-1, :]
        variance = forecast.variance.to_numpy()[-1, :]
        sigma = np.sqrt(variance)

        steps = np.arange(1, horizon + 1)
        lower, upper = self._bands(mean, sigma, alpha)

        return pd.DataFrame(
            {
                "t": steps,
                "mean": mean,
                "sigma": sigma,
                "lower": lower,
                "upper": upper,
            }
        )

    def plot_forecast_fixed(
        self,
        table: pd.DataFrame,
        title: str = "Ex-post forecast",
        filename: str | None = None,
        save: bool = True,
    ) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(
            table["t"],
            table["lower"],
            table["upper"],
            alpha=0.2,
            label="Forecast band",
        )
        ax.plot(
            table["t"],
            table["mean"],
            linewidth=1.2,
            label="Forecast mean",
        )
        ax.scatter(
            table["t"],
            table["actual"],
            s=12,
            color="black",
            label="Observed return",
        )

        ax.set_title(title)
        ax.set_xlabel("Observation index")
        ax.set_ylabel("Percentage return")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()

        if save:
            if filename is None:
                slug = title.lower().replace(" ", "_")
                filename = f"forecast_fixed_{slug}.png"
            self._save_figure(fig, filename)

        plt.show()
        plt.close()
        return fig

    def plot_forecast_rolling(
        self,
        table: pd.DataFrame,
        title: str = "Ex-ante forecast path",
        filename: str | None = None,
        save: bool = True,
    ) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(
            table["t"],
            table["lower"],
            table["upper"],
            alpha=0.2,
            label="Forecast band",
        )
        ax.plot(
            table["t"],
            table["mean"],
            linewidth=1.4,
            marker="o",
            markersize=4,
            label="Forecast mean",
        )

        ax.set_title(title)
        ax.set_xlabel("Steps ahead")
        ax.set_ylabel("Percentage return")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()

        if save:
            if filename is None:
                slug = title.lower().replace(" ", "_")
                filename = f"forecast_rolling_{slug}.png"
            self._save_figure(fig, filename)

        plt.show()
        plt.close()
        return fig

    def _run_forecast(
        self,
        fit,
        horizon: int,
        start: int,
        seed: int | None,
        n_sims: int,
    ):
        """Route forecast call to the correct method for this model type.

        For TGARCHResult: calls fit.forecast() which routes to rugarch's
        ugarchforecast via rpy2 — no simulation needed.
        For EGARCH (arch): uses simulation-based forecasting.
        For all other arch models: uses analytic forecasting.

        Args:
            fit: Fitted model result (ARCHModelResult or TGARCHResult).
            horizon: Number of steps ahead.
            start: Start index for rolling forecasts.
            seed: Random seed for simulation (ignored for TGARCHResult).
            n_sims: Number of simulations (ignored for TGARCHResult).

        Returns:
            Forecast result with .mean and .variance DataFrames.
        """
        if self._is_simulation_model(fit):
            generator = np.random.default_rng(seed)
            return fit.forecast(
                horizon=horizon,
                start=start,
                method="simulation",
                simulations=n_sims,
                rng=generator.standard_normal,
                reindex=False,
            )
        return fit.forecast(
            horizon=horizon,
            start=start,
            method="analytic",
            reindex=False,
        )


def run_forecasting(
    gm: object,
    horizon: int = 1,
    n_expost: int = 100,
    seed: int | None = 20240101,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:

    fc = GARCHForecaster.from_garch_models(gm)
    models = {
        "sGARCH(1,1)": gm.fit1,
        "GJR-GARCH(1,1)": gm.fit2,
        "T-GARCH(1,1)": gm.fit5,
        "IGARCH(1,1)": gm.fit6,
    }

    results: dict[str, pd.DataFrame] = {}
    for label, fit in models.items():
        table = fc.forecast_fixed(
            fit,
            horizon=horizon,
            n_expost=n_expost,
            seed=seed,
        )
        results[label] = table
        if verbose:
            print(f"\nEx-post forecasts -- {label} (horizon {horizon})")
            print(table.head().to_string(index=False))

    return results


def main() -> None:
    """Run the forecasting stage from the command line."""
    from src.garch_models import GARCHModels

    print("=" * 60)
    print("  ATVI Analysis -- GARCH Forecasting")
    print("=" * 60)

    gm = GARCHModels.from_processed_csv()
    gm.fit_sgarch()
    gm.fit_gjrgarch()
    gm.fit_tgarch()
    gm.fit_igarch()
    run_forecasting(gm)


if __name__ == "__main__":
    main()

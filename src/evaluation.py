"""Forecast evaluation for the ATVI reproducible research project.

Usage:
    from src.evaluation import evaluate_volatility_forecast

    table = fc.forecast_fixed(gm.fit5, horizon=1, n_expost=100, seed=1)
    measures = evaluate_volatility_forecast(
        realized=gk_vol_window, sigma=table["sigma"]
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


def _as_array(values) -> np.ndarray:

    return np.asarray(values, dtype=float).ravel()


def _random_walk_naive(y_true: np.ndarray) -> np.ndarray:

    naive = np.empty_like(y_true)
    naive[0] = np.nan
    naive[1:] = y_true[:-1]
    return naive


def error_measures(
    y_true,
    y_pred,
    naive=None,
) -> pd.Series:

    actual = _as_array(y_true)
    pred = _as_array(y_pred)
    if actual.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {actual.shape[0]} and {pred.shape[0]}."
        )

    error = actual - pred

    me = np.nanmean(error)
    mae = np.nanmean(np.abs(error))
    rmse = np.sqrt(np.nanmean(error**2))

    if np.all(actual > 0):
        pct_error = error / actual
        mpe = np.nanmean(pct_error)
        mape = np.nanmean(np.abs(pct_error))
        rmspe = np.sqrt(np.nanmean(pct_error**2))
    else:
        mpe = np.nan
        mape = np.nan
        rmspe = np.nan

    naive_values = _random_walk_naive(actual) if naive is None else _as_array(naive)
    naive_error = actual - naive_values
    naive_mae = np.nanmean(np.abs(naive_error))
    naive_rmse = np.sqrt(np.nanmean(naive_error**2))

    sc_mae = mae / naive_mae if naive_mae > 0 else np.nan
    sc_rmse = rmse / naive_rmse if naive_rmse > 0 else np.nan

    return pd.Series(
        {
            "ME": me,
            "MAE": mae,
            "RMSE": rmse,
            "MPE": mpe,
            "MAPE": mape,
            "RMSPE": rmspe,
            "ScMAE": sc_mae,
            "ScRMSE": sc_rmse,
        },
        name="value",
    )


def evaluate_return_forecast(actual, mean, naive=None) -> pd.Series:

    return error_measures(actual, mean, naive=naive)


def evaluate_volatility_forecast(realized, sigma, naive=None) -> pd.Series:

    return error_measures(realized, sigma, naive=naive)


def diebold_mariano(
    e1,
    e2,
    horizon: int = 1,
    power: int = 2,
    hln: bool = True,
) -> pd.Series:

    err1 = _as_array(e1)
    err2 = _as_array(e2)
    if err1.shape != err2.shape:
        raise ValueError("e1 and e2 must have the same length.")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    loss = np.abs(err1) ** power - np.abs(err2) ** power
    n = loss.shape[0]
    d_bar = np.mean(loss)

    gamma0 = np.mean((loss - d_bar) ** 2)
    long_run = gamma0
    for lag in range(1, horizon):
        cov = np.mean((loss[lag:] - d_bar) * (loss[:-lag] - d_bar))
        long_run += 2.0 * cov

    var_d_bar = long_run / n

    if var_d_bar <= 0:
        return pd.Series(
            {
                "statistic": 0.0,
                "p_value": 1.0,
                "horizon": horizon,
                "power": power,
            },
            name="diebold_mariano",
        )

    statistic = d_bar / np.sqrt(var_d_bar)

    if hln:
        correction = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        statistic = statistic * np.sqrt(correction)
        p_value = 2.0 * stats.t.cdf(-np.abs(statistic), df=n - 1)
    else:
        p_value = 2.0 * stats.norm.cdf(-np.abs(statistic))

    return pd.Series(
        {
            "statistic": statistic,
            "p_value": p_value,
            "horizon": horizon,
            "power": power,
        },
        name="diebold_mariano",
    )


def mincer_zarnowitz(y_true, y_pred, cov_type: str = "HAC") -> dict:

    actual = _as_array(y_true)
    pred = _as_array(y_pred)
    if actual.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    design = sm.add_constant(pred)
    model = sm.OLS(actual, design)

    if cov_type == "HAC":
        max_lags = int(np.floor(4 * (actual.shape[0] / 100.0) ** (2.0 / 9.0)))
        result = model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    elif cov_type == "HC0":
        result = model.fit(cov_type="HC0")
    else:
        result = model.fit()

    coef_table = pd.DataFrame(
        {
            "estimate": result.params,
            "std_error": result.bse,
            "t_value": result.tvalues,
            "p_value": result.pvalues,
        },
        index=["intercept", "slope"],
    )

    joint = result.f_test("const = 0, x1 = 1")

    return {
        "coef_table": coef_table,
        "f_stat": float(np.squeeze(joint.fvalue)),
        "f_pvalue": float(joint.pvalue),
        "cov_type": cov_type,
    }


def prediction_bands(mean, sigma, alpha: float = 0.05) -> pd.DataFrame:

    centre = _as_array(mean)
    spread = _as_array(sigma)
    if centre.shape != spread.shape:
        raise ValueError("mean and sigma must have the same length.")

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return pd.DataFrame(
        {
            "mean": centre,
            "lower": centre - z * spread,
            "upper": centre + z * spread,
        }
    )


def _check(condition: bool, label: str) -> None:

    if not condition:
        raise AssertionError(f"FAILED: {label}")
    print(f"  ok: {label}")


def _self_test() -> None:

    print("error_measures")
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 1.5, 3.5, 3.5, 5.5])
    err = y_true - y_pred
    em = error_measures(y_true, y_pred)
    _check(np.isclose(em["ME"], err.mean()), "ME matches manual formula")
    _check(np.isclose(em["MAE"], np.abs(err).mean()), "MAE matches manual formula")
    _check(
        np.isclose(em["RMSE"], np.sqrt((err**2).mean())),
        "RMSE matches manual formula",
    )

    signed = error_measures(np.array([-1.0, 2.0, -3.0]), np.array([-0.9, 1.8, -3.3]))
    _check(np.isnan(signed["MAPE"]), "MAPE is NaN for a signed series")

    positive = error_measures(
        np.array([10.0, 20.0, 30.0]), np.array([11.0, 19.0, 33.0])
    )
    _check(not np.isnan(positive["MAPE"]), "MAPE is defined for a positive series")

    perfect = error_measures(y_true, y_true)
    _check(np.isclose(perfect["RMSE"], 0.0), "RMSE is zero for a perfect forecast")

    try:
        error_measures(np.array([1.0, 2.0]), np.array([1.0]))
        _check(False, "length mismatch raises ValueError")
    except ValueError:
        _check(True, "length mismatch raises ValueError")

    print("evaluate_return_forecast / evaluate_volatility_forecast")
    actual = np.array([0.1, -0.2, 0.3, -0.1])
    mean = np.array([0.0, -0.1, 0.2, 0.0])
    _check(
        np.isclose(
            evaluate_return_forecast(actual, mean)["RMSE"],
            error_measures(actual, mean)["RMSE"],
        ),
        "return wrapper matches error_measures",
    )
    realized = np.array([1.0, 1.5, 2.0, 1.2])
    sigma = np.array([1.1, 1.4, 1.9, 1.3])
    _check(
        np.isclose(
            evaluate_volatility_forecast(realized, sigma)["MAE"],
            error_measures(realized, sigma)["MAE"],
        ),
        "volatility wrapper matches error_measures",
    )

    print("diebold_mariano")
    rng = np.random.default_rng(0)
    e1 = rng.normal(scale=1.0, size=120)
    e2 = rng.normal(scale=2.0, size=120)
    dm = diebold_mariano(e1, e2, horizon=1, power=2)
    _check(dm["statistic"] < 0, "statistic is negative when first model is better")
    _check(0.0 <= dm["p_value"] <= 1.0, "p-value lies in the unit interval")

    errors = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.3, 0.1])
    dm_eq = diebold_mariano(errors, errors)
    _check(
        np.isclose(dm_eq["statistic"], 0.0) and np.isclose(dm_eq["p_value"], 1.0),
        "identical errors give the degenerate result",
    )

    try:
        diebold_mariano(np.array([0.1, 0.2]), np.array([0.1]))
        _check(False, "length mismatch raises ValueError")
    except ValueError:
        _check(True, "length mismatch raises ValueError")

    print("mincer_zarnowitz")
    line = np.linspace(0.0, 10.0, 60)
    mz = mincer_zarnowitz(line, line, cov_type="HC0")
    coef = mz["coef_table"]
    _check(
        np.isclose(coef.loc["slope", "estimate"], 1.0),
        "perfect forecast gives unit slope",
    )
    _check(
        np.isclose(coef.loc["intercept", "estimate"], 0.0, atol=1e-8),
        "perfect forecast gives zero intercept",
    )
    noisy = mincer_zarnowitz(line, line + rng.normal(scale=0.1, size=60))
    _check(0.0 <= noisy["f_pvalue"] <= 1.0, "joint test p-value lies in unit interval")

    print("prediction_bands")
    bands = prediction_bands(
        np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]), alpha=0.05
    )
    half_width = (bands["upper"] - bands["mean"]).to_numpy()
    _check(
        np.allclose(half_width, 1.959964 * np.array([1.0, 2.0, 3.0]), atol=1e-4),
        "95% band half-width matches the normal quantile",
    )
    try:
        prediction_bands(np.array([0.0, 0.0]), np.array([1.0]))
        _check(False, "length mismatch raises ValueError")
    except ValueError:
        _check(True, "length mismatch raises ValueError")

    print("all evaluation self-tests passed")


if __name__ == "__main__":
    _self_test()

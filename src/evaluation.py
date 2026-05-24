"""Forecast evaluation for the ATVI reproducible research project.

Quantifies how well the GARCH-family forecasts produced by
``src.forecasting`` track what was actually observed. The module is pure:
every function takes plain arrays of numbers and returns numbers, with no
dependency on the forecasting or model-fitting modules.

Three groups of tools are provided:

    error measures      ME, MAE, RMSE and their percentage and
                        random-walk-scaled counterparts.
    Diebold-Mariano     a test of equal predictive accuracy between two
                        competing forecasts, with the Harvey-Leybourne-
                        Newbold small-sample correction.
    Mincer-Zarnowitz    a regression-based test of forecast optimality.

Two named wrappers, ``evaluate_return_forecast`` and
``evaluate_volatility_forecast``, apply the error measures to the two
quantities of interest. A GARCH model forecasts a conditional standard
deviation that is never directly observed, so the volatility wrapper
compares the forecast against a realized-volatility proxy supplied by the
caller (for example squared returns or Garman-Klass volatility).

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
    """Convert an input series or array to a 1-D float array.

    Args:
        values: A sequence, pandas Series, or numpy array.

    Returns:
        A 1-D numpy array of floats.
    """
    return np.asarray(values, dtype=float).ravel()


def _random_walk_naive(y_true: np.ndarray) -> np.ndarray:
    """Build the random-walk naive forecast for a series.

    The random-walk forecast for observation t is the value observed at
    t - 1. The first observation has no predecessor and is returned as NaN.

    Args:
        y_true: The observed series.

    Returns:
        Array of the same length holding the lagged series.
    """
    naive = np.empty_like(y_true)
    naive[0] = np.nan
    naive[1:] = y_true[:-1]
    return naive


def error_measures(
    y_true,
    y_pred,
    naive=None,
) -> pd.Series:
    """Compute forecast error measures for a forecast against actuals.

    Reports level measures (ME, MAE, RMSE), percentage measures (MPE,
    MAPE, RMSPE) and random-walk-scaled measures (ScMAE, ScRMSE).

    Percentage measures are only defined when every observed value is
    strictly positive; for a return series, which takes both signs, they
    are returned as NaN. Scaled measures divide the forecast error by the
    error of a random-walk naive forecast. When ``naive`` is not supplied
    the random-walk forecast is constructed internally from ``y_true``.

    Args:
        y_true: Observed values.
        y_pred: Forecast values, aligned with y_true.
        naive: Optional naive-forecast values aligned with y_true. When
            None, a random-walk naive forecast is built from y_true.

    Returns:
        Series indexed by measure name: ME, MAE, RMSE, MPE, MAPE, RMSPE,
        ScMAE, ScRMSE.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
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
    """Evaluate a conditional-mean return forecast.

    Thin wrapper around error_measures for the return forecast: the
    observed return is compared against the forecast conditional mean.

    Args:
        actual: Observed returns.
        mean: Forecast conditional mean returns.
        naive: Optional naive-forecast values. When None a random-walk
            naive forecast is used.

    Returns:
        Series of error measures, as returned by error_measures.
    """
    return error_measures(actual, mean, naive=naive)


def evaluate_volatility_forecast(realized, sigma, naive=None) -> pd.Series:
    """Evaluate a conditional-volatility forecast.

    A GARCH model forecasts a conditional standard deviation that is never
    directly observed. The forecast is therefore compared against a
    realized-volatility proxy supplied by the caller, such as the absolute
    return, the squared return, or Garman-Klass volatility. Both inputs
    must be expressed on the same scale.

    Args:
        realized: Realized-volatility proxy aligned with the forecasts.
        sigma: Forecast conditional standard deviation.
        naive: Optional naive-forecast values. When None a random-walk
            naive forecast is used.

    Returns:
        Series of error measures, as returned by error_measures.
    """
    return error_measures(realized, sigma, naive=naive)


def diebold_mariano(
    e1,
    e2,
    horizon: int = 1,
    power: int = 2,
    hln: bool = True,
) -> pd.Series:
    """Test two competing forecasts for equal predictive accuracy.

    The Diebold-Mariano test compares the loss differential between two
    forecast error series. The loss is the absolute error raised to
    ``power`` (power 1 for absolute-error loss, power 2 for squared-error
    loss). A long-run variance with a horizon-dependent lag truncation is
    used so the test is valid for multi-step forecasts.

    When ``hln`` is True the Harvey-Leybourne-Newbold small-sample
    correction is applied: the statistic is rescaled and compared against
    a t-distribution rather than a standard normal.

    Args:
        e1: Forecast errors of the first model.
        e2: Forecast errors of the second model.
        horizon: Forecast horizon; sets the lag truncation to horizon - 1.
        power: Power of the absolute error in the loss function.
        hln: Whether to apply the Harvey-Leybourne-Newbold correction.

    Returns:
        Series with the test statistic, p-value, horizon and power. A
        negative statistic favours the first model; a positive statistic
        favours the second. When the two error series are identical the
        loss differential has zero variance and the function returns a
        statistic of zero with a p-value of one.

    Raises:
        ValueError: If e1 and e2 have different lengths or horizon < 1.
    """
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
    """Run the Mincer-Zarnowitz forecast-optimality regression.

    The observed series is regressed on a constant and the forecast:

        y_true = a + b * y_pred + error

    A forecast is optimal when a = 0 and b = 1 jointly. The function fits
    the regression and reports the joint F-test of that hypothesis. The
    covariance estimator can be made robust to heteroskedasticity (HC0) or
    to heteroskedasticity and autocorrelation (HAC).

    Args:
        y_true: Observed values.
        y_pred: Forecast values aligned with y_true.
        cov_type: Covariance estimator: 'nonrobust', 'HC0', or 'HAC'.

    Returns:
        Dict with keys:
            'coef_table'  DataFrame of intercept and slope estimates
            'f_stat'      joint-test F statistic for (a=0, b=1)
            'f_pvalue'    p-value of the joint test
            'cov_type'    the covariance estimator used

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
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
    """Compute symmetric Gaussian prediction bands for a forecast.

    Args:
        mean: Forecast conditional means.
        sigma: Forecast conditional standard deviations.
        alpha: Total tail probability, e.g. 0.05 for a 95% band.

    Returns:
        DataFrame with columns mean, lower and upper.

    Raises:
        ValueError: If mean and sigma have different lengths.
    """
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

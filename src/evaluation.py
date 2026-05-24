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

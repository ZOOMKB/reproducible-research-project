"""Tests for forecast evaluation helpers on toy arrays."""

import numpy as np
from src.evaluation import (
    diebold_mariano,
    error_measures,
    evaluate_return_forecast,
    evaluate_volatility_forecast,
    prediction_bands,
)


def test_error_measures_match_manual_values() -> None:
    """Basic error measures should match manual calculations."""
    actual = np.array([1.0, 2.0, 4.0])
    forecast = np.array([1.5, 1.5, 5.0])
    naive = np.array([0.5, 1.5, 3.5])

    measures = error_measures(actual, forecast, naive=naive)

    assert np.isclose(measures["ME"], -1 / 3)
    assert np.isclose(measures["MAE"], 2 / 3)
    assert np.isclose(measures["RMSE"], np.sqrt(0.5))
    assert np.isclose(measures["MAPE"], 1 / 3)
    assert np.isclose(measures["ScMAE"], (2 / 3) / 0.5)


def test_return_forecast_metrics_handle_signed_series() -> None:
    """Percentage errors should be undefined for signed return series."""
    actual = np.array([0.1, -0.2, 0.3])
    forecast = np.array([0.0, -0.1, 0.2])

    measures = evaluate_return_forecast(actual, forecast)

    assert np.isclose(measures["MAE"], 0.1)
    assert np.isnan(measures["MAPE"])


def test_volatility_forecast_metrics_are_defined() -> None:
    """Volatility metrics should include percentage errors for positive data."""
    realized = np.array([1.0, 1.5, 2.0])
    sigma = np.array([1.1, 1.4, 1.8])

    measures = evaluate_volatility_forecast(realized, sigma)

    assert np.isclose(measures["MAE"], (0.1 + 0.1 + 0.2) / 3)
    assert not np.isnan(measures["MAPE"])


def test_prediction_bands_use_normal_quantile() -> None:
    """Prediction bands should expand symmetrically around the mean."""
    bands = prediction_bands(
        mean=np.array([0.0, 1.0]),
        sigma=np.array([1.0, 2.0]),
        alpha=0.05,
    )

    assert np.allclose(bands["mean"], [0.0, 1.0])
    assert np.allclose(bands["upper"] - bands["mean"], [1.959964, 3.919928])
    assert np.allclose(bands["mean"] - bands["lower"], [1.959964, 3.919928])


def test_diebold_mariano_identical_errors_are_neutral() -> None:
    """Identical forecast errors should produce the neutral DM result."""
    errors = np.array([0.1, -0.2, 0.3, -0.1])

    result = diebold_mariano(errors, errors)

    assert np.isclose(result["statistic"], 0.0)
    assert np.isclose(result["p_value"], 1.0)

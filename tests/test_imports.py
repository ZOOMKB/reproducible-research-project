"""Lightweight import checks for the main project modules."""

from src.arma_model import ARMAModel
from src.data import ATVIDataProcessor
from src.diagnostics import PriceEDA
from src.evaluation import evaluate_return_forecast
from src.garch_diagnostics import GARCHDiagnostics
from src.garch_models import GARCHModels
from src.garch_predict import GARCHForecaster
from src.pipeline import DataEDAResult
from src.returns_analysis import ReturnsAnalyzer


def test_main_project_classes_import() -> None:
    """Main classes should remain importable from their final modules."""
    assert ATVIDataProcessor
    assert PriceEDA
    assert ReturnsAnalyzer
    assert ARMAModel
    assert GARCHModels
    assert GARCHDiagnostics
    assert GARCHForecaster
    assert DataEDAResult


def test_evaluation_function_imports() -> None:
    """Evaluation helpers should remain available for the report."""
    assert evaluate_return_forecast

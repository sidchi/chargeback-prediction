"""Per-transaction explanations: exact SHAP values and ranked, human-readable risk factors."""

from chargeback_prediction.explain.factors import (
    DEFAULT_TOP_K,
    MIN_CONTRIBUTION,
    Direction,
    Factor,
    Importance,
    TransactionExplanation,
    explain_rows,
    format_value,
    global_importance,
    sigmoid,
    top_factors,
)
from chargeback_prediction.explain.run import (
    ExplainConfig,
    ExplanationReport,
    explain_test_window,
    explain_transactions,
    load_model,
)
from chargeback_prediction.explain.shapley import ShapValues, shap_values

__all__ = [
    "DEFAULT_TOP_K",
    "MIN_CONTRIBUTION",
    "Direction",
    "ExplainConfig",
    "ExplanationReport",
    "Factor",
    "Importance",
    "ShapValues",
    "TransactionExplanation",
    "explain_rows",
    "explain_test_window",
    "explain_transactions",
    "format_value",
    "global_importance",
    "load_model",
    "shap_values",
    "sigmoid",
    "top_factors",
]

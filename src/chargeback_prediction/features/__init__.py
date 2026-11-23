"""Point-in-time feature pipeline for chargeback-risk models.

See :mod:`chargeback_prediction.features.spec` for the catalog and ``docs/features.md`` for
the rendered feature table.
"""

from chargeback_prediction.features.pipeline import (
    LABEL_COLUMNS,
    FeatureRow,
    FeatureTable,
    build_features,
)
from chargeback_prediction.features.spec import (
    FEATURE_NAMES,
    FEATURES,
    HISTORY_CAP_DAYS,
    FeatureGroup,
    FeatureKind,
    FeatureSpec,
    render_feature_table,
)

__all__ = [
    "FEATURES",
    "FEATURE_NAMES",
    "HISTORY_CAP_DAYS",
    "LABEL_COLUMNS",
    "FeatureGroup",
    "FeatureKind",
    "FeatureRow",
    "FeatureSpec",
    "FeatureTable",
    "build_features",
    "render_feature_table",
]

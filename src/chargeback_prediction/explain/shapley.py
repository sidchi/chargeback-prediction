"""Exact interventional SHAP values for the two baseline models, in log-odds space.

Why not the ``shap`` package? Two reasons. Its :class:`~shap.TreeExplainer` does not pass its
own additivity check on scikit-learn's ``HistGradientBoostingClassifier`` (sum of attributions
off by ~0.25 log-odds on this model, ``shap`` 0.54), and it pulls numba, llvmlite and pandas
into a service that otherwise needs NumPy and scikit-learn. Both baselines have structure that
makes *exact* Shapley values cheap, so we compute them directly and test them against a
brute-force enumeration of every feature coalition.

**Value function.** Everything here is *interventional* SHAP (a.k.a. "independent" or baseline
Shapley): for a coalition ``S`` the value is the model's raw log-odds averaged over a background
sample ``z`` with ``x``'s values patched in for the features in ``S``::

    v(S) = mean_j f(x_S, z_j[~S])

The attributions satisfy local accuracy exactly::

    base_value + phi.sum(axis=1) == f(X)          base_value = mean_j f(z_j)

Log-odds rather than probability because attributions are additive there; ``exp(phi_i)`` reads
as "this feature multiplies the odds of a dispute by …", which is what the factor summaries use.

**Background.** Attributions are relative to the background: "why is this transaction riskier
than a typical *training-window* transaction". Pass a sample of training rows (a few hundred is
plenty: the estimate's standard error shrinks with ``1/sqrt(n)``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import factorial
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from chargeback_prediction.modeling.models import Classifier

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ShapValues:
    """Per-row, per-feature attributions in log-odds, plus what they sum to."""

    values: FloatArray
    """Shape ``(n_rows, n_features)``."""
    base_value: float
    """Mean raw log-odds over the background sample."""
    raw: FloatArray = field(repr=False)
    """Model log-odds for each explained row; ``base_value + values.sum(1)``."""
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        n, m = self.values.shape
        if m != len(self.feature_names):
            raise ValueError(f"{m} attribution columns but {len(self.feature_names)} names")
        if self.raw.shape != (n,):
            raise ValueError("raw must have one entry per row")

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def additivity_error(self) -> float:
        """Largest ``|base + sum(phi) - f(x)|`` over rows; ~1e-12 for an exact explainer."""
        if not len(self):
            return 0.0
        return float(np.max(np.abs(self.base_value + self.values.sum(axis=1) - self.raw)))

    def mean_abs(self) -> FloatArray:
        """Global importance: mean ``|phi|`` per feature."""
        if not len(self):
            return np.zeros(len(self.feature_names), dtype=np.float64)
        return np.asarray(np.abs(self.values).mean(axis=0), dtype=np.float64)


def shap_values(
    model: Classifier,
    X: FloatArray,
    background: FloatArray,
    feature_names: Sequence[str],
) -> ShapValues:
    """Exact interventional SHAP for a fitted baseline (see :func:`build_model`).

    Dispatches on the model type: a scikit-learn :class:`Pipeline` ending in a linear classifier
    (``coef_``/``intercept_``) with element-wise preprocessing, or a binary
    :class:`HistGradientBoostingClassifier`.

    Raises:
        TypeError: for an unsupported model.
        ValueError: on shape mismatches or an empty background.
    """
    X = np.asarray(X, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if background.ndim != 2 or len(background) == 0:
        raise ValueError("background must be a non-empty 2-D array")
    if X.ndim != 2 or X.shape[1] != background.shape[1] or X.shape[1] != len(feature_names):
        raise ValueError(
            f"X {X.shape} and background {background.shape} must have "
            f"{len(feature_names)} feature columns"
        )
    if isinstance(model, Pipeline):
        values, base = _linear_pipeline_shap(model, X, background)
    elif isinstance(model, HistGradientBoostingClassifier):
        values, base = _hist_gbm_shap(model, X, background)
    else:
        raise TypeError(f"no exact SHAP implementation for {type(model).__name__}")
    raw = np.asarray(model.decision_function(X), dtype=np.float64).reshape(len(X))
    return ShapValues(values=values, base_value=base, raw=raw, feature_names=tuple(feature_names))


# ------------------------------------------------------------------ linear


def _linear_pipeline_shap(
    pipe: Pipeline, X: FloatArray, background: FloatArray
) -> tuple[FloatArray, float]:
    """``phi_i = w_i · (T(x)_i − mean_j T(z_j)_i)`` where ``T`` is the preprocessing.

    Exact *provided every preprocessing step is element-wise per feature* (true for the
    baseline: signed-log1p then standardisation), because then the log-odds are a sum of
    one-feature terms and Shapley values of an additive function are just its centred terms.
    A step that mixes features (PCA, polynomial features) would break this, so we check it.
    """
    clf = pipe[-1]
    coef = np.asarray(getattr(clf, "coef_", None), dtype=np.float64)
    if coef.ndim != 2 or coef.shape[0] != 1:
        raise TypeError("pipeline must end in a fitted binary linear classifier")
    transform = pipe[:-1]
    tx = np.asarray(transform.transform(X), dtype=np.float64)
    tz = np.asarray(transform.transform(background), dtype=np.float64)
    if tx.shape != X.shape:
        raise TypeError("preprocessing must be element-wise (same number of columns in and out)")
    _check_elementwise(transform, background)
    w = coef[0]
    values = (tx - tz.mean(axis=0)) * w
    base = float(np.asarray(clf.intercept_, dtype=np.float64)[0] + tz.mean(axis=0) @ w)
    return values, base


def _check_elementwise(transform: Any, background: FloatArray) -> None:
    """Perturb one column and confirm only that column of the output moves."""
    probe = background[:2].copy()
    moved = probe.copy()
    moved[:, 0] = moved[:, 0] + 1.0
    delta = np.asarray(transform.transform(moved)) - np.asarray(transform.transform(probe))
    if np.any(np.abs(delta[:, 1:]) > 1e-12):
        raise TypeError("preprocessing mixes features; linear SHAP would not be exact")


# ------------------------------------------------------------------ trees


@dataclass(frozen=True, slots=True)
class _Leaf:
    value: float
    features: tuple[int, ...]
    """Distinct features split on along the root-to-leaf path."""
    conditions: tuple[tuple[tuple[float, bool, bool], ...], ...]
    """Per feature: ``(threshold, goes_left, missing_goes_left)`` for each split on it."""


def _leaves(nodes: npt.NDArray[Any]) -> list[_Leaf]:
    """Enumerate leaves of one HistGradientBoosting predictor tree with their path conditions."""
    out: list[_Leaf] = []
    stack: list[tuple[int, list[tuple[int, float, bool, bool]]]] = [(0, [])]
    while stack:
        idx, path = stack.pop()
        node = nodes[idx]
        if node["is_leaf"]:
            by_feature: dict[int, list[tuple[float, bool, bool]]] = {}
            for f, thr, left, miss_left in path:
                by_feature.setdefault(f, []).append((thr, left, miss_left))
            feats = tuple(sorted(by_feature))
            out.append(
                _Leaf(
                    value=float(node["value"]),
                    features=feats,
                    conditions=tuple(tuple(by_feature[f]) for f in feats),
                )
            )
            continue
        if node["is_categorical"]:
            raise TypeError("categorical splits are not supported")
        f, thr, miss_left = (
            int(node["feature_idx"]),
            float(node["num_threshold"]),
            bool(node["missing_go_to_left"]),
        )
        stack.append((int(node["left"]), [*path, (f, thr, True, miss_left)]))
        stack.append((int(node["right"]), [*path, (f, thr, False, miss_left)]))
    return out


def _satisfies(column: FloatArray, conds: Sequence[tuple[float, bool, bool]]) -> BoolArray:
    """Rows whose value on this feature follows every split on the path the same way."""
    ok = np.ones(len(column), dtype=np.bool_)
    missing = np.isnan(column)
    for thr, left, miss_left in conds:
        goes_left = np.where(missing, miss_left, column <= thr)
        ok &= goes_left if left else ~goes_left
    return ok


def _coalition_weight(a: int, b: int) -> float:
    """``a! b! / (a + b + 1)!``: Shapley weight shared by one feature with ``a``/``b`` partners."""
    return factorial(a) * factorial(b) / factorial(a + b + 1)


def _pattern_codes(ok: BoolArray) -> npt.NDArray[np.int64]:
    """Encode each row's per-feature booleans ``(n, d)`` as an integer in ``[0, 2**d)``."""
    weights = 1 << np.arange(ok.shape[1], dtype=np.int64)
    return np.asarray(ok.astype(np.int64) @ weights, dtype=np.int64)


def _leaf_table(d: int, z_counts: FloatArray) -> FloatArray:
    """Contribution per *x*-pattern and path feature, averaged over background patterns.

    Interventional Tree SHAP for one ``(x, z)`` pair (Lundberg et al. 2020): the leaf is reached
    by the hybrid input iff, for every path feature, ``x`` or ``z`` satisfies all its splits.
    Features only ``x`` satisfies form ``S_x`` (size ``a``), only ``z`` form ``S_z`` (size
    ``b``). The leaf value ``v`` is then credited ``+v·W(a−1, b)`` to each feature in ``S_x``
    and ``−v·W(a, b−1)`` to each in ``S_z``; features both satisfy get nothing.

    Rows sharing a satisfaction pattern get identical credit, so we work on the ``2**d``
    patterns rather than on rows: a depth-3 tree has at most 8×8 pattern pairs per leaf.
    Returned table has shape ``(2**d, d)`` and is per unit leaf value.
    """
    n_patterns = 1 << d
    table = np.zeros((n_patterns, d), dtype=np.float64)
    total = float(z_counts.sum())
    for px in range(n_patterns):
        xo = [(px >> k) & 1 == 1 for k in range(d)]
        for pz in np.flatnonzero(z_counts):
            zo = [(int(pz) >> k) & 1 == 1 for k in range(d)]
            if not all(xa or za for xa, za in zip(xo, zo, strict=True)):
                continue  # hybrid input never reaches this leaf
            in_x = [xa and not za for xa, za in zip(xo, zo, strict=True)]
            in_z = [za and not xa for xa, za in zip(xo, zo, strict=True)]
            a, b = sum(in_x), sum(in_z)
            share = z_counts[pz] / total
            for k in range(d):
                if in_x[k]:
                    table[px, k] += share * _coalition_weight(a - 1, b)
                elif in_z[k]:
                    table[px, k] -= share * _coalition_weight(a, b - 1)
    return table


MAX_PATH_FEATURES = 12
"""Pattern tables grow as ``4**d``; the baseline GBM has depth 3, so ``d <= 3``."""


def _hist_gbm_shap(
    model: HistGradientBoostingClassifier, X: FloatArray, background: FloatArray
) -> tuple[FloatArray, float]:
    predictors = getattr(model, "_predictors", None)
    if predictors is None or any(len(it) != 1 for it in predictors):
        raise TypeError("expected a fitted binary HistGradientBoostingClassifier")
    values = np.zeros_like(X)
    for (tree,) in predictors:
        for leaf in _leaves(tree.nodes):
            d = len(leaf.features)
            if d == 0 or leaf.value == 0.0:
                continue  # single-leaf tree: a constant, absorbed by the base value
            if d > MAX_PATH_FEATURES:
                raise ValueError(f"tree path splits on {d} features; too deep for exact SHAP")
            cols = list(leaf.features)
            x_ok = np.column_stack(
                [_satisfies(X[:, f], c) for f, c in zip(cols, leaf.conditions, strict=True)]
            )
            z_ok = np.column_stack(
                [
                    _satisfies(background[:, f], c)
                    for f, c in zip(cols, leaf.conditions, strict=True)
                ]
            )
            z_counts = np.bincount(_pattern_codes(z_ok), minlength=1 << d).astype(np.float64)
            table = _leaf_table(d, z_counts)
            values[:, cols] += leaf.value * table[_pattern_codes(x_ok)]
    base = float(np.mean(model.decision_function(background)))
    return values, base

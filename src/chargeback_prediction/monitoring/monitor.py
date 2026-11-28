"""Watch a trained model in production: score drift, feature drift and label-delayed calibration.

Chargeback labels arrive 30–120 days after the charge, so a model can be wrong for a whole
quarter before its errors show up in dispute counts. Monitoring therefore runs on two clocks:

* **Leading indicators, available the day traffic arrives.** The PSI of the *score*
  distribution against the training window (what every decision actually consumes), the PSI of
  each input feature (which explains *why* the score moved), and the mean score, which is the
  model's own forecast of the dispute rate.
* **The lagging truth, once labels mature.** For rows whose dispute window has closed, compare
  observed disputes with the sum of predicted probabilities. Under a calibrated model the count
  of disputes is a sum of independent Bernoullis, so ``z = (observed − Σp) / sqrt(Σ p(1 − p))``
  is approximately standard normal. A z-test is used instead of ECE because a 30-day window
  holds only a few dozen disputes, too few for a stable ten-bin reliability table, while a
  count test stays well-defined and has a known false-alarm rate.

Rows whose window is still open are never used for the calibration check: a ``False`` label
there means "not yet", and counting it would make every recent window look over-predicted.
Maturity depends only on ``created_at``, so the mature part of a window is an unbiased sample
of it with respect to the label.

Each window gets a status. Score drift and miscalibration can raise an ``alert``; feature
drift on its own only raises a ``warn``, because a feature can move without hurting the score
(it may carry little weight) and is a diagnosis rather than a symptom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from chargeback_prediction.modeling import Classifier, FeatureMatrix, predict_risk, time_split
from chargeback_prediction.modeling.train import TrainConfig
from chargeback_prediction.monitoring.drift import (
    DEFAULT_BINS,
    Bins,
    psi_from_shares,
    psi_noise_floor,
)

FloatArray = npt.NDArray[np.float64]


class Status(IntEnum):
    """Ordered so the worst status of several checks is their ``max``."""

    OK = 0
    WARN = 1
    ALERT = 2
    PENDING = -1
    """Not enough data to judge (no matured labels yet, or too few rows)."""

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    window_days: int = 30
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    z_warn: float = 2.0
    z_alert: float = 3.0
    min_window_rows: int = 200
    """Windows smaller than this are reported but not judged: PSI on a handful of rows is noise."""
    min_labeled_rows: int = 500
    """Matured rows needed before the calibration test runs on a window."""
    n_bins: int = DEFAULT_BINS
    top_features: int = 5
    """How many of the most-drifted features each window lists."""
    split: TrainConfig = field(default_factory=TrainConfig)
    """Must match the split the model was trained with: its training window is the reference."""

    def __post_init__(self) -> None:
        if self.window_days < 1:
            raise ValueError("window_days must be >= 1")
        if not 0.0 < self.psi_warn < self.psi_alert:
            raise ValueError("need 0 < psi_warn < psi_alert")
        if not 0.0 < self.z_warn < self.z_alert:
            raise ValueError("need 0 < z_warn < z_alert")
        if self.min_window_rows < 1 or self.min_labeled_rows < 1:
            raise ValueError("minimum row counts must be >= 1")
        if self.top_features < 0:
            raise ValueError("top_features must be >= 0")

    def psi_status(self, psi: float) -> Status:
        if psi >= self.psi_alert:
            return Status.ALERT
        if psi >= self.psi_warn:
            return Status.WARN
        return Status.OK

    def z_status(self, z: float) -> Status:
        if abs(z) >= self.z_alert:
            return Status.ALERT
        if abs(z) >= self.z_warn:
            return Status.WARN
        return Status.OK


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    feature: str
    psi: float
    reference_mean: float
    current_mean: float
    status: Status


@dataclass(frozen=True, slots=True)
class CalibrationCheck:
    """Observed vs. expected disputes on a window's matured rows."""

    rows: int
    observed: int
    expected: float
    z: float
    status: Status

    @property
    def observed_rate(self) -> float:
        return self.observed / self.rows

    @property
    def expected_rate(self) -> float:
        return self.expected / self.rows

    @property
    def ratio(self) -> float:
        """Observed over expected disputes; 1.0 is perfectly calibrated in aggregate."""
        return self.observed / self.expected if self.expected else float("nan")


@dataclass(frozen=True, slots=True)
class WindowReport:
    start: datetime
    end: datetime
    """Exclusive."""
    rows: int
    mean_score: float
    """The model's forecast of this window's dispute rate."""
    score_psi: float
    noise_floor: float
    """PSI expected from sampling noise alone at this window's size."""
    features: tuple[FeatureDrift, ...]
    """Every feature, most drifted first."""
    label_coverage: float
    """Share of the window's rows whose dispute window has closed."""
    calibration: CalibrationCheck | None
    """``None`` until enough labels have matured."""
    drift_status: Status
    """Leading indicators: score PSI (can alert) and feature PSI (can only warn)."""
    reasons: tuple[str, ...]
    """Why the status is what it is, one short phrase per triggered check."""

    @property
    def calibration_status(self) -> Status:
        """The lagging check; ``PENDING`` until enough of the window's labels have matured."""
        return self.calibration.status if self.calibration is not None else Status.PENDING

    @property
    def status(self) -> Status:
        """Worst of the drift and calibration checks."""
        return max(self.drift_status, self.calibration_status)

    def drifted_features(self, top: int) -> tuple[FeatureDrift, ...]:
        return tuple(f for f in self.features[:top] if f.status is not Status.OK)


@dataclass(frozen=True, slots=True)
class MonitorReport:
    config: MonitorConfig
    reference_start: datetime
    reference_end: datetime
    reference_rows: int
    reference_mean_score: float
    reference_dispute_rate: float
    monitored_from: datetime
    """The model's training date (the split cutoff): monitoring starts here."""
    windows: tuple[WindowReport, ...]

    @property
    def status(self) -> Status:
        """Worst status across windows (``PENDING`` only if nothing could be judged)."""
        return max((w.status for w in self.windows), default=Status.PENDING)

    @property
    def drift_status(self) -> Status:
        return max((w.drift_status for w in self.windows), default=Status.PENDING)

    @property
    def calibration_status(self) -> Status:
        return max((w.calibration_status for w in self.windows), default=Status.PENDING)

    @property
    def labeled_windows(self) -> tuple[WindowReport, ...]:
        return tuple(w for w in self.windows if w.calibration is not None)

    def top_drivers(self, n: int = 3) -> tuple[str, ...]:
        """Features that most often rank among each judged window's most-drifted."""
        tally: dict[str, float] = {}
        for w in self.windows:
            for d in w.drifted_features(self.config.top_features):
                tally[d.feature] = tally.get(d.feature, 0.0) + d.psi
        return tuple(sorted(tally, key=lambda f: tally[f], reverse=True)[:n])

    def headline(self) -> str:
        """One-paragraph verdict separating "the inputs moved" from "the model is wrong"."""
        judged = [w for w in self.windows if w.drift_status is not Status.PENDING]
        parts: list[str] = []
        if not judged:
            parts.append("No window was large enough to judge drift.")
        else:
            worst = max(judged, key=lambda w: w.score_psi)
            drivers = self.top_drivers()
            if self.drift_status >= Status.WARN:
                parts.append(
                    f"The score distribution moved (PSI up to {worst.score_psi:.2f} in the "
                    f"window from {worst.start:%Y-%m-%d})"
                    + (f", driven mostly by {', '.join(drivers)}." if drivers else ".")
                )
            else:
                parts.append(
                    f"Scores are stable (PSI at most {worst.score_psi:.2f}, alert at "
                    f"{self.config.psi_alert})."
                )
        labeled = self.labeled_windows
        if not labeled:
            parts.append("No window has matured labels yet, so calibration is unverified.")
        else:
            bad = [w for w in labeled if w.calibration_status is not Status.OK]
            if not bad:
                parts.append(
                    f"Calibration held in all {len(labeled)} windows with matured labels, so "
                    "the probabilities are still usable; re-check volume-based thresholds."
                )
            else:
                spans = ", ".join(f"{w.start:%Y-%m-%d}" for w in bad)
                parts.append(
                    f"Calibration failed in {len(bad)} of {len(labeled)} labeled windows "
                    f"(starting {spans}): the probabilities are off; retrain or recalibrate."
                )
        return " ".join(parts)


def _reference_window(
    matrix: FeatureMatrix, config: MonitorConfig
) -> tuple[FeatureMatrix, datetime]:
    """The training window of ``matrix`` and the training date, as ``cbp train`` split them."""
    split = time_split(
        matrix,
        test_fraction=config.split.test_fraction,
        label_gap_days=config.split.label_gap_days,
    )
    return split.train, split.cutoff


def calibration_check(
    y: npt.NDArray[np.bool_], scores: FloatArray, config: MonitorConfig
) -> CalibrationCheck:
    """One-sample z-test of the dispute count against the sum of predicted probabilities."""
    if y.shape != scores.shape or y.size == 0:
        raise ValueError("labels and scores must be non-empty and aligned")
    expected = float(scores.sum())
    variance = float(np.sum(scores * (1.0 - scores)))
    observed = int(y.sum())
    z = (observed - expected) / math.sqrt(variance) if variance > 0 else 0.0
    return CalibrationCheck(
        rows=int(y.size), observed=observed, expected=expected, z=z, status=config.z_status(z)
    )


def _window_bounds(start: datetime, last: datetime, days: int) -> list[tuple[datetime, datetime]]:
    bounds: list[tuple[datetime, datetime]] = []
    step = timedelta(days=days)
    lo = start
    while lo <= last:
        bounds.append((lo, lo + step))
        lo += step
    return bounds


def monitor(
    model: Classifier,
    reference: FeatureMatrix,
    current: FeatureMatrix | None = None,
    config: MonitorConfig | None = None,
) -> MonitorReport:
    """Monitor ``model`` on the traffic that arrived after it was trained.

    Args:
        model: A model fitted on the training window of ``reference`` (as ``cbp train`` does).
        reference: The feature table the model was trained from. Its training window (same
            split as training) is the baseline every window is compared with.
        current: Traffic to monitor; defaults to ``reference`` itself, i.e. what happened after
            the training date in the same history. Only rows at or after the training date
            (the split cutoff) are monitored, in consecutive ``window_days`` windows.
        config: Thresholds and window size.

    Raises:
        ValueError: if the reference cannot be split or no monitored rows remain.
    """
    cfg = config or MonitorConfig()
    ref, cutoff = _reference_window(reference, cfg)
    cur = current if current is not None else reference
    if cur.feature_names != ref.feature_names:
        raise ValueError("current and reference feature tables have different columns")
    stamps = np.array([t.timestamp() for t in cur.created_at])
    after = cur.subset(stamps >= cutoff.timestamp())
    if len(after) == 0:
        raise ValueError(f"no monitored rows at or after the training date {cutoff:%Y-%m-%d}")

    ref_scores = predict_risk(model, ref.X)
    score_bins = Bins.fit(ref_scores, n_bins=cfg.n_bins)
    ref_score_shares = score_bins.shares(ref_scores)
    feature_bins = [Bins.fit(ref.X[:, j], n_bins=cfg.n_bins) for j in range(ref.X.shape[1])]
    ref_feature_shares = [b.shares(ref.X[:, j]) for j, b in enumerate(feature_bins)]
    ref_means = ref.X.mean(axis=0)

    cur_scores = predict_risk(model, after.X)
    cur_stamps = np.array([t.timestamp() for t in after.created_at])
    windows: list[WindowReport] = []
    for lo, hi in _window_bounds(cutoff, after.created_at[-1], cfg.window_days):
        mask = (cur_stamps >= lo.timestamp()) & (cur_stamps < hi.timestamp())
        n = int(mask.sum())
        if n == 0:
            continue
        X, s = after.X[mask], cur_scores[mask]
        mature, y = after.mature[mask], after.y[mask]

        score_psi = psi_from_shares(ref_score_shares, score_bins.shares(s))
        drifts = sorted(
            (
                FeatureDrift(
                    feature=name,
                    psi=(psi := psi_from_shares(ref_feature_shares[j], b.shares(X[:, j]))),
                    reference_mean=float(ref_means[j]),
                    current_mean=float(X[:, j].mean()),
                    status=min(cfg.psi_status(psi), Status.WARN),
                )
                for j, (name, b) in enumerate(zip(ref.feature_names, feature_bins, strict=True))
            ),
            key=lambda d: d.psi,
            reverse=True,
        )
        calibration = (
            calibration_check(y[mature], s[mature], cfg)
            if int(mature.sum()) >= cfg.min_labeled_rows
            else None
        )

        reasons: list[str] = []
        if n < cfg.min_window_rows:
            drift_status = Status.PENDING
            reasons.append(f"only {n} rows (< {cfg.min_window_rows}); drift not judged")
        else:
            drift_status = cfg.psi_status(score_psi)
            if drift_status is not Status.OK:
                reasons.append(f"score PSI {score_psi:.2f}")
            moved = [d for d in drifts if d.status is not Status.OK]
            if moved:
                drift_status = max(drift_status, Status.WARN)
                names = ", ".join(d.feature for d in moved[:3])
                more = f" +{len(moved) - 3} more" if len(moved) > 3 else ""
                reasons.append(f"feature drift: {names}{more}")
        if calibration is not None and calibration.status is not Status.OK:
            direction = "under" if calibration.z > 0 else "over"
            reasons.append(
                f"{direction}-predicting disputes: {calibration.observed} observed vs. "
                f"{calibration.expected:.1f} expected (z = {calibration.z:+.1f})"
            )

        windows.append(
            WindowReport(
                start=lo,
                end=hi,
                rows=n,
                mean_score=float(s.mean()),
                score_psi=score_psi,
                noise_floor=psi_noise_floor(score_bins.count, len(ref), n),
                features=tuple(drifts),
                label_coverage=float(mature.mean()),
                calibration=calibration,
                drift_status=drift_status,
                reasons=tuple(reasons),
            )
        )
    return MonitorReport(
        config=cfg,
        reference_start=ref.created_at[0],
        reference_end=ref.created_at[-1],
        reference_rows=len(ref),
        reference_mean_score=float(ref_scores.mean()),
        reference_dispute_rate=ref.positive_rate,
        monitored_from=cutoff,
        windows=tuple(windows),
    )

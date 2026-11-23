"""Glue between the trained model, the feature matrix, dispute records and the reasoner."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from chargeback_prediction.explain.run import ExplainConfig, explain_transactions
from chargeback_prediction.modeling import Classifier, FeatureMatrix
from chargeback_prediction.reason_codes import DisputeCategory, ReasonCode, lookup
from chargeback_prediction.reasoning.playbook import Strategy, recommend
from chargeback_prediction.reasoning.prompt import DisputeContext
from chargeback_prediction.reasoning.reasoner import feature_values
from chargeback_prediction.synthetic.schema import Dispute


def parse_reason_code(value: str) -> ReasonCode:
    """``visa:13.2`` or ``mastercard:4837`` to a catalog entry.

    Raises:
        ValueError: for a malformed value, unknown network or unknown code.
    """
    network, sep, code = value.partition(":")
    if not sep or not network or not code:
        raise ValueError(f"reason code {value!r} must look like 'visa:13.2'")
    try:
        return lookup(network.strip().lower(), code)
    except KeyError as exc:
        raise ValueError(str(exc.args[0])) from None


def dispute_reason_codes(disputes: Iterable[Dispute]) -> dict[str, ReasonCode]:
    """Transaction id to the reason code of the (first) dispute filed on it."""
    out: dict[str, ReasonCode] = {}
    for d in sorted(disputes, key=lambda d: d.filed_at):
        out.setdefault(d.transaction_id, parse_reason_code(d.reason_code))
    return out


def build_contexts(
    model: Classifier,
    matrix: FeatureMatrix,
    reasons: Mapping[str, ReasonCode],
    config: ExplainConfig | None = None,
) -> list[DisputeContext]:
    """Explain each disputed transaction and attach its reason code and full feature row.

    Raises:
        KeyError: naming the first transaction id not found in ``matrix``.
    """
    ids = list(reasons)
    explanations = explain_transactions(model, matrix, ids, config)
    index = {txn: i for i, txn in enumerate(matrix.transaction_ids)}
    return [
        DisputeContext(
            explanation=exp,
            reason=reasons[exp.transaction_id],
            features=feature_values(matrix.feature_names, matrix.X[index[exp.transaction_id]]),
        )
        for exp in explanations
    ]


@dataclass(frozen=True, slots=True)
class BacktestRow:
    category: DisputeCategory
    strategy: Strategy
    disputes: int
    friendly_fraud: int
    """Disputes the generator labelled as filed by the real cardholder (ground truth)."""

    @property
    def friendly_share(self) -> float:
        return self.friendly_fraud / self.disputes if self.disputes else 0.0


def backtest_playbook(
    disputes: Iterable[Dispute], matrix: FeatureMatrix
) -> tuple[BacktestRow, ...]:
    """Apply the playbook to every dispute and compare with the synthetic ground truth.

    Only the synthetic data has an ``is_friendly_fraud`` label; in production the equivalent is
    representment win/loss outcomes, which arrive months later.
    """
    index = {txn: i for i, txn in enumerate(matrix.transaction_ids)}
    counts: Counter[tuple[DisputeCategory, Strategy]] = Counter()
    friendly: Counter[tuple[DisputeCategory, Strategy]] = Counter()
    for d in disputes:
        if d.transaction_id not in index:
            continue
        reason = parse_reason_code(d.reason_code)
        row = feature_values(matrix.feature_names, matrix.X[index[d.transaction_id]])
        key = (reason.category, recommend(reason, row).strategy)
        counts[key] += 1
        friendly[key] += d.is_friendly_fraud
    order = {s: i for i, s in enumerate(Strategy)}
    return tuple(
        BacktestRow(cat, strat, n, friendly[(cat, strat)])
        for (cat, strat), n in sorted(counts.items(), key=lambda kv: (kv[0][0], order[kv[0][1]]))
    )


def render_backtest(rows: Iterable[BacktestRow]) -> str:
    lines = [
        "| Category | Playbook strategy | Disputes | Friendly fraud (ground truth) |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r.category} | {r.strategy} | {r.disputes:,} | "
        f"{r.friendly_fraud:,} ({r.friendly_share:.0%}) |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"

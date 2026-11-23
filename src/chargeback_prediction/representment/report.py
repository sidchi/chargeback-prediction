"""Portfolio view: how many disputes the assembler can back with data, per reason code.

The table compares each readiness verdict with the synthetic ground truth
(``Dispute.is_friendly_fraud``). That label is read **here only**, for evaluation; the
assembler itself never sees it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from chargeback_prediction.representment.packet import Readiness, ResponsePacket
from chargeback_prediction.synthetic.schema import Dispute


@dataclass(frozen=True, slots=True)
class ReadinessRow:
    reason_code: str
    readiness: Readiness
    disputes: int
    friendly_fraud: int

    @property
    def friendly_share(self) -> float:
        return self.friendly_fraud / self.disputes if self.disputes else 0.0


def readiness_table(
    packets: Iterable[ResponsePacket], disputes: Iterable[Dispute]
) -> tuple[ReadinessRow, ...]:
    """Disputes per (reason code, readiness), with the friendly-fraud count from ground truth."""
    friendly = {d.dispute_id: d.is_friendly_fraud for d in disputes}
    counts: Counter[tuple[str, Readiness]] = Counter()
    ff: Counter[tuple[str, Readiness]] = Counter()
    for p in packets:
        key = (p.reason.key, p.readiness)
        counts[key] += 1
        ff[key] += friendly.get(p.dispute_id, False)
    order = {r: i for i, r in enumerate(Readiness)}
    return tuple(
        ReadinessRow(code, ready, n, ff[(code, ready)])
        for (code, ready), n in sorted(counts.items(), key=lambda kv: (kv[0][0], order[kv[0][1]]))
    )


def render_readiness(rows: Iterable[ReadinessRow]) -> str:
    lines = [
        "| Reason code | Readiness | Disputes | Friendly fraud (ground truth) |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r.reason_code} | {r.readiness} | {r.disputes:,} | "
        f"{r.friendly_fraud:,} ({r.friendly_share:.0%}) |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"

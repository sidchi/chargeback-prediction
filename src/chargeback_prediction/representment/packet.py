"""The draft response packet: exhibits, readiness verdict, rebuttal and rendering.

A packet has two audiences. The **exhibits** and **rebuttal** are written to be sent to the
issuer (via the acquirer). The **adverse facts** and **internal notes** are for the analyst only:
they explain why a packet should not be sent, or context (such as the customer's prior disputes)
that must never appear in a submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from chargeback_prediction.reason_codes import ReasonCode
from chargeback_prediction.representment.requirements import ExhibitKind


class ExhibitStatus(StrEnum):
    FOUND = "found"
    """Assembled from the data and supports the merchant's case."""
    WEAK = "weak"
    """Assembled, but falls short of what the reason code needs (e.g. too few prior charges)."""
    MISSING = "missing"
    """The data holds nothing for this exhibit."""
    ATTACH = "attach"
    """Lives in a merchant system outside the dataset; the analyst must attach it."""


class Readiness(StrEnum):
    READY = "ready"
    """Every required exhibit was assembled from the data."""
    NEEDS_ATTACHMENTS = "needs_attachments"
    """Data-backed exhibits are in place; required external attachments are outstanding."""
    WEAK = "weak"
    """A required exhibit is weak or missing: represent only if attachments make up for it."""
    DO_NOT_REPRESENT = "do_not_represent"
    """An adverse fact (e.g. charged after a cancellation request) means the merchant would lose."""


@dataclass(frozen=True, slots=True)
class Exhibit:
    kind: ExhibitKind
    status: ExhibitStatus
    required: bool
    why: str
    summary: str
    facts: tuple[tuple[str, str], ...] = ()
    """Label/value pairs shown in the packet, in display order."""
    transaction_ids: tuple[str, ...] = ()
    """Transactions this exhibit cites, so the analyst can pull receipts."""

    @property
    def title(self) -> str:
        return self.kind.label

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "title": self.title,
            "status": str(self.status),
            "required": self.required,
            "why": self.why,
            "summary": self.summary,
            "facts": dict(self.facts),
            "transaction_ids": list(self.transaction_ids),
            "source": self.kind.external_source,
        }


def readiness(exhibits: tuple[Exhibit, ...], adverse: tuple[str, ...]) -> Readiness:
    """Overall verdict: adverse facts dominate, then gaps in the data, then attachments."""
    if adverse:
        return Readiness.DO_NOT_REPRESENT
    required = [e for e in exhibits if e.required]
    if any(e.status in (ExhibitStatus.WEAK, ExhibitStatus.MISSING) for e in required):
        return Readiness.WEAK
    if any(e.status is ExhibitStatus.ATTACH for e in required):
        return Readiness.NEEDS_ATTACHMENTS
    return Readiness.READY


@dataclass(frozen=True, slots=True)
class ResponsePacket:
    dispute_id: str
    transaction_id: str
    reason: ReasonCode
    filed_at: datetime
    respond_by: datetime
    as_of: datetime
    """Point in time the evidence was assembled at; nothing later was read."""
    amount_cents: int
    currency: str
    exhibits: tuple[Exhibit, ...]
    adverse: tuple[str, ...]
    internal_notes: tuple[str, ...]
    rebuttal: str

    @property
    def readiness(self) -> Readiness:
        return readiness(self.exhibits, self.adverse)

    def exhibit(self, kind: ExhibitKind) -> Exhibit:
        """Raises: KeyError if this reason code does not use ``kind``."""
        for e in self.exhibits:
            if e.kind is kind:
                return e
        raise KeyError(f"packet has no {kind} exhibit")

    @property
    def to_attach(self) -> tuple[Exhibit, ...]:
        return tuple(e for e in self.exhibits if e.status is ExhibitStatus.ATTACH)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispute_id": self.dispute_id,
            "transaction_id": self.transaction_id,
            "reason_code": self.reason.key,
            "reason_title": self.reason.title,
            "category": str(self.reason.category),
            "filed_at": self.filed_at.isoformat(),
            "respond_by": self.respond_by.isoformat(),
            "as_of": self.as_of.isoformat(),
            "amount": format_money(self.amount_cents, self.currency),
            "readiness": str(self.readiness),
            "rebuttal": self.rebuttal,
            "exhibits": [e.to_dict() for e in self.exhibits],
            "adverse": list(self.adverse),
            "internal_notes": list(self.internal_notes),
        }

    def render_markdown(self) -> str:
        """The draft packet as Markdown: a submission section, then an analyst-only section."""
        rc = self.reason
        lines = [
            f"# Representment draft — {self.dispute_id}",
            "",
            f"**Readiness: {_READINESS_LABEL[self.readiness]}**",
            "",
            "| | |",
            "|---|---|",
            f"| Reason code | {rc.network.title()} {rc.code} — {rc.title} |",
            f"| Disputed transaction | `{self.transaction_id}` |",
            f"| Amount | {format_money(self.amount_cents, self.currency)} |",
            f"| Filed | {self.filed_at:%Y-%m-%d} |",
            f"| Respond by | {self.respond_by:%Y-%m-%d} (check your acquirer's deadline) |",
            "",
            "## Rebuttal",
            "",
            self.rebuttal,
            "",
            "## Exhibits",
            "",
        ]
        for i, e in enumerate(self.exhibits, start=1):
            flag = "required" if e.required else "optional"
            lines += [
                f"### {i}. {e.title} — {_STATUS_LABEL[e.status]} ({flag})",
                "",
                f"*Why:* {e.why}.",
                "",
                e.summary,
            ]
            if e.facts:
                lines += ["", "| | |", "|---|---|"]
                lines += [f"| {label} | {value} |" for label, value in e.facts]
            if e.transaction_ids:
                shown = ", ".join(f"`{t}`" for t in e.transaction_ids[:10])
                more = len(e.transaction_ids) - 10
                lines += ["", f"Transactions: {shown}" + (f" and {more} more" if more > 0 else "")]
            lines.append("")

        lines += ["---", "", "## Analyst only — do not submit", ""]
        if self.adverse:
            lines += ["**Adverse facts:**", ""]
            lines += [f"- {a}" for a in self.adverse]
            lines.append("")
        if self.to_attach:
            lines += ["**Attach before submitting:**", ""]
            lines += [
                f"- [ ] {e.title} ({'required' if e.required else 'optional'}) — "
                f"from {e.kind.external_source}"
                for e in self.to_attach
            ]
            lines.append("")
        if self.internal_notes:
            lines += ["**Notes:**", ""]
            lines += [f"- {n}" for n in self.internal_notes]
            lines.append("")
        lines.append(
            f"_Assembled from data as of {self.as_of:%Y-%m-%d %H:%M} UTC. Synthetic data._"
        )
        return "\n".join(lines) + "\n"


_STATUS_LABEL = {
    ExhibitStatus.FOUND: "found",
    ExhibitStatus.WEAK: "weak",
    ExhibitStatus.MISSING: "missing",
    ExhibitStatus.ATTACH: "attach",
}

_READINESS_LABEL = {
    Readiness.READY: "ready to submit",
    Readiness.NEEDS_ATTACHMENTS: "needs attachments",
    Readiness.WEAK: "weak case",
    Readiness.DO_NOT_REPRESENT: "do not represent",
}


def format_money(cents: int, currency: str) -> str:
    """``2500, "USD"`` → ``"USD 25.00"``. Assumes a two-decimal currency, as the generator does."""
    sign = "-" if cents < 0 else ""
    return f"{currency} {sign}{abs(cents) // 100:,}.{abs(cents) % 100:02d}"

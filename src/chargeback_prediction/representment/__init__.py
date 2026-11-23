"""Representment evidence assembler: gather the evidence for a dispute and draft its packet.

:mod:`.requirements` says which exhibits each reason code needs, :mod:`.assembler` builds them
point in time from the raw tables, and :mod:`.packet` holds the result, its readiness verdict and
its Markdown / JSON renderings.
"""

from chargeback_prediction.representment.assembler import (
    CE3_MIN_TRANSACTIONS,
    CE3_WINDOW_DAYS,
    DEFAULT_RESPONSE_DAYS,
    EvidenceIndex,
    assemble,
    assemble_all,
)
from chargeback_prediction.representment.packet import (
    Exhibit,
    ExhibitStatus,
    Readiness,
    ResponsePacket,
    format_money,
    readiness,
)
from chargeback_prediction.representment.report import (
    ReadinessRow,
    readiness_table,
    render_readiness,
)
from chargeback_prediction.representment.requirements import (
    ExhibitKind,
    Requirement,
    requirements_for,
)

__all__ = [
    "CE3_MIN_TRANSACTIONS",
    "CE3_WINDOW_DAYS",
    "DEFAULT_RESPONSE_DAYS",
    "EvidenceIndex",
    "Exhibit",
    "ExhibitKind",
    "ExhibitStatus",
    "Readiness",
    "ReadinessRow",
    "Requirement",
    "ResponsePacket",
    "assemble",
    "assemble_all",
    "format_money",
    "readiness",
    "readiness_table",
    "render_readiness",
    "requirements_for",
]

"""Synthetic customers, subscriptions, transactions and disputes for a CNP subscription merchant.

All data produced here is synthetic. See :mod:`chargeback_prediction.synthetic.generator` for
the behaviour patterns it simulates.
"""

from chargeback_prediction.synthetic.config import GeneratorConfig
from chargeback_prediction.synthetic.generator import DatasetSummary, SyntheticDataset, generate
from chargeback_prediction.synthetic.io import write_dataset
from chargeback_prediction.synthetic.schema import (
    CardUpdate,
    CardUpdateSource,
    Customer,
    Dispute,
    Persona,
    Plan,
    Subscription,
    Transaction,
    TransactionKind,
)

__all__ = [
    "CardUpdate",
    "CardUpdateSource",
    "Customer",
    "DatasetSummary",
    "Dispute",
    "GeneratorConfig",
    "Persona",
    "Plan",
    "Subscription",
    "SyntheticDataset",
    "Transaction",
    "TransactionKind",
    "generate",
    "write_dataset",
]

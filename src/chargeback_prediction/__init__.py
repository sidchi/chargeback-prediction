"""Chargeback prediction: score card transactions for dispute risk and explain each score.

All data used by this project is synthetic. No real cardholder or merchant data is ever used.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chargeback-prediction")
except PackageNotFoundError:  # pragma: no cover - only hit when running from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]

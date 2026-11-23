from __future__ import annotations

import pytest

from chargeback_prediction.reason_codes import (
    DisputeCategory,
    Network,
    all_reason_codes,
    by_category,
    lookup,
)


def test_catalog_covers_roadmap_codes() -> None:
    keys = {rc.key for rc in all_reason_codes()}
    assert keys == {"visa:10.4", "visa:13.1", "visa:13.2", "mastercard:4837", "mastercard:4853"}


def test_keys_are_unique() -> None:
    keys = [rc.key for rc in all_reason_codes()]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    ("network", "code", "category"),
    [
        (Network.VISA, "10.4", DisputeCategory.FRAUD),
        ("visa", "13.2", DisputeCategory.CONSUMER),
        ("mastercard", " 4837 ", DisputeCategory.FRAUD),
    ],
)
def test_lookup(network: Network | str, code: str, category: DisputeCategory) -> None:
    assert lookup(network, code).category is category


def test_lookup_unknown_code_raises_key_error() -> None:
    with pytest.raises(KeyError, match=r"visa:99\.9"):
        lookup(Network.VISA, "99.9")


def test_lookup_unknown_network_raises_value_error() -> None:
    with pytest.raises(ValueError, match="amex"):
        lookup("amex", "C08")


def test_by_category_partitions_catalog() -> None:
    total = sum(len(by_category(c)) for c in DisputeCategory)
    assert total == len(all_reason_codes())
    assert all(rc.category is DisputeCategory.FRAUD for rc in by_category(DisputeCategory.FRAUD))


def test_reason_code_is_immutable() -> None:
    rc = lookup(Network.VISA, "10.4")
    with pytest.raises(AttributeError):
        rc.code = "10.5"  # type: ignore[misc]

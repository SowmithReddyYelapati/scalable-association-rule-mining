from collections.abc import Iterable

from src.arm.transaction import Itemset
from src.arm.support import support


def filter_frequent_itemsets(
    transactions: Iterable[Itemset],
    candidates: Iterable[Itemset],
    min_support: float,
) -> dict[Itemset, float]:
    """Return itemsets whose support meets the minimum threshold."""
    if not 0.0 <= min_support <= 1.0:
        raise ValueError("min_support must be between 0.0 and 1.0")

    transactions = list(transactions)

    return {
        itemset: support(transactions, itemset)
        for itemset in candidates
        if support(transactions, itemset) >= min_support
    }
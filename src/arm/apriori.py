from collections.abc import Iterable

from src.arm.frequent import filter_frequent_itemsets
from src.arm.transaction import Itemset


def apriori(
    transactions: Iterable[Itemset],
    min_support: float,
) -> dict[Itemset, float]:
    """Mine frequent 1-itemsets from the transaction database."""

    transactions = list(transactions)

    if not transactions:
        return {}

    items = set()

    for transaction in transactions:
        items.update(transaction)

    candidates = [frozenset([item]) for item in items]

    return filter_frequent_itemsets(
        transactions,
        candidates,
        min_support,
    )
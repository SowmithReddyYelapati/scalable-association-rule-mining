from collections.abc import Iterable

from src.arm.transaction import Itemset


def support_count(
    transactions: Iterable[Itemset],
    itemset: Itemset,
) -> int:
    """Return the number of transactions containing the itemset."""
    return sum(itemset.issubset(transaction) for transaction in transactions)


def support(
    transactions: Iterable[Itemset],
    itemset: Itemset,
) -> float:
    """Return the proportion of transactions containing the itemset."""
    transactions = list(transactions)

    if not transactions:
        return 0.0

    return support_count(transactions, itemset) / len(transactions)
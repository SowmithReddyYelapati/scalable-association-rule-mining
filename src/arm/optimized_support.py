from collections import defaultdict

from src.arm.transaction import Itemset


def build_transaction_index(
    transactions: list[Itemset],
) -> dict[str, set[int]]:
    """Map each item to the transaction IDs containing it."""

    index: dict[str, set[int]] = defaultdict(set)

    for transaction_id, transaction in enumerate(transactions):
        for item in transaction:
            index[item].add(transaction_id)

    return dict(index)


def count_itemset_occurrences(
    itemset: Itemset,
    transaction_index: dict[str, set[int]],
) -> int:
    """Count transactions containing the complete itemset."""

    if not itemset:
        return 0

    transaction_ids = [
        transaction_index.get(item, set())
        for item in itemset
    ]

    if any(not ids for ids in transaction_ids):
        return 0

    common_transactions = transaction_ids[0].copy()

    for ids in transaction_ids[1:]:
        common_transactions.intersection_update(ids)

        if not common_transactions:
            return 0

    return len(common_transactions)


def calculate_support_indexed(
    itemset: Itemset,
    transaction_index: dict[str, set[int]],
    transaction_count: int,
) -> float:
    """Calculate support using the transaction index."""

    if transaction_count == 0:
        return 0.0

    occurrences = count_itemset_occurrences(
        itemset,
        transaction_index,
    )

    return occurrences / transaction_count
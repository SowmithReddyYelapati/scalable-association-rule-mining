from collections.abc import Iterable

from src.arm.candidate import generate_candidates
from src.arm.frequent import filter_frequent_itemsets
from src.arm.optimized_support import (
    build_transaction_index,
    calculate_support_indexed,
)
from src.arm.pruning import prune_candidates
from src.arm.transaction import Itemset


def apriori(
    transactions: Iterable[Itemset],
    min_support: float,
) -> dict[Itemset, float]:
    """Mine all frequent itemsets using the Apriori algorithm."""

    transactions = list(transactions)

    if not transactions:
        return {}

    # Build the index once and reuse it for every candidate.
    transaction_index = build_transaction_index(transactions)

    items = set()

    for transaction in transactions:
        items.update(transaction)

    candidates = {
        frozenset([item])
        for item in items
    }

    all_frequent: dict[Itemset, float] = {}

    k = 1

    while candidates:
        frequent: dict[Itemset, float] = {}

        for candidate in candidates:
            support = calculate_support_indexed(
                candidate,
                transaction_index,
                len(transactions),
            )

            if support >= min_support:
                frequent[candidate] = support

        if not frequent:
            break

        all_frequent.update(frequent)

        k += 1

        next_candidates = generate_candidates(
            set(frequent),
            k,
        )

        candidates = prune_candidates(
            next_candidates,
            set(frequent),
        )

    return all_frequent
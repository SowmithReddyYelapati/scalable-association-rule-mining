from collections.abc import Iterable

from src.arm.candidate import generate_candidates
from src.arm.frequent import filter_frequent_itemsets
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
        frequent = filter_frequent_itemsets(
            transactions,
            candidates,
            min_support,
        )

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
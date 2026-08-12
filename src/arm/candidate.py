from collections.abc import Iterable

from src.arm.transaction import Itemset


def generate_candidates(
    frequent_itemsets: Iterable[Itemset],
    k: int,
) -> set[Itemset]:
    """Generate candidate k-itemsets from frequent (k-1)-itemsets."""

    if k < 2:
        raise ValueError("k must be at least 2")

    frequent_itemsets = list(frequent_itemsets)

    if not frequent_itemsets:
        return set()

    candidates: set[Itemset] = set()

    for i, first in enumerate(frequent_itemsets):
        for second in frequent_itemsets[i + 1:]:
            union = first | second

            if len(union) == k:
                candidates.add(union)

    return candidates
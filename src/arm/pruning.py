from collections.abc import Iterable
from itertools import combinations

from src.arm.transaction import Itemset


def prune_candidates(
    candidates: Iterable[Itemset],
    frequent_itemsets: Iterable[Itemset],
) -> set[Itemset]:
    """Remove candidates having a non-frequent (k-1)-subset."""

    frequent_itemsets = set(frequent_itemsets)
    valid_candidates: set[Itemset] = set()

    for candidate in candidates:
        subsets = (
            frozenset(subset)
            for subset in combinations(candidate, len(candidate) - 1)
        )

        if all(subset in frequent_itemsets for subset in subsets):
            valid_candidates.add(candidate)

    return valid_candidates
from typing import Iterable, FrozenSet


Itemset = FrozenSet[str]


def create_itemset(items: Iterable[str]) -> Itemset:
    """
    Create an immutable itemset from a collection of items.

    Items are normalized by removing surrounding whitespace
    and ignoring empty values.
    """
    return frozenset(
        item.strip()
        for item in items
        if item and item.strip()
    )
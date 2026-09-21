from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.arm.transaction import Itemset


@dataclass
class FPNode:
    """
    Node used by the FP-Tree.
    """

    item: object | None
    count: int = 0
    parent: FPNode | None = None
    children: dict[object, FPNode] = field(default_factory=dict)
    link: FPNode | None = None


def _build_item_counts(
    transactions: list[Itemset],
) -> dict[object, int]:

    counts: dict[object, int] = defaultdict(int)

    for transaction in transactions:

        for item in set(transaction):
            counts[item] += 1

    return dict(counts)


def _build_header_table(
    transactions: list[Itemset],
    min_support: float,
):

    if not transactions:
        return {}, {}

    item_counts = _build_item_counts(
        transactions
    )

    minimum_count = (
        min_support * len(transactions)
    )

    frequent_items = {
        item: count
        for item, count in item_counts.items()
        if count >= minimum_count
    }

    header = {}

    for item, count in frequent_items.items():
        header[item] = [count, None]

    return header, frequent_items


def _insert_transaction(
    transaction: list[object],
    root: FPNode,
    header,
) -> None:

    current = root

    for item in transaction:

        if item in current.children:

            child = current.children[item]
            child.count += 1

        else:

            child = FPNode(
                item=item,
                count=1,
                parent=current,
            )

            current.children[item] = child

            first_node = header[item][1]

            if first_node is None:

                header[item][1] = child

            else:

                last_node = first_node

                while last_node.link is not None:
                    last_node = last_node.link

                last_node.link = child

        current = child


def _build_fp_tree(
    transactions: list[Itemset],
    min_support: float,
):

    if not transactions:
        return None, {}

    header, frequent_counts = (
        _build_header_table(
            transactions,
            min_support,
        )
    )

    if not header:
        return None, {}

    root = FPNode(
        item=None,
        count=0,
        parent=None,
    )

    def item_order(item):

        return (
            -frequent_counts[item],
            str(item),
        )

    for transaction in transactions:

        filtered = [
            item
            for item in set(transaction)
            if item in frequent_counts
        ]

        if not filtered:
            continue

        ordered = sorted(
            filtered,
            key=item_order,
        )

        _insert_transaction(
            ordered,
            root,
            header,
        )

    return root, header


def _follow_node_links(
    first_node: FPNode | None,
):

    nodes = []

    current = first_node

    while current is not None:

        nodes.append(current)
        current = current.link

    return nodes


def _conditional_pattern_base(
    item,
    header,
):

    if item not in header:
        return []

    first_node = header[item][1]

    paths = []

    for node in _follow_node_links(
        first_node
    ):

        path = []

        parent = node.parent

        while (
            parent is not None
            and parent.item is not None
        ):

            path.append(
                parent.item
            )

            parent = parent.parent

        if path:

            path.reverse()

            paths.append(
                (
                    path,
                    node.count,
                )
            )

    return paths


def _build_conditional_tree(
    pattern_base,
    min_count,
):

    if not pattern_base:
        return None, {}

    item_counts = defaultdict(int)

    for path, count in pattern_base:

        for item in set(path):
            item_counts[item] += count

    frequent_counts = {
        item: count
        for item, count in item_counts.items()
        if count >= min_count
    }

    if not frequent_counts:
        return None, {}

    header = {}

    for item, count in frequent_counts.items():
        header[item] = [count, None]

    root = FPNode(
        item=None,
        count=0,
        parent=None,
    )

    def item_order(item):

        return (
            -frequent_counts[item],
            str(item),
        )

    for path, path_count in pattern_base:

        filtered = [
            item
            for item in set(path)
            if item in frequent_counts
        ]

        if not filtered:
            continue

        ordered = sorted(
            filtered,
            key=item_order,
        )

        current = root

        for item in ordered:

            if item in current.children:

                child = current.children[item]
                child.count += path_count

            else:

                child = FPNode(
                    item=item,
                    count=path_count,
                    parent=current,
                )

                current.children[item] = child

                first_node = header[item][1]

                if first_node is None:

                    header[item][1] = child

                else:

                    last_node = first_node

                    while last_node.link is not None:
                        last_node = last_node.link

                    last_node.link = child

            current = child

    return root, header


def _mine_tree(
    header,
    min_count,
    prefix,
    results,
    transaction_count,
):

    items = sorted(
        header.keys(),
        key=lambda item: (
            header[item][0],
            str(item),
        ),
    )

    for item in items:

        new_pattern = frozenset(
            set(prefix) | {item}
        )

        count = header[item][0]

        support = (
            count / transaction_count
        )

        results[new_pattern] = support

        pattern_base = (
            _conditional_pattern_base(
                item,
                header,
            )
        )

        if not pattern_base:
            continue

        (
            conditional_root,
            conditional_header,
        ) = _build_conditional_tree(
            pattern_base,
            min_count,
        )

        if conditional_root is None:
            continue

        _mine_tree(
            conditional_header,
            min_count,
            new_pattern,
            results,
            transaction_count,
        )


def fp_growth(
    transactions: Iterable[Itemset],
    min_support: float,
) -> dict[Itemset, float]:
    """
    Mine frequent itemsets using FP-Growth.

    min_support is represented as a fraction:

        0.01 = 1%
        0.05 = 5%
        0.10 = 10%
    """

    if not 0 < min_support <= 1:

        raise ValueError(
            "min_support must be between 0 and 1."
        )

    transactions = [
        frozenset(transaction)
        for transaction in transactions
        if transaction
    ]

    if not transactions:
        return {}

    root, header = _build_fp_tree(
        transactions,
        min_support,
    )

    if root is None or not header:
        return {}

    results = {}

    minimum_count = (
        min_support
        * len(transactions)
    )

    _mine_tree(
        header=header,
        min_count=minimum_count,
        prefix=frozenset(),
        results=results,
        transaction_count=len(
            transactions
        ),
    )

    return results
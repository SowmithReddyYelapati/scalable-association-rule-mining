from src.arm.optimized_support import (
    build_transaction_index,
    calculate_support_indexed,
    count_itemset_occurrences,
)
from src.arm.transaction import create_itemset


def test_build_transaction_index():
    transactions = [
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
        create_itemset(["B", "C"]),
    ]

    index = build_transaction_index(transactions)

    assert index["A"] == {0, 1}
    assert index["B"] == {0, 2}
    assert index["C"] == {1, 2}


def test_count_single_item():
    transactions = [
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
        create_itemset(["B", "C"]),
    ]

    index = build_transaction_index(transactions)

    assert count_itemset_occurrences(
        create_itemset(["A"]),
        index,
    ) == 2


def test_count_itemset():
    transactions = [
        create_itemset(["A", "B", "C"]),
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
        create_itemset(["B", "C"]),
    ]

    index = build_transaction_index(transactions)

    assert count_itemset_occurrences(
        create_itemset(["A", "B"]),
        index,
    ) == 2


def test_calculate_support_indexed():
    transactions = [
        create_itemset(["A", "B", "C"]),
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
        create_itemset(["B", "C"]),
    ]

    index = build_transaction_index(transactions)

    support = calculate_support_indexed(
        create_itemset(["A", "B"]),
        index,
        len(transactions),
    )

    assert support == 0.50


def test_missing_item_has_zero_support():
    transactions = [
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
    ]

    index = build_transaction_index(transactions)

    assert calculate_support_indexed(
        create_itemset(["X"]),
        index,
        len(transactions),
    ) == 0.0


def test_empty_transactions():
    index = build_transaction_index([])

    assert calculate_support_indexed(
        create_itemset(["A"]),
        index,
        0,
    ) == 0.0
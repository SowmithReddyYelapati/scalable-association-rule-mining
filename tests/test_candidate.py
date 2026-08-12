import pytest

from src.arm.candidate import generate_candidates
from src.arm.transaction import create_itemset


def test_generate_two_item_candidates():
    frequent_itemsets = [
        create_itemset(["A"]),
        create_itemset(["B"]),
        create_itemset(["C"]),
    ]

    candidates = generate_candidates(frequent_itemsets, 2)

    assert candidates == {
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
        create_itemset(["B", "C"]),
    }


def test_generate_three_item_candidates():
    frequent_itemsets = [
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
        create_itemset(["B", "C"]),
    ]

    candidates = generate_candidates(frequent_itemsets, 3)

    assert candidates == {
        create_itemset(["A", "B", "C"]),
    }


def test_empty_input():
    assert generate_candidates([], 2) == set()


def test_single_itemset():
    frequent_itemsets = [
        create_itemset(["A"]),
    ]

    assert generate_candidates(frequent_itemsets, 2) == set()


@pytest.mark.parametrize("invalid_k", [0, 1, -1])
def test_invalid_k(invalid_k):
    with pytest.raises(ValueError):
        generate_candidates([], invalid_k)
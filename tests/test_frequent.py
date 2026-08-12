import pytest

from src.arm.frequent import filter_frequent_itemsets
from src.arm.transaction import create_itemset


def create_test_transactions():
    return [
        create_itemset(["Milk", "Bread"]),
        create_itemset(["Milk", "Butter"]),
        create_itemset(["Milk", "Bread"]),
        create_itemset(["Bread", "Butter"]),
    ]


def test_filter_frequent_itemsets():
    transactions = create_test_transactions()

    candidates = [
        create_itemset(["Milk"]),
        create_itemset(["Bread"]),
        create_itemset(["Butter"]),
        create_itemset(["Eggs"]),
    ]

    frequent = filter_frequent_itemsets(
        transactions,
        candidates,
        min_support=0.50,
    )

    assert frequent == {
        create_itemset(["Milk"]): 0.75,
        create_itemset(["Bread"]): 0.75,
        create_itemset(["Butter"]): 0.50,
    }


def test_minimum_support_boundary_is_included():
    transactions = create_test_transactions()

    candidates = [create_itemset(["Butter"])]

    frequent = filter_frequent_itemsets(
        transactions,
        candidates,
        min_support=0.50,
    )

    assert create_itemset(["Butter"]) in frequent


def test_itemset_below_minimum_support_is_removed():
    transactions = create_test_transactions()

    candidates = [create_itemset(["Eggs"])]

    frequent = filter_frequent_itemsets(
        transactions,
        candidates,
        min_support=0.50,
    )

    assert create_itemset(["Eggs"]) not in frequent


@pytest.mark.parametrize("invalid_support", [-0.1, 1.1])
def test_invalid_minimum_support(invalid_support):
    with pytest.raises(ValueError):
        filter_frequent_itemsets([], [], invalid_support)
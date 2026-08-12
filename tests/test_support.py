from src.arm.support import support, support_count
from src.arm.transaction import create_itemset


def create_test_transactions():
    return [
        create_itemset(["Milk", "Bread"]),
        create_itemset(["Milk", "Butter"]),
        create_itemset(["Milk", "Bread"]),
        create_itemset(["Bread", "Butter"]),
    ]


def test_support_count():
    transactions = create_test_transactions()
    itemset = create_itemset(["Milk", "Bread"])

    assert support_count(transactions, itemset) == 2


def test_support():
    transactions = create_test_transactions()
    itemset = create_itemset(["Milk", "Bread"])

    assert support(transactions, itemset) == 0.5


def test_itemset_not_present():
    transactions = create_test_transactions()
    itemset = create_itemset(["Milk", "Eggs"])

    assert support_count(transactions, itemset) == 0
    assert support(transactions, itemset) == 0.0


def test_empty_transactions():
    itemset = create_itemset(["Milk", "Bread"])

    assert support([], itemset) == 0.0
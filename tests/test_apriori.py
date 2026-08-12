from src.arm.apriori import apriori
from src.arm.transaction import create_itemset


def test_apriori_finds_frequent_single_items():
    transactions = [
        create_itemset(["A", "B", "C"]),
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
        create_itemset(["A", "B"]),
    ]

    result = apriori(transactions, min_support=0.50)

    assert result == {
        create_itemset(["A"]): 1.00,
        create_itemset(["B"]): 0.75,
        create_itemset(["C"]): 0.50,
    }


def test_apriori_filters_infrequent_single_items():
    transactions = [
        create_itemset(["A", "B"]),
        create_itemset(["A"]),
        create_itemset(["A"]),
        create_itemset(["C"]),
    ]

    result = apriori(transactions, min_support=0.75)

    assert result == {
        create_itemset(["A"]): 0.75,
    }


def test_apriori_empty_transactions():
    assert apriori([], min_support=0.50) == {}
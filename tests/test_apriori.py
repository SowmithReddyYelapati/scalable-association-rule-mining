from src.arm.apriori import apriori
from src.arm.transaction import create_itemset


def test_apriori_finds_frequent_single_items():
    transactions = [
        create_itemset(["A", "B", "C"]),
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
        create_itemset(["A", "B"]),
    ]

    result = apriori(
        transactions,
        min_support=0.50,
    )

    assert result[create_itemset(["A"])] == 1.00
    assert result[create_itemset(["B"])] == 0.75
    assert result[create_itemset(["C"])] == 0.50


def test_apriori_filters_infrequent_single_items():
    transactions = [
        create_itemset(["A", "B"]),
        create_itemset(["A"]),
        create_itemset(["A"]),
        create_itemset(["C"]),
    ]

    result = apriori(
        transactions,
        min_support=0.75,
    )

    assert result == {
        create_itemset(["A"]): 0.75,
    }


def test_apriori_empty_transactions():
    assert apriori([], min_support=0.50) == {}


def test_apriori_finds_two_itemsets():
    transactions = [
        create_itemset(["A", "B", "C"]),
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
        create_itemset(["A", "B"]),
    ]

    result = apriori(
        transactions,
        min_support=0.50,
    )

    assert create_itemset(["A", "B"]) in result
    assert create_itemset(["A", "C"]) in result
    assert create_itemset(["B", "C"]) not in result


def test_apriori_finds_three_itemsets():
    transactions = [
        create_itemset(["A", "B", "C"]),
        create_itemset(["A", "B", "C"]),
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
    ]

    result = apriori(
        transactions,
        min_support=0.50,
    )

    assert create_itemset(["A", "B"]) in result
    assert create_itemset(["A", "C"]) in result
    assert create_itemset(["B", "C"]) in result
    assert create_itemset(["A", "B", "C"]) in result
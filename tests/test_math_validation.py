"""
Mathematical validation test suite for Apriori, Support, Confidence, and Lift.

Performs hand-verified theoretical calculations on a known benchmark dataset
and asserts exact equality with the algorithm output.
"""

from src.arm.apriori import apriori
from src.arm.rules import generate_rules
from src.arm.transaction import Itemset


def test_math_validation_manual_dataset():
    """
    Manually verified dataset:
    T1 = {Bread, Milk}
    T2 = {Bread, Butter}
    T3 = {Bread, Milk, Butter}
    T4 = {Milk, Butter}

    Total transactions |T| = 4

    Hand Calculations:
    - Support({Bread})              = 3 / 4 = 0.75
    - Support({Milk})               = 3 / 4 = 0.75
    - Support({Butter})             = 3 / 4 = 0.75
    - Support({Bread, Milk})        = 2 / 4 = 0.50
    - Support({Bread, Butter})      = 2 / 4 = 0.50
    - Support({Milk, Butter})       = 2 / 4 = 0.50

    Rule: Bread -> Milk
    - Support(Bread -> Milk)        = Support({Bread, Milk}) = 0.50
    - Confidence(Bread -> Milk)     = 0.50 / 0.75 = 2/3 ≈ 0.6666667
    - Lift(Bread -> Milk)           = 0.50 / (0.75 * 0.75) = 0.50 / 0.5625 = 8/9 ≈ 0.8888889
    """
    transactions: list[Itemset] = [
        frozenset({"Bread", "Milk"}),
        frozenset({"Bread", "Butter"}),
        frozenset({"Bread", "Milk", "Butter"}),
        frozenset({"Milk", "Butter"}),
    ]

    # Run Apriori with min_support = 0.40 (40%)
    frequent_itemsets = apriori(transactions, min_support=0.40)

    # 1. Verify Frequent Itemsets & Support values
    assert frequent_itemsets[frozenset({"Bread"})] == pytest.approx(0.75)
    assert frequent_itemsets[frozenset({"Milk"})] == pytest.approx(0.75)
    assert frequent_itemsets[frozenset({"Butter"})] == pytest.approx(0.75)
    assert frequent_itemsets[frozenset({"Bread", "Milk"})] == pytest.approx(0.50)
    assert frequent_itemsets[frozenset({"Bread", "Butter"})] == pytest.approx(0.50)
    assert frequent_itemsets[frozenset({"Milk", "Butter"})] == pytest.approx(0.50)

    # 2. Generate Rules with min_confidence = 0.50 (50%)
    rules = generate_rules(frequent_itemsets, min_confidence=0.50)

    # Find the rule Bread -> Milk
    bread_to_milk = [
        r for r in rules
        if r.antecedent == frozenset({"Bread"}) and r.consequent == frozenset({"Milk"})
    ]

    assert len(bread_to_milk) == 1
    rule = bread_to_milk[0]

    # 3. Assert Support, Confidence, and Lift match theoretical math
    assert rule.support == pytest.approx(0.50)
    assert rule.confidence == pytest.approx(2 / 3)
    assert rule.lift == pytest.approx(8 / 9)


import pytest

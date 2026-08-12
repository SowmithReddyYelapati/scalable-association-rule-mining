import pytest

from src.arm.rules import (
    AssociationRule,
    calculate_confidence,
    calculate_lift,
    generate_rule_candidates,
    generate_rules,
)
from src.arm.transaction import create_itemset


def test_generate_rules_from_two_itemset():
    itemset = create_itemset(["A", "B"])

    result = generate_rule_candidates(itemset)

    assert set(result) == {
        (
            create_itemset(["A"]),
            create_itemset(["B"]),
        ),
        (
            create_itemset(["B"]),
            create_itemset(["A"]),
        ),
    }


def test_generate_rules_from_three_itemset():
    itemset = create_itemset(["A", "B", "C"])

    result = generate_rule_candidates(itemset)

    assert len(result) == 6


def test_single_itemset_produces_no_rules():
    itemset = create_itemset(["A"])

    assert generate_rule_candidates(itemset) == []


def test_empty_itemset_produces_no_rules():
    assert generate_rule_candidates(frozenset()) == []


def test_calculate_confidence():
    support_data = {
        create_itemset(["A"]): 0.80,
        create_itemset(["B"]): 0.70,
        create_itemset(["A", "B"]): 0.60,
    }

    confidence = calculate_confidence(
        create_itemset(["A"]),
        create_itemset(["B"]),
        support_data,
    )

    assert confidence == pytest.approx(0.75)


def test_calculate_confidence_with_zero_antecedent_support():
    support_data = {
        create_itemset(["A"]): 0.0,
        create_itemset(["A", "B"]): 0.0,
    }

    confidence = calculate_confidence(
        create_itemset(["A"]),
        create_itemset(["B"]),
        support_data,
    )

    assert confidence == 0.0


def test_calculate_lift():
    support_data = {
        create_itemset(["A"]): 0.80,
        create_itemset(["B"]): 0.70,
        create_itemset(["A", "B"]): 0.60,
    }

    lift = calculate_lift(
        create_itemset(["A"]),
        create_itemset(["B"]),
        support_data,
    )

    assert lift == pytest.approx(0.60 / (0.80 * 0.70))


def test_calculate_lift_with_zero_support():
    support_data = {
        create_itemset(["A"]): 0.0,
        create_itemset(["B"]): 0.70,
        create_itemset(["A", "B"]): 0.0,
    }

    lift = calculate_lift(
        create_itemset(["A"]),
        create_itemset(["B"]),
        support_data,
    )

    assert lift == 0.0


def test_generate_rules():
    frequent_itemsets = {
        create_itemset(["A"]): 0.80,
        create_itemset(["B"]): 0.70,
        create_itemset(["A", "B"]): 0.60,
    }

    rules = generate_rules(
        frequent_itemsets,
        min_confidence=0.80,
    )

    assert len(rules) == 1

    rule = rules[0]

    assert rule.antecedent == create_itemset(["B"])
    assert rule.consequent == create_itemset(["A"])
    assert rule.support == pytest.approx(0.60)
    assert rule.confidence == pytest.approx(0.60 / 0.70)
    assert rule.lift == pytest.approx(0.60 / (0.80 * 0.70))


def test_generate_rules_filters_by_confidence():
    frequent_itemsets = {
        create_itemset(["A"]): 0.80,
        create_itemset(["B"]): 0.70,
        create_itemset(["A", "B"]): 0.60,
    }

    rules = generate_rules(
        frequent_itemsets,
        min_confidence=0.90,
    )

    assert rules == []
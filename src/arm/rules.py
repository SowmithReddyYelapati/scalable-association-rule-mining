from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from src.arm.transaction import Itemset


@dataclass(frozen=True)
class AssociationRule:
    antecedent: Itemset
    consequent: Itemset
    support: float
    confidence: float
    lift: float


def generate_rules(
    frequent_itemsets: dict[Itemset, float],
    min_confidence: float,
) -> list[AssociationRule]:
    """Generate association rules from frequent itemsets."""

    rules: list[AssociationRule] = []

    for itemset, itemset_support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        for antecedent, consequent in generate_rule_candidates(itemset):
            confidence = calculate_confidence(
                antecedent,
                consequent,
                frequent_itemsets,
            )

            if confidence < min_confidence:
                continue

            lift = calculate_lift(
                antecedent,
                consequent,
                frequent_itemsets,
            )

            rules.append(
                AssociationRule(
                    antecedent=antecedent,
                    consequent=consequent,
                    support=itemset_support,
                    confidence=confidence,
                    lift=lift,
                )
            )

    return rules


def generate_rule_candidates(
    frequent_itemset: Itemset,
) -> list[tuple[Itemset, Itemset]]:
    """Generate antecedent/consequent pairs from a frequent itemset."""

    if len(frequent_itemset) < 2:
        return []

    items = list(frequent_itemset)
    rules = []

    for size in range(1, len(items)):
        for antecedent_items in combinations(items, size):
            antecedent = frozenset(antecedent_items)
            consequent = frequent_itemset - antecedent

            rules.append((antecedent, consequent))

    return rules


def calculate_confidence(
    antecedent: Itemset,
    consequent: Itemset,
    support_data: dict[Itemset, float],
) -> float:
    """Calculate confidence for an association rule."""

    union = antecedent | consequent

    antecedent_support = support_data[antecedent]
    union_support = support_data[union]

    if antecedent_support == 0:
        return 0.0

    return union_support / antecedent_support


def calculate_lift(
    antecedent: Itemset,
    consequent: Itemset,
    support_data: dict[Itemset, float],
) -> float:
    """Calculate lift for an association rule."""

    union = antecedent | consequent

    antecedent_support = support_data[antecedent]
    consequent_support = support_data[consequent]
    union_support = support_data[union]

    denominator = antecedent_support * consequent_support

    if denominator == 0:
        return 0.0

    return union_support / denominator
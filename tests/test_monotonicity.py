import pytest
from src.arm.fp_growth import fp_growth
from src.arm.apriori import apriori
from src.arm.rules import generate_rules


@pytest.fixture
def transactions_dataset():
    """Deterministic transaction dataset for monotonicity testing."""
    return [
        {"milk", "bread", "butter"},
        {"bread", "butter", "cheese"},
        {"milk", "bread", "cheese"},
        {"milk", "butter", "eggs"},
        {"bread", "butter", "eggs"},
        {"milk", "bread", "butter", "cheese", "eggs"},
        {"milk", "bread"},
        {"milk", "cheese"},
        {"bread", "butter"},
        {"milk", "butter"},
        {"bread", "eggs"},
        {"milk", "bread", "butter"},
    ]


def test_fp_growth_support_monotonicity(transactions_dataset):
    """
    Verify support monotonicity for FP-Growth:
    Lower support threshold must NOT produce fewer frequent itemsets
    than a higher support threshold.
    """
    support_thresholds = [0.10, 0.05, 0.02, 0.01, 0.001]
    itemset_counts = []
    itemset_sets = []

    for supp in support_thresholds:
        itemsets = fp_growth(transactions_dataset, min_support=supp)
        itemset_counts.append(len(itemsets))
        itemset_sets.append(set(itemsets.keys()))

    # As support decreases, number of frequent itemsets must be non-decreasing
    for i in range(len(support_thresholds) - 1):
        higher_supp = support_thresholds[i]
        lower_supp = support_thresholds[i + 1]
        higher_count = itemset_counts[i]
        lower_count = itemset_counts[i + 1]

        assert lower_count >= higher_count, (
            f"Monotonicity violation: support {lower_supp} produced {lower_count} itemsets, "
            f"which is fewer than support {higher_supp} producing {higher_count} itemsets."
        )

        # The set of itemsets at higher support must be a subset of itemsets at lower support
        assert itemset_sets[i].issubset(itemset_sets[i + 1]), (
            f"Itemset subset violation between support {higher_supp} and {lower_supp}"
        )


def test_apriori_support_monotonicity(transactions_dataset):
    """
    Verify support monotonicity for Apriori:
    Lower support threshold must NOT produce fewer frequent itemsets.
    """
    support_thresholds = [0.10, 0.05, 0.02, 0.01, 0.001]
    itemset_counts = []
    itemset_sets = []

    for supp in support_thresholds:
        itemsets = apriori(transactions_dataset, min_support=supp)
        itemset_counts.append(len(itemsets))
        itemset_sets.append(set(itemsets.keys()))

    for i in range(len(support_thresholds) - 1):
        assert itemset_counts[i + 1] >= itemset_counts[i]
        assert itemset_sets[i].issubset(itemset_sets[i + 1])


def test_confidence_monotonicity(transactions_dataset):
    """
    Verify confidence monotonicity:
    Increasing confidence threshold must NOT produce MORE rules than a lower threshold.
    """
    frequent_itemsets = fp_growth(transactions_dataset, min_support=0.05)
    confidence_thresholds = [0.10, 0.50, 0.60, 0.90, 1.00]

    rule_counts = []
    rule_sets = []

    for conf in confidence_thresholds:
        rules = generate_rules(frequent_itemsets, min_confidence=conf)
        rule_counts.append(len(rules))
        # Unique representation for rule subset check: (antecedent, consequent)
        rule_identities = {(r.antecedent, r.consequent) for r in rules}
        rule_sets.append(rule_identities)

    # As confidence increases, number of rules must be non-increasing
    for i in range(len(confidence_thresholds) - 1):
        lower_conf = confidence_thresholds[i]
        higher_conf = confidence_thresholds[i + 1]
        lower_count = rule_counts[i]
        higher_count = rule_counts[i + 1]

        assert higher_count <= lower_count, (
            f"Confidence monotonicity violation: confidence {higher_conf} produced {higher_count} rules, "
            f"which is MORE than confidence {lower_conf} producing {lower_count} rules."
        )

        # Rules at higher confidence must be a subset of rules at lower confidence
        assert rule_sets[i + 1].issubset(rule_sets[i]), (
            f"Rule subset violation between confidence {lower_conf} and {higher_conf}"
        )

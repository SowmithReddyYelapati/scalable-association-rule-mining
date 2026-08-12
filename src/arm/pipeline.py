from pathlib import Path

from src.arm.apriori import apriori
from src.arm.loader import load_transactions
from src.arm.rules import AssociationRule, generate_rules


def mine_dataset(
    path: str | Path,
    min_support: float,
    min_confidence: float,
) -> tuple[dict, list[AssociationRule]]:
    """Load a dataset and mine frequent itemsets and association rules."""

    transactions = load_transactions(path)

    frequent_itemsets = apriori(
        transactions,
        min_support,
    )

    rules = generate_rules(
        frequent_itemsets,
        min_confidence,
    )

    return frequent_itemsets, rules
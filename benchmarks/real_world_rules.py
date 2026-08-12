import csv

from src.arm.apriori import apriori
from src.arm.product_mapping import load_product_mapping
from src.arm.retail_loader import load_online_retail
from src.arm.rules import generate_rules


DATASET = "data/Online Retail.xlsx"
OUTPUT_FILE = "benchmarks/real_world_rules.csv"

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.50


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load real-world transactions
    # ---------------------------------------------------------
    print("Loading Online Retail dataset...")

    transactions = load_online_retail(DATASET)

    print(f"Transactions: {len(transactions):,}")

    # ---------------------------------------------------------
    # 2. Run Apriori
    # ---------------------------------------------------------
    print("\nRunning Apriori...")

    frequent_itemsets = apriori(
        transactions,
        min_support=MIN_SUPPORT,
    )

    print(f"Frequent itemsets: {len(frequent_itemsets):,}")

    # ---------------------------------------------------------
    # 3. Generate association rules
    # ---------------------------------------------------------
    print("\nGenerating association rules...")

    rules = generate_rules(
        frequent_itemsets,
        min_confidence=MIN_CONFIDENCE,
    )

    print(f"Rules: {len(rules):,}")

    # ---------------------------------------------------------
    # 4. Load product-code → description mapping
    # ---------------------------------------------------------
    print("\nLoading product descriptions...")

    product_mapping = load_product_mapping(DATASET)

    # Sort rules by lift
    rules = sorted(
        rules,
        key=lambda rule: rule.lift,
        reverse=True,
    )

    # ---------------------------------------------------------
    # 5. Save rules to CSV
    # ---------------------------------------------------------
    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "rank",
                "antecedent",
                "consequent",
                "support",
                "confidence",
                "lift",
            ]
        )

        for index, rule in enumerate(rules, start=1):

            antecedent = " + ".join(
                product_mapping.get(
                    item,
                    item,
                )
                for item in sorted(rule.antecedent)
            )

            consequent = " + ".join(
                product_mapping.get(
                    item,
                    item,
                )
                for item in sorted(rule.consequent)
            )

            writer.writerow(
                [
                    index,
                    antecedent,
                    consequent,
                    rule.support,
                    rule.confidence,
                    rule.lift,
                ]
            )

    print(
        f"\nRules saved to: {OUTPUT_FILE}"
    )

    # ---------------------------------------------------------
    # 6. Display top 20 rules
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOP 20 ASSOCIATION RULES BY LIFT")
    print("=" * 70)

    for index, rule in enumerate(
        rules[:20],
        start=1,
    ):

        antecedent = " + ".join(
            product_mapping.get(
                item,
                item,
            )
            for item in sorted(rule.antecedent)
        )

        consequent = " + ".join(
            product_mapping.get(
                item,
                item,
            )
            for item in sorted(rule.consequent)
        )

        print(f"\n{index}.")
        print(f"   {antecedent}")
        print(f"      → {consequent}")
        print(f"   Support:    {rule.support:.4f}")
        print(f"   Confidence: {rule.confidence:.4f}")
        print(f"   Lift:       {rule.lift:.4f}")


if __name__ == "__main__":
    main()
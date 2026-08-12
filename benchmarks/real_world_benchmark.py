import csv
import time
from pathlib import Path

from src.arm.apriori import apriori
from src.arm.retail_loader import load_online_retail


DATASET = "data/Online Retail.xlsx"
OUTPUT_FILE = Path("benchmarks/real_world_benchmark.csv")

SUPPORT_LEVELS = [
    0.05,
    0.02,
    0.01,
]


def main() -> None:
    print("Loading Online Retail transactions...")

    transactions = load_online_retail(DATASET)

    print(f"Transactions: {len(transactions):,}")
    print()

    results = []

    for min_support in SUPPORT_LEVELS:
        print(
            f"Running Apriori with "
            f"min_support={min_support:.2f}..."
        )

        start = time.perf_counter()

        frequent_itemsets = apriori(
            transactions,
            min_support=min_support,
        )

        elapsed = time.perf_counter() - start

        itemsets_count = len(frequent_itemsets)

        results.append(
            {
                "transactions": len(transactions),
                "min_support": min_support,
                "frequent_itemsets": itemsets_count,
                "runtime_seconds": round(elapsed, 4),
            }
        )

        print(
            f"  Frequent itemsets: {itemsets_count:,}"
        )

        print(
            f"  Runtime: {elapsed:.4f}s"
        )

        print()

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "transactions",
                "min_support",
                "frequent_itemsets",
                "runtime_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"Benchmark results saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.arm.apriori import apriori


def generate_transactions(
    num_transactions: int,
    num_items: int,
    items_per_transaction: int = 10,
):
    """Generate transactions containing frequently co-occurring items."""

    items = [f"Item{i}" for i in range(num_items)]
    core_items = items[:10]

    transactions = []

    for _ in range(num_transactions):
        transaction_items = set(
            random.sample(
                core_items,
                min(5, len(core_items)),
            )
        )

        remaining = [
            item
            for item in items
            if item not in transaction_items
        ]

        extra_count = min(
            items_per_transaction - len(transaction_items),
            len(remaining),
        )

        if extra_count > 0:
            transaction_items.update(
                random.sample(remaining, extra_count)
            )

        transactions.append(frozenset(transaction_items))

    return transactions


def benchmark(
    num_transactions: int,
    num_items: int,
    min_support: float,
):
    """Run Apriori and return benchmark information."""

    transactions = generate_transactions(
        num_transactions,
        num_items,
    )

    start = time.perf_counter()

    frequent_itemsets = apriori(
        transactions,
        min_support,
    )

    elapsed = time.perf_counter() - start

    result = {
        "transactions": num_transactions,
        "items": num_items,
        "min_support": min_support,
        "frequent_itemsets": len(frequent_itemsets),
        "time": elapsed,
    }

    print(
        f"Transactions: {num_transactions:,} | "
        f"Items: {num_items} | "
        f"Min support: {min_support:.2f} | "
        f"Frequent itemsets: {len(frequent_itemsets):,} | "
        f"Time: {elapsed:.4f}s"
    )

    return result


if __name__ == "__main__":
    random.seed(42)

    results = []

    results.append(
        benchmark(
            num_transactions=10_000,
            num_items=20,
            min_support=0.20,
        )
    )

    results.append(
        benchmark(
            num_transactions=10_000,
            num_items=30,
            min_support=0.20,
        )
    )

    results.append(
        benchmark(
            num_transactions=10_000,
            num_items=40,
            min_support=0.20,
        )
    )

    print("\nBenchmark completed.")
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
    """Generate deterministic transactions with recurring item patterns."""

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
    transactions = generate_transactions(
        num_transactions=num_transactions,
        num_items=num_items,
    )

    start = time.perf_counter()

    frequent_itemsets = apriori(
        transactions,
        min_support,
    )

    elapsed = time.perf_counter() - start

    print(
        f"Transactions: {num_transactions:,} | "
        f"Items: {num_items} | "
        f"Min support: {min_support:.2f} | "
        f"Frequent itemsets: {len(frequent_itemsets):,} | "
        f"Time: {elapsed:.4f}s"
    )


if __name__ == "__main__":
    random.seed(42)

    benchmark(
        num_transactions=10_000,
        num_items=40,
        min_support=0.20,
    )

    benchmark(
        num_transactions=25_000,
        num_items=40,
        min_support=0.20,
    )

    benchmark(
        num_transactions=50_000,
        num_items=40,
        min_support=0.20,
    )

    benchmark(
        num_transactions=100_000,
        num_items=40,
        min_support=0.20,
    )
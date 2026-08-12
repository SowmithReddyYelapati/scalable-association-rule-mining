import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_FILE = Path("benchmarks/real_world_benchmark.csv")


def load_results():
    supports = []
    itemsets = []
    runtimes = []

    with RESULTS_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            supports.append(float(row["min_support"]))
            itemsets.append(int(row["frequent_itemsets"]))
            runtimes.append(float(row["runtime_seconds"]))

    return supports, itemsets, runtimes


def create_plots():
    supports, itemsets, runtimes = load_results()

    labels = [
        f"{support * 100:.0f}%"
        for support in supports
    ]

    # Runtime plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        labels,
        runtimes,
        marker="o",
    )

    plt.xlabel("Minimum Support")
    plt.ylabel("Runtime (seconds)")
    plt.title(
        "Apriori Runtime vs Minimum Support\n"
        "UCI Online Retail Dataset"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    runtime_file = Path(
        "benchmarks/real_world_runtime.png"
    )

    plt.savefig(
        runtime_file,
        dpi=300,
    )

    plt.close()

    # Frequent itemsets plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        labels,
        itemsets,
        marker="o",
    )

    plt.xlabel("Minimum Support")
    plt.ylabel("Number of Frequent Itemsets")
    plt.title(
        "Frequent Itemsets vs Minimum Support\n"
        "UCI Online Retail Dataset"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    itemsets_file = Path(
        "benchmarks/real_world_itemsets.png"
    )

    plt.savefig(
        itemsets_file,
        dpi=300,
    )

    plt.close()

    print(f"Runtime plot saved to: {runtime_file}")
    print(f"Itemset plot saved to: {itemsets_file}")


if __name__ == "__main__":
    create_plots()
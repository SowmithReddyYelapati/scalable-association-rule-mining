import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_FILE = Path(__file__).parent / "results.csv"
OUTPUT_FILE = Path(__file__).parent / "scalability.png"


def load_results():
    transactions = []
    times = []

    with RESULTS_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            transactions.append(
                int(row["transactions"])
            )

            times.append(
                float(row["time_seconds"])
            )

    return transactions, times


def create_plot():
    transactions, times = load_results()

    plt.figure(figsize=(8, 5))

    plt.plot(
        transactions,
        times,
        marker="o",
    )

    plt.xlabel("Number of Transactions")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Apriori Scalability")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=150,
    )

    print(
        f"Scalability plot saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    create_plot()
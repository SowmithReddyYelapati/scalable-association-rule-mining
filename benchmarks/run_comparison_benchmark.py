import csv
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.arm.apriori import apriori
from src.arm.fp_growth import fp_growth
from src.arm.retail_loader import load_online_retail

DATASET = "data/online_retail_transactions.csv"
OUTPUT_FILE = Path("benchmarks/comparison_benchmark.csv")
PLOT_FILE = Path("benchmarks/real_world_runtime_comparison.png")
STATIC_PLOT_FILE = Path("webapp/static/real_world_runtime_comparison.png")

SUPPORT_LEVELS = [0.05, 0.02, 0.01]

def main():
    print("Loading Online Retail transactions...")
    transactions = load_online_retail(DATASET)
    print(f"Transactions: {len(transactions):,}\n")

    results = []

    for support in SUPPORT_LEVELS:
        print(f"--- Support Level: {support * 100:.0f}% ({support}) ---")
        
        # Apriori
        print("Running Apriori...")
        start_apriori = time.perf_counter()
        apriori_itemsets = apriori(transactions, support)
        time_apriori = time.perf_counter() - start_apriori
        count_apriori = len(apriori_itemsets)
        print(f"  Apriori: {count_apriori:,} itemsets in {time_apriori:.4f}s")

        # FP-Growth
        print("Running FP-Growth...")
        start_fp = time.perf_counter()
        fp_itemsets = fp_growth(transactions, support)
        time_fp = time.perf_counter() - start_fp
        count_fp = len(fp_itemsets)
        print(f"  FP-Growth: {count_fp:,} itemsets in {time_fp:.4f}s")

        # Validation
        assert count_apriori == count_fp, f"Counts mismatch: {count_apriori} vs {count_fp}"
        
        results.append({
            "min_support": support,
            "frequent_itemsets": count_apriori,
            "apriori_runtime": round(time_apriori, 4),
            "fp_growth_runtime": round(time_fp, 4),
        })
        print()

    # Save to CSV
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["min_support", "frequent_itemsets", "apriori_runtime", "fp_growth_runtime"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {OUTPUT_FILE}")

    # Generate Comparison Plot
    labels = [f"{s * 100:.0f}%" for s in SUPPORT_LEVELS]
    apriori_times = [r["apriori_runtime"] for r in results]
    fp_times = [r["fp_growth_runtime"] for r in results]

    plt.figure(figsize=(9, 5))
    x = range(len(labels))
    width = 0.35

    plt.bar([pos - width/2 for pos in x], apriori_times, width, label="Apriori (Candidate Generation)", color="#1a73e8")
    plt.bar([pos + width/2 for pos in x], fp_times, width, label="FP-Growth (Pattern Tree)", color="#34a853")

    plt.xlabel("Minimum Support Threshold")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Algorithm Runtime Comparison: Apriori vs. FP-Growth\nUCI Online Retail Dataset")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.savefig(STATIC_PLOT_FILE, dpi=300)
    plt.close()
    print(f"Comparison plot saved to {PLOT_FILE} and {STATIC_PLOT_FILE}")

if __name__ == "__main__":
    main()

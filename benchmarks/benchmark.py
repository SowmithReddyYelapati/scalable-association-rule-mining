from __future__ import annotations

import sys
import time
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPLEMENTATIONS
# ============================================================

from src.arm.apriori import apriori
from src.arm.fp_growth import fp_growth
from src.arm.rules import generate_rules
from src.arm.retail_loader import load_online_retail
from src.data.ingestion import process_dataset


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORT_VALUES = [
    0.05,    # 5%
    0.02,    # 2%
    0.01,    # 1%
    0.005,   # 0.5%
    0.001,   # 0.1%
]

CONFIDENCE_VALUES = [
    0.20,    # 20%
    0.40,    # 40%
    0.60,    # 60%
]

DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_dataset() -> Path:
    """
    Find the best dataset available for benchmarking.

    The original Online Retail Excel dataset is preferred
    because it contains InvoiceNo and StockCode, allowing
    transactions to be grouped into real baskets.
    """

    preferred_names = [
        "Online Retail.xlsx",
        "online_retail.xlsx",
        "Groceries_dataset.csv",
        "groceries_dataset.csv",
        "groceries.csv",
        "online_retail_transactions.csv",
    ]

    for name in preferred_names:

        path = DATA_DIR / name

        if path.exists():
            return path

    raise FileNotFoundError(
        f"No suitable dataset found in {DATA_DIR}"
    )


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

def load_transactions() -> list:
    """
    Load transactions from the appropriate project loader.

    Online Retail:
        load_online_retail() -> list[Itemset]

    Other CSV datasets:
        process_dataset() -> dictionary containing transactions.
    """

    dataset = find_dataset()

    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(dataset)

    # --------------------------------------------------------
    # ONLINE RETAIL
    # --------------------------------------------------------

    if dataset.suffix.lower() == ".xlsx":

        print()
        print("Loading using Online Retail transaction loader...")

        transactions = load_online_retail(
            dataset
        )

    # --------------------------------------------------------
    # FALLBACK CSV DATASET
    # --------------------------------------------------------

    else:

        print()
        print("Loading using generic dataset ingestion...")

        result = process_dataset(
            dataset,
            filename=dataset.name,
        )

        if not isinstance(result, dict):

            raise RuntimeError(
                "process_dataset() returned an unexpected result."
            )

        print(
            "Ingestion status:",
            result.get("status"),
        )

        print(
            "Available keys:",
            list(result.keys()),
        )

        transactions = result.get(
            "transactions"
        )

        if transactions is None:

            raise RuntimeError(
                "Could not find transactions "
                "in process_dataset() result."
            )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    transactions = list(
        transactions
    )

    if not transactions:

        raise RuntimeError(
            "Dataset contains no transactions."
        )

    # --------------------------------------------------------
    # Basket statistics
    # --------------------------------------------------------

    basket_sizes = [
        len(transaction)
        for transaction in transactions
    ]

    unique_products = set()

    for transaction in transactions:

        unique_products.update(
            transaction
        )

    average_basket_size = (
        sum(basket_sizes)
        / len(basket_sizes)
    )

    largest_basket = max(
        basket_sizes
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BENCHMARK DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Dataset             : {dataset.name}"
    )

    print(
        f"Baskets             : "
        f"{len(transactions):,}"
    )

    print(
        f"Unique Products     : "
        f"{len(unique_products):,}"
    )

    print(
        f"Avg. Items / Basket : "
        f"{average_basket_size:.2f}"
    )

    print(
        f"Largest Basket      : "
        f"{largest_basket} items"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Important validation
    # --------------------------------------------------------

    if largest_basket <= 1:

        print()
        print(
            "WARNING:"
        )

        print(
            "Every transaction contains at most "
            "one product."
        )

        print(
            "Association-rule mining cannot discover "
            "meaningful multi-item patterns."
        )

        print()

    else:

        print()
        print(
            "✓ Multi-item baskets detected."
        )

    return transactions


# ============================================================
# BENCHMARK ONE ALGORITHM
# ============================================================

def benchmark_algorithm(
    algorithm_name: str,
    transactions: list,
    min_support: float,
    min_confidence: float,
) -> dict:
    """
    Execute one mining algorithm and measure runtime.
    """

    # --------------------------------------------------------
    # APRIORI
    # --------------------------------------------------------

    if algorithm_name == "Apriori":

        start = time.perf_counter()

        frequent_itemsets = apriori(
            transactions,
            min_support,
        )

        runtime = (
            time.perf_counter()
            - start
        )

    # --------------------------------------------------------
    # FP-GROWTH
    # --------------------------------------------------------

    elif algorithm_name == "FP-Growth":

        start = time.perf_counter()

        frequent_itemsets = fp_growth(
            transactions,
            min_support,
        )

        runtime = (
            time.perf_counter()
            - start
        )

    else:

        raise ValueError(
            f"Unknown algorithm: {algorithm_name}"
        )

    # --------------------------------------------------------
    # Association rules
    # --------------------------------------------------------

    rules = generate_rules(
        frequent_itemsets,
        min_confidence,
    )

    return {
        "algorithm": algorithm_name,

        "support": min_support,

        "confidence": min_confidence,

        "runtime": runtime,

        "frequent_itemsets": (
            len(frequent_itemsets)
        ),

        "association_rules": (
            len(rules)
        ),
    }


# ============================================================
# COMPARE RESULTS
# ============================================================

def compare_results(
    apriori_result: dict,
    fp_result: dict,
) -> dict:
    """
    Compare Apriori and FP-Growth benchmark results.
    """

    apriori_runtime = (
        apriori_result["runtime"]
    )

    fp_runtime = (
        fp_result["runtime"]
    )

    # --------------------------------------------------------
    # Speedup
    # --------------------------------------------------------

    if fp_runtime > 0:

        speedup = (
            apriori_runtime
            / fp_runtime
        )

    else:

        speedup = None

    # --------------------------------------------------------
    # Result equivalence
    # --------------------------------------------------------

    same_itemset_count = (
        apriori_result[
            "frequent_itemsets"
        ]
        ==
        fp_result[
            "frequent_itemsets"
        ]
    )

    same_rule_count = (
        apriori_result[
            "association_rules"
        ]
        ==
        fp_result[
            "association_rules"
        ]
    )

    return {
        "support": (
            apriori_result["support"]
        ),

        "confidence": (
            apriori_result["confidence"]
        ),

        "apriori_runtime": (
            apriori_runtime
        ),

        "fp_growth_runtime": (
            fp_runtime
        ),

        "speedup": speedup,

        "same_itemset_count": (
            same_itemset_count
        ),

        "same_rule_count": (
            same_rule_count
        ),
    }


# ============================================================
# RUN BENCHMARK
# ============================================================

def run_benchmark() -> list[dict]:
    """
    Run Apriori and FP-Growth for every configured
    support/confidence combination.
    """

    transactions = load_transactions()

    results = []

    print()
    print("=" * 70)
    print("APRIORI VS FP-GROWTH BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # Support values
    # --------------------------------------------------------

    for support in SUPPORT_VALUES:

        # ----------------------------------------------------
        # Confidence values
        # ----------------------------------------------------

        for confidence in CONFIDENCE_VALUES:

            print()
            print("-" * 70)

            print(
                f"Support: "
                f"{support * 100:.2f}% | "
                f"Confidence: "
                f"{confidence * 100:.2f}%"
            )

            print("-" * 70)

            # ------------------------------------------------
            # Apriori
            # ------------------------------------------------

            apriori_result = benchmark_algorithm(
                algorithm_name="Apriori",
                transactions=transactions,
                min_support=support,
                min_confidence=confidence,
            )

            # ------------------------------------------------
            # FP-Growth
            # ------------------------------------------------

            fp_result = benchmark_algorithm(
                algorithm_name="FP-Growth",
                transactions=transactions,
                min_support=support,
                min_confidence=confidence,
            )

            # ------------------------------------------------
            # Compare
            # ------------------------------------------------

            comparison = compare_results(
                apriori_result,
                fp_result,
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            results.append(
                {
                    **apriori_result,
                    "comparison": comparison,
                }
            )

            results.append(
                {
                    **fp_result,
                    "comparison": comparison,
                }
            )

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print(
                f"Apriori   : "
                f"{apriori_result['runtime']:.4f}s | "
                f"{apriori_result['frequent_itemsets']} "
                f"itemsets | "
                f"{apriori_result['association_rules']} "
                f"rules"
            )

            print(
                f"FP-Growth : "
                f"{fp_result['runtime']:.4f}s | "
                f"{fp_result['frequent_itemsets']} "
                f"itemsets | "
                f"{fp_result['association_rules']} "
                f"rules"
            )

            if comparison["speedup"] is not None:

                print(
                    f"Speedup   : "
                    f"{comparison['speedup']:.2f}x"
                )

            else:

                print(
                    "Speedup   : N/A"
                )

            print(
                "Equivalent:",
                comparison[
                    "same_itemset_count"
                ],
                "itemsets /",
                comparison[
                    "same_rule_count"
                ],
                "rules",
            )

    return results


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(
    results: list[dict],
) -> None:
    """
    Print final benchmark table.
    """

    print()
    print()

    print("=" * 90)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 90)

    print(
        f"{'Support':<10}"
        f"{'Confidence':<12}"
        f"{'Apriori':<14}"
        f"{'FP-Growth':<14}"
        f"{'Speedup':<10}"
        f"{'Items':<10}"
        f"{'Rules':<10}"
    )

    print("-" * 90)

    # --------------------------------------------------------
    # Iterate over every configuration
    # --------------------------------------------------------

    for support in SUPPORT_VALUES:

        for confidence in CONFIDENCE_VALUES:

            # ------------------------------------------------
            # Apriori result
            # ------------------------------------------------

            apriori_result = next(
                result
                for result in results
                if (
                    result["algorithm"]
                    == "Apriori"
                    and
                    result["support"]
                    == support
                    and
                    result["confidence"]
                    == confidence
                )
            )

            # ------------------------------------------------
            # FP-Growth result
            # ------------------------------------------------

            fp_result = next(
                result
                for result in results
                if (
                    result["algorithm"]
                    == "FP-Growth"
                    and
                    result["support"]
                    == support
                    and
                    result["confidence"]
                    == confidence
                )
            )

            comparison = (
                apriori_result[
                    "comparison"
                ]
            )

            speedup = comparison[
                "speedup"
            ]

            if speedup is not None:

                speedup_text = (
                    f"{speedup:.2f}x"
                )

            else:

                speedup_text = "-"

            print(
                f"{support * 100:>7.2f}%  "
                f"{confidence * 100:>7.0f}%     "
                f"{apriori_result['runtime']:>8.4f}s     "
                f"{fp_result['runtime']:>8.4f}s     "
                f"{speedup_text:>7}   "
                f"{apriori_result['frequent_itemsets']:>6}     "
                f"{apriori_result['association_rules']:>5}"
            )

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        benchmark_results = run_benchmark()

        print_summary(
            benchmark_results
        )

        print()

        print(
            "Benchmark completed successfully."
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print("BENCHMARK FAILED")
        print("=" * 70)

        print(
            type(exc).__name__,
            ":",
            exc,
        )

        raise
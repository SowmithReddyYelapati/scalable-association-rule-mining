import pytest
from src.arm.apriori import apriori
from src.arm.fp_growth import fp_growth


@pytest.fixture
def deterministic_transactions_1():
    """Standard transactional benchmark dataset."""
    return [
        {"milk", "bread", "butter"},
        {"bread", "butter"},
        {"milk", "bread"},
        {"milk", "bread", "butter", "eggs"},
        {"bread", "eggs"},
        {"milk", "eggs"},
        {"milk", "bread", "butter", "cheese"},
        {"bread", "butter", "cheese"},
        {"milk", "bread", "cheese"},
        {"milk", "butter", "cheese"},
    ]


@pytest.fixture
def deterministic_transactions_2():
    """Dense multi-item transactional dataset."""
    return [
        {"item1", "item2", "item3", "item4", "item5"},
        {"item2", "item3", "item4"},
        {"item1", "item3", "item5"},
        {"item1", "item2", "item4", "item5"},
        {"item2", "item3", "item5"},
        {"item1", "item2", "item3", "item4"},
        {"item1", "item4", "item5"},
        {"item3", "item4", "item5"},
        {"item1", "item2", "item3", "item5"},
        {"item1", "item2", "item3", "item4", "item5"},
        {"item2", "item4"},
        {"item1", "item3"},
    ]


@pytest.fixture
def deterministic_sparse_transactions():
    """Sparse transactions with isolated and overlapping items."""
    return [
        {"A", "B"},
        {"B", "C", "D"},
        {"A", "C", "E"},
        {"B", "D", "E"},
        {"A", "B", "C", "E"},
        {"F", "G"},
        {"A", "B", "D"},
        {"C", "D", "E"},
        {"A", "B", "C", "D", "E"},
        {"B", "E"},
    ]


@pytest.mark.parametrize("support_threshold", [0.10, 0.20, 0.30, 0.40, 0.50])
def test_apriori_fpgrowth_equivalence_dataset1(deterministic_transactions_1, support_threshold):
    """
    Verify Apriori and FP-Growth produce identical frequent itemsets and supports
    on dataset 1 across multiple support thresholds.
    """
    apriori_itemsets = apriori(deterministic_transactions_1, min_support=support_threshold)
    fpgrowth_itemsets = fp_growth(deterministic_transactions_1, min_support=support_threshold)

    # 1. Compare set of frequent itemsets
    assert set(apriori_itemsets.keys()) == set(fpgrowth_itemsets.keys()), (
        f"Itemset mismatch at support={support_threshold}:\n"
        f"Apriori: {set(apriori_itemsets.keys())}\n"
        f"FP-Growth: {set(fpgrowth_itemsets.keys())}"
    )

    # 2. Compare support values for every itemset
    for itemset in apriori_itemsets:
        assert fpgrowth_itemsets[itemset] == pytest.approx(apriori_itemsets[itemset], abs=1e-7)


@pytest.mark.parametrize("support_threshold", [0.15, 0.25, 0.35, 0.50])
def test_apriori_fpgrowth_equivalence_dataset2(deterministic_transactions_2, support_threshold):
    """
    Verify Apriori and FP-Growth produce identical frequent itemsets and supports
    on dense dataset 2 across multiple support thresholds.
    """
    apriori_itemsets = apriori(deterministic_transactions_2, min_support=support_threshold)
    fpgrowth_itemsets = fp_growth(deterministic_transactions_2, min_support=support_threshold)

    # 1. Compare set of frequent itemsets
    assert set(apriori_itemsets.keys()) == set(fpgrowth_itemsets.keys())

    # 2. Compare support values
    for itemset in apriori_itemsets:
        assert fpgrowth_itemsets[itemset] == pytest.approx(apriori_itemsets[itemset], abs=1e-7)


@pytest.mark.parametrize("support_threshold", [0.10, 0.20, 0.30])
def test_apriori_fpgrowth_equivalence_sparse_dataset(deterministic_sparse_transactions, support_threshold):
    """
    Verify Apriori and FP-Growth produce identical frequent itemsets and supports
    on sparse dataset across multiple support thresholds.
    """
    apriori_itemsets = apriori(deterministic_sparse_transactions, min_support=support_threshold)
    fpgrowth_itemsets = fp_growth(deterministic_sparse_transactions, min_support=support_threshold)

    assert set(apriori_itemsets.keys()) == set(fpgrowth_itemsets.keys())

    for itemset in apriori_itemsets:
        assert fpgrowth_itemsets[itemset] == pytest.approx(apriori_itemsets[itemset], abs=1e-7)

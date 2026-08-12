from src.arm.pruning import prune_candidates
from src.arm.transaction import create_itemset


def test_keep_candidate_when_all_subsets_are_frequent():
    candidates = [
        create_itemset(["A", "B", "C"]),
    ]

    frequent_itemsets = [
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
        create_itemset(["B", "C"]),
    ]

    assert prune_candidates(candidates, frequent_itemsets) == {
        create_itemset(["A", "B", "C"]),
    }


def test_remove_candidate_when_subset_is_not_frequent():
    candidates = [
        create_itemset(["A", "B", "C"]),
    ]

    frequent_itemsets = [
        create_itemset(["A", "B"]),
        create_itemset(["A", "C"]),
    ]

    assert prune_candidates(candidates, frequent_itemsets) == set()


def test_multiple_candidates():
    candidates = [
        create_itemset(["A", "B", "C"]),
        create_itemset(["A", "B", "D"]),
    ]

    frequent_itemsets = [
    create_itemset(["A", "B"]),
    create_itemset(["A", "C"]),
    create_itemset(["B", "C"]),
    create_itemset(["A", "D"]),
    ]

    assert prune_candidates(candidates, frequent_itemsets) == {
        create_itemset(["A", "B", "C"]),
    }


def test_empty_candidates():
    assert prune_candidates([], []) == set()
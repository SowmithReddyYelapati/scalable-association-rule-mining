from src.arm.apriori import apriori
from src.arm.fp_growth import fp_growth


def test_fp_growth_finds_single_items():

    transactions = [
        {"A", "B", "C"},
        {"A", "B"},
        {"A", "C"},
        {"B", "C"},
    ]

    result = fp_growth(
        transactions,
        min_support=0.5,
    )

    assert frozenset({"A"}) in result
    assert frozenset({"B"}) in result
    assert frozenset({"C"}) in result


def test_fp_growth_finds_pairs():

    transactions = [
        {"A", "B", "C"},
        {"A", "B"},
        {"A", "C"},
        {"B", "C"},
    ]

    result = fp_growth(
        transactions,
        min_support=0.5,
    )

    assert frozenset({"A", "B"}) in result
    assert frozenset({"A", "C"}) in result
    assert frozenset({"B", "C"}) in result


def test_fp_growth_support():

    transactions = [
        {"A", "B"},
        {"A", "B"},
        {"A"},
        {"B"},
    ]

    result = fp_growth(
        transactions,
        min_support=0.5,
    )

    assert result[
        frozenset({"A"})
    ] == 0.75

    assert result[
        frozenset({"B"})
    ] == 0.75

    assert result[
        frozenset({"A", "B"})
    ] == 0.50


def test_empty_transactions():

    result = fp_growth(
        [],
        min_support=0.5,
    )

    assert result == {}


def test_high_support_returns_empty_when_nothing_qualifies():

    transactions = [
        {"A"},
        {"B"},
        {"C"},
    ]

    result = fp_growth(
        transactions,
        min_support=1.0,
    )

    assert len(result) == 0


def test_invalid_support():

    transactions = [
        {"A", "B"},
    ]

    try:

        fp_growth(
            transactions,
            min_support=0,
        )

        assert False

    except ValueError:

        assert True


def test_fp_growth_matches_apriori():

    transactions = [
        {"A", "B", "C"},
        {"A", "B"},
        {"A", "C"},
        {"B", "C"},
        {"A", "B", "C"},
    ]

    min_support = 0.4

    apriori_result = apriori(
        transactions,
        min_support,
    )

    fp_result = fp_growth(
        transactions,
        min_support,
    )

    assert set(
        apriori_result.keys()
    ) == set(
        fp_result.keys()
    )

    for itemset in apriori_result:

        assert abs(
            apriori_result[itemset]
            - fp_result[itemset]
        ) < 1e-9
from pathlib import Path

from src.arm.retail_loader import load_online_retail


DATASET = Path("data/Online Retail.xlsx")


def test_online_retail_dataset_exists():
    assert DATASET.exists()


def test_online_retail_loads_transactions():
    transactions = load_online_retail(DATASET)

    assert len(transactions) == 20_136
    assert len(transactions) > 0


def test_transactions_are_frozensets():
    transactions = load_online_retail(DATASET)

    assert all(isinstance(transaction, frozenset) for transaction in transactions)


def test_transactions_contain_products():
    transactions = load_online_retail(DATASET)

    assert all(len(transaction) > 0 for transaction in transactions)


def test_expected_product_count():
    transactions = load_online_retail(DATASET)

    products = set().union(*transactions)

    assert len(products) == 3_925
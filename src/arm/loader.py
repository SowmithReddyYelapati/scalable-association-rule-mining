import csv
from collections import defaultdict
from pathlib import Path

from src.arm.transaction import Itemset


def load_transactions(path: str | Path) -> list[Itemset]:
    """Load transactions from a CSV containing transaction_id and item."""

    transactions: dict[str, set[str]] = defaultdict(set)

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV file must contain a header")

        required_columns = {"transaction_id", "item"}

        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain transaction_id and item columns"
            )

        for row in reader:
            transaction_id = row["transaction_id"].strip()
            item = row["item"].strip()

            if transaction_id and item:
                transactions[transaction_id].add(item)

    return [frozenset(items) for items in transactions.values()]
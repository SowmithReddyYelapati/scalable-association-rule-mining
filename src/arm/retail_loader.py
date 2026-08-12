from pathlib import Path

import pandas as pd

from src.arm.transaction import Itemset


def load_online_retail(
    file_path: str | Path,
) -> list[Itemset]:
    """Load the UCI Online Retail dataset as transaction itemsets."""

    file_path = Path(file_path)

    cache_path = file_path.with_name(
        "online_retail_transactions.csv"
    )

    if cache_path.exists():
        return _load_cached_transactions(cache_path)

    df = pd.read_excel(file_path)

    # Remove cancelled invoices.
    df = df[
        ~df["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    ]

    # Keep only actual purchases.
    df = df[df["Quantity"] > 0]

    # Remove rows without product descriptions.
    df = df[df["Description"].notna()]

    grouped = (
        df.groupby("InvoiceNo")["StockCode"]
        .apply(
            lambda items: "|".join(
                sorted(
                    set(items.astype(str))
                )
            )
        )
        .reset_index()
    )

    grouped.to_csv(
        cache_path,
        index=False,
    )

    return [
        frozenset(row.split("|"))
        for row in grouped["StockCode"]
        if row
    ]


def _load_cached_transactions(
    cache_path: Path,
) -> list[Itemset]:
    """Load preprocessed transactions from the cache."""

    df = pd.read_csv(cache_path)

    return [
        frozenset(row.split("|"))
        for row in df["StockCode"].dropna()
        if row
    ]
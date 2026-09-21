"""
Transaction builder module.

Converts a preprocessed DataFrame and schema definition into
standardized transaction itemsets, product mapping dictionary,
and detailed dataset statistics.
"""

from typing import Any
import pandas as pd

from src.arm.transaction import Itemset
from src.data.schema_detector import SchemaDetectionResult


def build_transactions_from_df(
    df: pd.DataFrame,
    schema: SchemaDetectionResult,
) -> tuple[list[Itemset], dict[str, str], dict[str, Any]]:
    """
    Build standardized transaction itemsets from a DataFrame.

    Returns:
        (transactions, product_names, statistics_metadata)
    """
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if not schema.transaction_col:
        raise ValueError("No transaction identifier column could be determined.")

    if not schema.product_col:
        raise ValueError("No product identifier column could be determined.")

    rows_raw = len(df)
    data = df.copy()

    # 1. Handle Transaction Column
    if isinstance(schema.transaction_col, tuple):
        col1, col2 = schema.transaction_col
        data = data.dropna(subset=[col1, col2])
        data["_txn_id"] = (
            data[col1].astype(str).str.strip()
            + "_"
            + data[col2].astype(str).str.strip()
        )
        transaction_col_name = "_txn_id"
    else:
        transaction_col_name = schema.transaction_col
        data = data.dropna(subset=[transaction_col_name])
        data["_txn_id"] = data[transaction_col_name].astype(str).str.strip()
        transaction_col_name = "_txn_id"

    # Remove empty transaction IDs
    data = data[data["_txn_id"] != ""]
    data = data[~data["_txn_id"].str.upper().str.startswith("C")]  # Online retail cancellation invoices

    # 2. Handle Quantity Column
    if schema.quantity_col and schema.quantity_col in data.columns:
        data[schema.quantity_col] = pd.to_numeric(
            data[schema.quantity_col],
            errors="coerce"
        )
        data = data[data[schema.quantity_col] > 0]

    # 3. Handle Product Column
    product_col_name = schema.product_col
    data = data.dropna(subset=[product_col_name])
    data["_product"] = data[product_col_name].astype(str).str.strip()

    # Remove empty product strings
    data = data[data["_product"] != ""]

    rows_valid = len(data)
    rows_removed = rows_raw - rows_valid

    if data.empty:
        raise ValueError("No valid transaction rows remained after dataset cleaning.")

    # 4. Build Product Name Mapping (Code -> Description if applicable)
    product_names: dict[str, str] = {}
    if schema.product_code_col and schema.product_code_col in data.columns:
        mapping_df = data[[schema.product_code_col, schema.product_col]].dropna()
        for _, row in mapping_df.iterrows():
            code = str(row[schema.product_code_col]).strip()
            name = str(row[schema.product_col]).strip()
            if code and name:
                product_names.setdefault(code, name)

    # 5. Build Transaction Itemsets
    transactions: list[Itemset] = []
    unique_products: set[str] = set()

    for _, group in data.groupby("_txn_id", sort=False):
        items = frozenset(group["_product"].tolist())
        if items:
            transactions.append(items)
            unique_products.update(items)

    transaction_count = len(transactions)

    if not transactions:
        raise ValueError("No valid transaction baskets could be constructed.")

    average_items = (
        sum(len(t) for t in transactions) / transaction_count
        if transaction_count
        else 0.0
    )

    largest_transaction = max(len(t) for t in transactions) if transactions else 0

    statistics = {
        "rows_raw": rows_raw,
        "rows_valid": rows_valid,
        "rows_removed": rows_removed,
        "transactions": transaction_count,
        "products": len(unique_products),
        "average_items": round(average_items, 2),
        "largest_transaction": largest_transaction,
        "columns": len(df.columns),
        "transaction_col_display": (
            f"{schema.transaction_col[0]} + {schema.transaction_col[1]}"
            if isinstance(schema.transaction_col, tuple)
            else schema.transaction_col
        ),
        "product_col_display": schema.product_col,
        "quantity_col_display": schema.quantity_col or "None",
        "inferred": schema.inferred,
        "inferred_reason": schema.inferred_reason,
    }

    return transactions, product_names, statistics

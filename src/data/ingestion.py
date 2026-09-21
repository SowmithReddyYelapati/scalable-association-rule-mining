"""
High-level dataset ingestion pipeline.

Handles file loading with encoding fallbacks, schema detection,
manual column mapping overrides, dataset cleaning, transaction building,
and structured logging.
"""

from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
import pandas as pd

from src.arm.transaction import Itemset
from src.data.schema_detector import SchemaDetectionResult, detect_schema
from src.data.transaction_builder import build_transactions_from_df


def process_dataset(
    file_source: str | Path | BytesIO | StringIO,
    filename: str = "dataset.csv",
    manual_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Process an uploaded tabular dataset file into standard transaction itemsets.

    Args:
        file_source: Path or binary/text file stream.
        filename: Original file name.
        manual_mapping: Optional dict with keys 'transaction', 'product', 'quantity'.

    Returns:
        Dictionary with status ('success' or 'needs_mapping'), transactions, metadata.
    """
    ext = Path(filename).suffix.lower()

    # 1. Read file into pandas DataFrame
    df = _read_dataframe(file_source, ext)

    if df is None or df.empty:
        raise ValueError("The uploaded dataset file is empty or could not be parsed.")

    # Deduplicate column names if needed
    df.columns = [str(col).strip() for col in df.columns]

    # 2. Detect Schema or apply Manual Mapping
    if manual_mapping and manual_mapping.get("transaction") and manual_mapping.get("product"):
        txn_val = manual_mapping["transaction"]
        prod_val = manual_mapping["product"]
        qty_val = manual_mapping.get("quantity") if manual_mapping.get("quantity") != "None" else None

        schema = SchemaDetectionResult(
            transaction_col=txn_val,
            product_col=prod_val,
            quantity_col=qty_val,
            inferred=False,
            confidence=1.0,
            all_columns=list(df.columns)
        )
    else:
        schema = detect_schema(df)

    # If detection is ambiguous (no transaction or product column) and no manual mapping
    if (not schema.transaction_col or not schema.product_col) and schema.confidence < 0.5:
        return {
            "status": "needs_mapping",
            "columns": schema.all_columns,
            "filename": filename,
            "schema_suggestion": schema.detected_mapping,
            "message": "Unable to identify a transactional structure. Please ensure your CSV contains transaction/basket IDs and product names."
        }

    # 3. Build Transactions & Statistics
    transactions, product_names, statistics = build_transactions_from_df(df, schema)

    # 4. Console Logging
    _log_ingestion_summary(filename, statistics)

    return {
        "status": "success",
        "transactions": transactions,
        "product_names": product_names,
        "metrics": statistics,
        "filename": filename,
    }


def _read_dataframe(file_source: Any, ext: str) -> pd.DataFrame:
    """Read CSV or Excel dataset with robust encoding fallbacks."""
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_source)

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            if isinstance(file_source, (str, Path)):
                return pd.read_csv(file_source, encoding=enc)

            if isinstance(file_source, BytesIO):
                file_source.seek(0)
                return pd.read_csv(file_source, encoding=enc)

            if isinstance(file_source, StringIO):
                file_source.seek(0)
                return pd.read_csv(file_source)

        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    # Final attempt default read
    if isinstance(file_source, BytesIO):
        file_source.seek(0)
    return pd.read_csv(file_source)


def _log_ingestion_summary(filename: str, stats: dict[str, Any]) -> None:
    """Print clean, structured ingestion log block to the server terminal."""
    print()
    print("=" * 60)
    print("DATASET INGESTION SUMMARY")
    print("=" * 60)
    print(f"File                : {filename}")
    print(f"Uploaded Rows       : {stats['rows_raw']:,}")
    print(f"Detected Transaction: {stats['transaction_col_display']}")
    print(f"Detected Product    : {stats['product_col_display']}")
    print(f"Detected Quantity   : {stats['quantity_col_display']}")
    if stats.get("inferred_reason"):
        print(f"Inference Note      : {stats['inferred_reason']}")
    print(f"Rows Removed        : {stats['rows_removed']:,}")
    print(f"Valid Rows          : {stats['rows_valid']:,}")
    print(f"Baskets (Txns)      : {stats['transactions']:,}")
    print(f"Unique Products     : {stats['products']:,}")
    print(f"Avg. Items / Basket : {stats['average_items']}")
    print(f"Largest Basket      : {stats['largest_transaction']} items")
    print("=" * 60)
    print()

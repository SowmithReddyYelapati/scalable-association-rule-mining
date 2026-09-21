"""
Schema detection module for transactional CSV datasets.

Provides normalized column matching, semantic column inference,
and structural heuristics to identify transaction identifiers,
product columns, and quantity columns across diverse schemas.
"""

from dataclasses import dataclass, field
import re
import pandas as pd


@dataclass
class SchemaDetectionResult:
    """Result of automated schema detection on a dataset."""
    transaction_col: str | tuple[str, str] | None = None
    product_col: str | None = None
    quantity_col: str | None = None
    member_col: str | None = None
    date_col: str | None = None
    product_code_col: str | None = None
    inferred: bool = False
    inferred_reason: str | None = None
    confidence: float = 0.0
    detected_mapping: dict[str, str] = field(default_factory=dict)
    all_columns: list[str] = field(default_factory=list)


# Candidate normalized column names
TRANSACTION_CANDIDATES = [
    "transactionid", "transaction_id", "transaction", "invoiceno", "invoice_no",
    "invoicenum", "invoicenumber", "invoice", "orderid", "order_id", "orderno",
    "order_no", "ordernumber", "order", "basketid", "basket_id", "basket",
    "cartid", "cart_id", "cart", "purchaseid", "purchase_id", "receiptid", "receipt_no"
]

PRODUCT_CANDIDATES = [
    "productname", "product_name", "product", "itemname", "item_name", "item",
    "itemdescription", "item_description", "description", "productdescription",
    "stockcode", "stock_code", "sku", "skucode", "productid", "product_id", "itemid", "item_id"
]

QUANTITY_CANDIDATES = [
    "quantity", "qty", "count", "units", "amount", "unitcount", "itemcount"
]

MEMBER_CANDIDATES = [
    "membernumber", "member_number", "memberid", "member_id", "member",
    "customerid", "customer_id", "customerno", "customer_no", "customer",
    "userid", "user_id", "clientid", "client_id"
]

DATE_CANDIDATES = [
    "date", "orderdate", "order_date", "invoicedate", "invoice_date",
    "transactiondate", "transaction_date", "purchasedate", "purchase_date", "time", "timestamp"
]


def normalize_column_name(column: str) -> str:
    """
    Normalize column names for matching.

    Removes BOMs, spaces, underscores, hyphens, dots, punctuation,
    and converts to lowercase.
    """
    if not isinstance(column, str):
        column = str(column)

    # Strip BOM marker
    column = column.replace("\ufeff", "")

    # Lowercase and strip punctuation/spaces/underscores/hyphens
    normalized = re.sub(r"[\s_\-\.\:\;,\(\)\[\]]+", "", column).lower().strip()

    return normalized


def detect_schema(df: pd.DataFrame) -> SchemaDetectionResult:
    """
    Detect transaction, product, and quantity columns from a DataFrame.

    Combines normalized column name matching with structural heuristics.
    """
    if df is None or df.empty:
        return SchemaDetectionResult(confidence=0.0)

    raw_columns = list(df.columns)
    norm_map = {col: normalize_column_name(col) for col in raw_columns}

    transaction_col = None
    product_col = None
    product_code_col = None
    quantity_col = None
    member_col = None
    date_col = None

    # Helper matcher
    def find_best_match(candidates: list[str]) -> str | None:
        for candidate in candidates:
            for raw_col, norm_col in norm_map.items():
                if norm_col == candidate:
                    return raw_col
        # Partial match
        for candidate in candidates:
            for raw_col, norm_col in norm_map.items():
                if candidate in norm_col or norm_col in candidate:
                    return raw_col
        return None

    # 1. Detect Transaction Column
    transaction_col = find_best_match(TRANSACTION_CANDIDATES)

    # 2. Detect Product Column
    # Check if both StockCode and Description exist
    desc_col = None
    code_col = None

    for raw_col, norm_col in norm_map.items():
        if norm_col in ["description", "itemdescription", "productdescription", "productname", "itemname"]:
            desc_col = raw_col
        elif norm_col in ["stockcode", "stock_code", "sku", "skucode", "productid", "itemid"]:
            code_col = raw_col

    if desc_col:
        product_col = desc_col
        if code_col and code_col != desc_col:
            product_code_col = code_col
    else:
        product_col = find_best_match(PRODUCT_CANDIDATES)

    # 3. Detect Quantity Column
    quantity_col = find_best_match(QUANTITY_CANDIDATES)

    # 4. Detect Member and Date columns
    member_col = find_best_match(MEMBER_CANDIDATES)
    date_col = find_best_match(DATE_CANDIDATES)

    # Validate Transaction Column vs Member Column
    # If transaction_col was matched to customer/member, check if a better transaction ID exists
    if transaction_col and member_col and transaction_col == member_col:
        # Check if there is an actual order/invoice column
        invoice_cand = find_best_match(["invoiceno", "invoice", "orderid", "transactionid"])
        if invoice_cand and invoice_cand != member_col:
            transaction_col = invoice_cand

    # Structural Validation of Transaction Column
    inferred = False
    inferred_reason = None
    confidence = 0.8 if (transaction_col and product_col) else 0.4

    # 5. Member + Date Fallback
    # If no valid transaction_col, but member_col and date_col exist
    if not transaction_col and member_col and date_col and product_col:
        transaction_col = (member_col, date_col)
        inferred = True
        inferred_reason = f"Transaction identifier inferred from {member_col} + {date_col}."
        confidence = 0.85

    # 6. Structural Heuristics Fallback
    # If still no transaction_col, look for a column with repetition (mean rows per unique > 1.05)
    if not transaction_col and product_col and len(df) > 1:
        best_repeat_col = None
        best_repeat_ratio = 1.0

        for col in raw_columns:
            if col == product_col or col == quantity_col:
                continue
            unique_count = df[col].nunique()
            if 1 < unique_count < len(df):
                ratio = len(df) / unique_count
                if ratio > best_repeat_ratio:
                    best_repeat_ratio = ratio
                    best_repeat_col = col

        if best_repeat_col and best_repeat_ratio >= 1.05:
            transaction_col = best_repeat_col
            inferred = True
            inferred_reason = f"Transaction column inferred from structural repetition in {best_repeat_col}."
            confidence = 0.7

    # Build mapping dict
    mapping = {}
    if transaction_col:
        if isinstance(transaction_col, tuple):
            mapping["transaction"] = f"{transaction_col[0]} + {transaction_col[1]}"
        else:
            mapping["transaction"] = transaction_col
    if product_col:
        mapping["product"] = product_col
    if quantity_col:
        mapping["quantity"] = quantity_col

    return SchemaDetectionResult(
        transaction_col=transaction_col,
        product_col=product_col,
        quantity_col=quantity_col,
        member_col=member_col,
        date_col=date_col,
        product_code_col=product_code_col,
        inferred=inferred,
        inferred_reason=inferred_reason,
        confidence=confidence if (transaction_col and product_col) else 0.2,
        detected_mapping=mapping,
        all_columns=raw_columns
    )

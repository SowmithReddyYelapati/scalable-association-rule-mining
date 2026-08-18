"""
Unit test suite for the robust dataset ingestion and schema detection layer.
"""

from io import StringIO
import pytest
import pandas as pd

from src.data.schema_detector import normalize_column_name, detect_schema
from src.data.transaction_builder import build_transactions_from_df
from src.data.ingestion import process_dataset


# ============================================================
# COLUMN NORMALIZATION TESTS
# ============================================================

def test_column_normalization():
    assert normalize_column_name("TransactionID") == "transactionid"
    assert normalize_column_name("transaction_id") == "transactionid"
    assert normalize_column_name("Transaction ID") == "transactionid"
    assert normalize_column_name("transaction-id") == "transactionid"
    assert normalize_column_name("TRANSACTION ID") == "transactionid"
    assert normalize_column_name("\ufeffTransactionID") == "transactionid"
    assert normalize_column_name("Invoice.No") == "invoiceno"
    assert normalize_column_name("Product_Name (Desc)") == "productnamedesc"


# ============================================================
# SCHEMA DETECTION TESTS
# ============================================================

def test_transaction_column_detection():
    # TEST 1: Standard TransactionID
    df1 = pd.DataFrame({"TransactionID": [1, 1, 2], "ProductID": ["A", "B", "A"]})
    schema1 = detect_schema(df1)
    assert schema1.transaction_col == "TransactionID"
    assert schema1.product_col == "ProductID"


def test_invoice_column_detection():
    # TEST 2: UCI Online Retail schema
    df2 = pd.DataFrame({
        "InvoiceNo": ["536365", "536365", "536366"],
        "StockCode": ["85123A", "71053", "22632"],
        "Description": ["WHITE HANGING HEART", "WHITE METAL LANTERN", "HAND WARMER RED"],
        "Quantity": [6, 6, 6]
    })
    schema2 = detect_schema(df2)
    assert schema2.transaction_col == "InvoiceNo"
    assert schema2.product_col == "Description"  # Prefers human readable description over StockCode
    assert schema2.quantity_col == "Quantity"


def test_order_column_detection():
    # TEST 4: Order ID + Product + Quantity
    df4 = pd.DataFrame({"Order ID": [101, 101], "Product": ["Laptop", "Mouse"], "Quantity": [1, 2]})
    schema4 = detect_schema(df4)
    assert schema4.transaction_col == "Order ID"
    assert schema4.product_col == "Product"
    assert schema4.quantity_col == "Quantity"


def test_member_date_transaction_detection():
    # TEST 3: Groceries dataset (Member_number + Date + itemDescription)
    df3 = pd.DataFrame({
        "Member_number": [1808, 1808, 1808, 2550],
        "Date": ["21-07-2015", "21-07-2015", "21-07-2015", "22-07-2015"],
        "itemDescription": ["tropical fruit", "whole milk", "yogurt", "soda"]
    })
    schema3 = detect_schema(df3)
    assert schema3.transaction_col == ("Member_number", "Date")
    assert schema3.product_col == "itemDescription"
    assert schema3.inferred is True


def test_product_column_detection():
    df = pd.DataFrame({"Transaction ID": [1, 2], "Item": ["Bread", "Butter"]})
    schema = detect_schema(df)
    assert schema.transaction_col == "Transaction ID"
    assert schema.product_col == "Item"


def test_quantity_detection():
    df = pd.DataFrame({"InvoiceNo": ["101"], "ItemName": ["Apple"], "Qty": [5]})
    schema = detect_schema(df)
    assert schema.quantity_col == "Qty"


# ============================================================
# TRANSACTION BUILDER TESTS
# ============================================================

def test_online_retail_dataset():
    csv_data = """InvoiceNo,StockCode,Description,Quantity,InvoiceDate,CustomerID,Country
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,12/1/2010 8:26,17850,United Kingdom
536365,71053,WHITE METAL LANTERN,6,12/1/2010 8:26,17850,United Kingdom
536366,22632,HAND WARMER RED RETROSPOT,6,12/1/2010 8:28,17850,United Kingdom
"""
    result = process_dataset(StringIO(csv_data), filename="online_retail_test.csv")
    assert result["status"] == "success"
    assert len(result["transactions"]) == 2
    assert frozenset({"WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN"}) in result["transactions"]
    assert frozenset({"HAND WARMER RED RETROSPOT"}) in result["transactions"]


def test_groceries_dataset():
    csv_data = """Member_number,Date,itemDescription
1808,21-07-2015,tropical fruit
1808,21-07-2015,whole milk
1808,21-07-2015,yogurt
2550,22-07-2015,soda
"""
    result = process_dataset(StringIO(csv_data), filename="groceries_test.csv")
    assert result["status"] == "success"
    assert len(result["transactions"]) == 2
    assert frozenset({"tropical fruit", "whole milk", "yogurt"}) in result["transactions"]
    assert frozenset({"soda"}) in result["transactions"]
    assert result["metrics"]["inferred"] is True


def test_duplicate_rows_and_missing_values():
    csv_data = """TransactionID,ProductName,Quantity
1,Bread,2
1,Bread,2
1,Milk,1
2,Milk,NaN
2,,3
"""
    df = pd.read_csv(StringIO(csv_data))
    schema = detect_schema(df)
    transactions, names, stats = build_transactions_from_df(df, schema)

    assert len(transactions) == 1
    assert transactions[0] == frozenset({"Bread", "Milk"})
    assert stats["rows_removed"] > 0


def test_negative_quantity():
    csv_data = """TransactionID,Item,Quantity
1,Bread,5
1,Butter,-2
2,Milk,0
2,Juice,1
"""
    df = pd.read_csv(StringIO(csv_data))
    schema = detect_schema(df)
    transactions, names, stats = build_transactions_from_df(df, schema)

    assert len(transactions) == 2
    assert transactions[0] == frozenset({"Bread"})
    assert transactions[1] == frozenset({"Juice"})


# ============================================================
# EDGE CASES & INVALID DATASETS
# ============================================================

def test_invalid_dataset():
    csv_data = """Customer,Age,City
Alice,30,New York
Bob,25,London
"""
    result = process_dataset(StringIO(csv_data), filename="invalid.csv")
    assert result["status"] == "needs_mapping"
    assert "Customer" in result["columns"]


def test_empty_dataset():
    with pytest.raises(ValueError):
        process_dataset(StringIO(""), filename="empty.csv")


def test_missing_product_column():
    df = pd.DataFrame({"TransactionID": [1, 2, 3], "Price": [10, 20, 30]})
    schema = detect_schema(df)
    assert schema.product_col is None


def test_missing_transaction_information():
    df = pd.DataFrame({"Product": ["Apple", "Banana", "Cherry"]})
    schema = detect_schema(df)
    assert schema.transaction_col is None

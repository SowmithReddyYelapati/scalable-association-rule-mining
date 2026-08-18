from pathlib import Path
import sys
import time

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    Response,
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Required when running:
# python webapp\app.py
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXISTING MINING ENGINE
# ============================================================

from src.arm.apriori import apriori
from src.arm.rules import generate_rules
from src.arm.retail_loader import load_online_retail
from src.data.ingestion import process_dataset


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

app.secret_key = "scalable-association-rule-mining"


# ============================================================
# APPLICATION STATE
# ============================================================

analysis_state = {
    "dataset_name": None,

    "transactions": [],

    "product_names": {},

    "frequent_itemsets": {},

    "rules": [],

    "support": None,

    "confidence": None,

    "lift": None,

    "runtime": None,

    "metrics": {},
}


# ============================================================
# DATASET HELPERS
# ============================================================

def normalize_column_name(value):
    """
    Normalize a column name so common variations are treated equally.

    Examples:
        TransactionID -> transactionid
        Transaction ID -> transactionid
        transaction_id -> transactionid
        Invoice No -> invoiceno
    """
    return (
        str(value)
        .replace("\ufeff", "")
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace(".", "")
    )


def clean_dataframe_columns(df):
    """
    Clean CSV/Excel headers without changing the actual data.
    Handles BOM characters and duplicate-looking headers.
    """
    data = df.copy()

    data.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in data.columns
    ]

    return data


def find_column(columns, candidates):
    """
    Robust dataset column detection.

    Supports variations such as:
        TransactionID
        Transaction ID
        transaction_id
        InvoiceNo
        Invoice No
        Order ID
        ProductID
        Product ID
        StockCode
        Description
        ProductName
    """

    normalized_columns = {
        normalize_column_name(column): column
        for column in columns
    }

    # 1. Exact normalized matching
    for candidate in candidates:
        key = normalize_column_name(candidate)

        if key in normalized_columns:
            return normalized_columns[key]

    # 2. Safe partial matching
    normalized_candidates = [
        normalize_column_name(candidate)
        for candidate in candidates
    ]

    for column in columns:
        normalized_column = normalize_column_name(column)

        for candidate in normalized_candidates:
            if (
                candidate
                and (
                    candidate in normalized_column
                    or normalized_column in candidate
                )
            ):
                return column

    return None


def read_dataset(file_path):
    """
    Read CSV or Excel dataset.
    """

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    if extension == ".csv":

        df = pd.read_csv(file_path)
        return clean_dataframe_columns(df)

    if extension in [".xlsx", ".xls"]:

        df = pd.read_excel(file_path)
        return clean_dataframe_columns(df)

    raise ValueError(
        "Unsupported file type. "
        "Please upload CSV, XLSX, or XLS."
    )


# ============================================================
# CONVERT DATASET TO TRANSACTIONS
# ============================================================

def prepare_transactions(df):
    """
    Convert uploaded transaction data into:

        [
            frozenset({"P001", "P002"}),
            frozenset({"P001", "P003"}),
            ...
        ]

    Also creates a product-code -> product-name mapping.
    """

    if df is None or df.empty:

        raise ValueError(
            "The uploaded dataset is empty."
        )

    df = clean_dataframe_columns(df)

    print()
    print("-" * 60)
    print("DATASET COLUMNS DETECTED")
    print("-" * 60)
    print(list(df.columns))
    print("-" * 60)
    print()

    # --------------------------------------------------------
    # Transaction column
    # --------------------------------------------------------

    transaction_column = find_column(
        df.columns,
        [
            "TransactionID",
            "Transaction ID",
            "Transaction",
            "InvoiceNo",
            "Invoice No",
            "Invoice",
            "OrderID",
            "Order ID",
        ],
    )

    # --------------------------------------------------------
    # Product ID column
    # --------------------------------------------------------

    product_column = find_column(
        df.columns,
        [
            "ProductID",
            "Product ID",
            "ProductCode",
            "Product Code",
            "StockCode",
            "Stock Code",
            "ItemID",
            "Item ID",
            "ItemCode",
            "Item Code",
        ],
    )

    # --------------------------------------------------------
    # Product name column
    # --------------------------------------------------------

    description_column = find_column(
        df.columns,
        [
            "ProductName",
            "Product Name",
            "Product",
            "Description",
            "ItemName",
            "Item Name",
            "Item",
        ],
    )

    # --------------------------------------------------------
    # Quantity column
    # --------------------------------------------------------

    quantity_column = find_column(
        df.columns,
        [
            "Quantity",
            "Qty",
        ],
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if transaction_column is None:

        raise ValueError(
            "Transaction column not found. "
            f"Detected columns: {list(df.columns)}. "
            "Expected TransactionID, InvoiceNo, "
            "Transaction ID, Transaction, OrderID, "
            "or Order ID."
        )

    if (
        product_column is None
        and description_column is None
    ):

        raise ValueError(
            "Product column not found. "
            f"Detected columns: {list(df.columns)}. "
            "Expected ProductID, ProductName, "
            "StockCode, Description, Item, or Product."
        )

    data = df.copy()

    # --------------------------------------------------------
    # Remove missing transactions
    # --------------------------------------------------------

    data = data.dropna(
        subset=[transaction_column]
    )

    # --------------------------------------------------------
    # Remove cancelled Online Retail invoices
    # --------------------------------------------------------

    invoice_text = (
        data[transaction_column]
        .astype(str)
        .str.strip()
    )

    data = data[
        ~invoice_text
        .str.upper()
        .str.startswith("C")
    ]

    # --------------------------------------------------------
    # Quantity filtering
    # --------------------------------------------------------

    if quantity_column is not None:

        data[quantity_column] = pd.to_numeric(
            data[quantity_column],
            errors="coerce",
        )

        data = data[
            data[quantity_column] > 0
        ]

    # --------------------------------------------------------
    # Product identifier
    # --------------------------------------------------------

    if product_column is not None:

        data = data.dropna(
            subset=[product_column]
        )

        data["_product"] = (
            data[product_column]
            .astype(str)
            .str.strip()
        )

    else:

        data = data.dropna(
            subset=[description_column]
        )

        data["_product"] = (
            data[description_column]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Remove blank products
    # --------------------------------------------------------

    data = data[
        data["_product"] != ""
    ]

    # --------------------------------------------------------
    # Product ID -> Product Name
    # --------------------------------------------------------

    product_names = {}

    if (
        product_column is not None
        and description_column is not None
    ):

        mapping_data = data[
            [
                product_column,
                description_column,
            ]
        ].dropna()

        for _, row in (
            mapping_data.iterrows()
        ):

            code = str(
                row[product_column]
            ).strip()

            name = str(
                row[description_column]
            ).strip()

            if code and name:

                product_names.setdefault(
                    code,
                    name,
                )

    # --------------------------------------------------------
    # Build transaction list
    # --------------------------------------------------------

    transactions = []

    for _, group in data.groupby(
        transaction_column,
        sort=False,
    ):

        items = frozenset(
            group["_product"].tolist()
        )

        if items:

            transactions.append(
                items
            )

    if not transactions:

        raise ValueError(
            "No valid transactions could be created."
        )

    return (
        transactions,
        product_names,
    )


# ============================================================
# DATASET STATISTICS
# ============================================================

def calculate_statistics(
    df,
    transactions,
):
    """
    Calculate dataset statistics for UI.
    """

    products = set()

    for transaction in transactions:

        products.update(
            transaction
        )

    transaction_count = len(
        transactions
    )

    if transaction_count:

        average_items = (
            sum(
                len(transaction)
                for transaction in transactions
            )
            / transaction_count
        )

    else:

        average_items = 0

    largest_transaction = (
        max(
            len(transaction)
            for transaction in transactions
        )
        if transactions
        else 0
    )

    return {
        "rows": len(df),

        "transactions":
            transaction_count,

        "products":
            len(products),

        "average_items":
            round(
                average_items,
                2,
            ),

        "largest_transaction":
            largest_transaction,

        "columns":
            len(df.columns),
    }


# ============================================================
# PRODUCT DISPLAY
# ============================================================

def display_product(
    product,
    product_names,
):
    """
    Convert product ID to readable form.

    Example:

        22916

    becomes:

        HERB MARKER THYME (22916)
    """

    product = str(product)

    name = product_names.get(
        product
    )

    if name:

        return (
            f"{name} ({product})"
        )

    return product


def format_itemset(
    itemset,
    product_names,
):
    """
    Convert an itemset into readable product names.
    """

    return [
        display_product(
            item,
            product_names,
        )
        for item in sorted(
            itemset,
            key=str,
        )
    ]


# ============================================================
# FORMAT MINING RESULTS
# ============================================================

def format_results(
    min_lift=None,
):
    """
    Convert mining-engine objects into
    dictionaries suitable for Jinja.

    Optional min_lift filtering is applied
    only to generated rules.

    The Apriori engine itself remains unchanged.
    """

    product_names = (
        analysis_state[
            "product_names"
        ]
    )

    # --------------------------------------------------------
    # Frequent itemsets
    # --------------------------------------------------------

    itemsets = []

    for (
        itemset,
        support,
    ) in analysis_state[
        "frequent_itemsets"
    ].items():

        itemsets.append(
            {
                "items":
                    format_itemset(
                        itemset,
                        product_names,
                    ),

                "support":
                    float(support),

                "size":
                    len(itemset),
            }
        )

    itemsets.sort(
        key=lambda x: x["support"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Association rules
    # --------------------------------------------------------

    rules = []

    for rule in (
        analysis_state[
            "rules"
        ]
    ):

        rule_lift = float(
            rule.lift
        )

        # Apply optional lift filter
        if (
            min_lift is not None
            and rule_lift < min_lift
        ):
            continue

        rules.append(
            {
                "antecedent":
                    format_itemset(
                        rule.antecedent,
                        product_names,
                    ),

                "consequent":
                    format_itemset(
                        rule.consequent,
                        product_names,
                    ),

                "support":
                    float(rule.support),

                "confidence":
                    float(rule.confidence),

                "lift":
                    rule_lift,
            }
        )

    # Highest lift first
    rules.sort(
        key=lambda x: x["lift"],
        reverse=True,
    )

    return (
        itemsets,
        rules,
    )


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    preview = []

    if analysis_state.get("transactions"):

        product_names = analysis_state.get("product_names", {})

        for transaction in analysis_state["transactions"][:10]:

            preview.append(
                format_itemset(
                    transaction,
                    product_names,
                )
            )

    return render_template(
        "dashboard.html",
        analysis=analysis_state,
        preview=preview,
    )


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.route(
    "/upload",
    methods=["POST"],
)
def upload():

    file = request.files.get("dataset")

    if file is None or file.filename == "":
        flash("Please select a dataset first.", "error")
        return redirect(url_for("dashboard"))

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    if extension not in {".csv", ".xlsx", ".xls"}:
        flash("Only CSV, XLSX and XLS files are supported.", "error")
        return redirect(url_for("dashboard"))

    destination = UPLOAD_DIR / filename

    try:
        file.save(destination)

        result = process_dataset(destination, filename=filename)

        if result["status"] == "needs_mapping":
            analysis_state.update({
                "dataset_name": filename,
                "file_path": str(destination),
                "needs_mapping": True,
                "mapping_columns": result["columns"],
                "mapping_suggestion": result.get("schema_suggestion", {}),
                "transactions": [],
                "product_names": {},
                "metrics": {},
            })
            flash(result["message"], "error")
            return redirect(url_for("dashboard"))

        analysis_state.update({
            "dataset_name": filename,
            "file_path": str(destination),
            "needs_mapping": False,
            "transactions": result["transactions"],
            "product_names": result["product_names"],
            "frequent_itemsets": {},
            "rules": [],
            "support": None,
            "confidence": None,
            "lift": None,
            "runtime": None,
            "metrics": result["metrics"],
        })

        flash(f"{filename} uploaded and processed successfully.", "success")

    except Exception as exc:
        print("DATASET UPLOAD ERROR:", repr(exc))
        flash(
            "Dataset could not be processed. Please check that your CSV contains valid transaction and product information.",
            "error",
        )

    return redirect(url_for("dashboard"))


@app.route(
    "/map-columns",
    methods=["POST"],
)
def map_columns():

    file_path = analysis_state.get("file_path")
    filename = analysis_state.get("dataset_name", "uploaded_dataset.csv")

    if not file_path or not Path(file_path).exists():
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("dashboard"))

    txn_col = request.form.get("transaction_col")
    prod_col = request.form.get("product_col")
    qty_col = request.form.get("quantity_col")

    if not txn_col or not prod_col:
        flash("Please select both a transaction column and a product column.", "error")
        return redirect(url_for("dashboard"))

    manual_mapping = {
        "transaction": txn_col,
        "product": prod_col,
        "quantity": qty_col or "None",
    }

    try:
        result = process_dataset(Path(file_path), filename=filename, manual_mapping=manual_mapping)

        analysis_state.update({
            "needs_mapping": False,
            "transactions": result["transactions"],
            "product_names": result["product_names"],
            "frequent_itemsets": {},
            "rules": [],
            "support": None,
            "confidence": None,
            "lift": None,
            "runtime": None,
            "metrics": result["metrics"],
        })

        flash(f"Columns mapped successfully for {filename}.", "success")

    except Exception as exc:
        print("COLUMN MAPPING ERROR:", repr(exc))
        flash("Could not process dataset with selected column mapping.", "error")

    return redirect(url_for("dashboard"))


# ============================================================
# USE ONLINE RETAIL SAMPLE
# ============================================================

import json

@app.route(
    "/use-sample",
    methods=["POST"],
)
def use_sample():

    sample_excel = DATA_DIR / "Online Retail.xlsx"
    sample_csv = DATA_DIR / "online_retail_transactions.csv"
    product_json = DATA_DIR / "online_retail_product_names.json"

    if not sample_excel.exists() and not sample_csv.exists():
        flash(
            "Sample dataset could not be loaded. Please check the dataset file configuration.",
            "error",
        )
        return redirect(url_for("dashboard"))

    try:
        # Load transactions (uses cached CSV if present for instant loading)
        target_path = sample_excel if sample_excel.exists() else sample_csv
        transactions = load_online_retail(str(target_path))

        product_names = {}
        if product_json.exists():
            try:
                with open(product_json, "r", encoding="utf-8") as f:
                    product_names = json.load(f)
            except Exception:
                product_names = {}

        if not product_names and sample_excel.exists():
            try:
                df = pd.read_excel(sample_excel, usecols=["StockCode", "Description"]).dropna()
                for _, row in df.iterrows():
                    code = str(row["StockCode"]).strip()
                    name = str(row["Description"]).strip()
                    if code and name:
                        product_names.setdefault(code, name)
                with open(product_json, "w", encoding="utf-8") as f:
                    json.dump(product_names, f)
            except Exception:
                pass

        products = set()
        for transaction in transactions:
            products.update(transaction)

        transaction_count = len(transactions)
        average_items = (
            sum(len(t) for t in transactions) / transaction_count
            if transaction_count
            else 0
        )
        largest_transaction = (
            max(len(t) for t in transactions)
            if transactions
            else 0
        )

        statistics = {
            "rows": sum(len(t) for t in transactions),
            "transactions": transaction_count,
            "products": len(products),
            "average_items": round(average_items, 2),
            "largest_transaction": largest_transaction,
            "columns": 2,
        }

        analysis_state.update(
            {
                "dataset_name": "Online Retail Dataset",
                "transactions": transactions,
                "product_names": product_names,
                "frequent_itemsets": {},
                "rules": [],
                "support": None,
                "confidence": None,
                "lift": None,
                "runtime": None,
                "metrics": statistics,
            }
        )

        flash(
            "Online Retail sample dataset loaded successfully.",
            "success",
        )

    except Exception as exc:
        print("SAMPLE DATASET ERROR:", repr(exc))
        flash(
            "Sample dataset could not be loaded. Please check the dataset file configuration.",
            "error",
        )

    return redirect(url_for("dashboard"))


# ============================================================
# APRIORI ANALYSIS
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"],
)
def analyze():

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not analysis_state[
        "transactions"
    ]:

        flash(
            "Please upload a dataset or load the sample dataset first.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )

    try:

        # ----------------------------------------------------
        # Read support
        # ----------------------------------------------------

        support_raw = request.form.get(
            "support",
            "0.02",
        )

        # ----------------------------------------------------
        # Read confidence
        # ----------------------------------------------------

        confidence_raw = request.form.get(
            "confidence",
            "0.60",
        )

        # ----------------------------------------------------
        # Read lift
        #
        # Lift is NOT used by the Apriori engine.
        # It is used after rule generation to filter results.
        # ----------------------------------------------------

        lift_raw = request.form.get(
            "lift",
            "",
        )

        support = float(
            support_raw
        )

        confidence = float(
            confidence_raw
        )

        # Lift is calculated by the rules engine.
        # If the UI sends a lift filter, apply it after rule generation.
        # If no lift filter is supplied, do not filter rules.
        min_lift = (
            float(lift_raw)
            if str(lift_raw).strip() != ""
            else None
        )

        # ----------------------------------------------------
        # Validate support
        # ----------------------------------------------------

        # UI range:
        # Support    = 0.1% to 10%  -> 0.001 to 0.10
        # Confidence = 10% to 100% -> 0.10 to 1.00

        if not (
            0.001
            <= support
            <= 0.10
        ):

            raise ValueError(
                "Minimum support must be between 0.1% and 10%."
            )

        # ----------------------------------------------------
        # Validate confidence
        # ----------------------------------------------------

        if not (
            0.10
            <= confidence
            <= 1.0
        ):

            raise ValueError(
                "Minimum confidence must be between 10% and 100%."
            )

        # ----------------------------------------------------
        # Validate lift
        # ----------------------------------------------------

        if (
            min_lift is not None
            and min_lift < 0
        ):

            raise ValueError(
                "Minimum lift cannot be negative."
            )

        transactions = (
            analysis_state[
                "transactions"
            ]
        )

        # ====================================================
        # DEBUG
        # ====================================================

        print()

        print("=" * 60)
        print("STARTING APRIORI ANALYSIS")
        print("=" * 60)

        print(
            "Dataset:",
            analysis_state[
                "dataset_name"
            ],
        )

        print(
            "Transactions:",
            len(transactions),
        )

        print(
            "Minimum support:",
            support,
        )

        print(
            "Minimum confidence:",
            confidence,
        )

        print(
            "Minimum lift:",
            min_lift,
        )

        print("=" * 60)

        # ----------------------------------------------------
        # Start timer
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        # ====================================================
        # STEP 1
        # APRIORI
        # ====================================================

        print(
            "Running Apriori..."
        )

        frequent_itemsets = apriori(
            transactions,
            support,
        )

        print(
            "Frequent itemsets found:",
            len(
                frequent_itemsets
            ),
        )

        # ====================================================
        # STEP 2
        # ASSOCIATION RULES
        # ====================================================

        print(
            "Generating association rules..."
        )

        # Existing rules engine
        rules = generate_rules(
            frequent_itemsets,
            confidence,
        )

        print(
            "Rules generated before lift filter:",
            len(rules),
        )

        # ----------------------------------------------------
        # Lift filtering
        # ----------------------------------------------------

        if min_lift is not None:

            rules = [
                rule
                for rule in rules
                if float(rule.lift)
                >= min_lift
            ]

        print(
            "Rules after lift filter:",
            len(rules),
        )

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        runtime = (
            time.perf_counter()
            - start_time
        )

        print(
            "Runtime:",
            round(
                runtime,
                4,
            ),
            "seconds",
        )

        print("=" * 60)

        print(
            "APRIORI ANALYSIS COMPLETED"
        )

        print("=" * 60)

        print()

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        analysis_state.update(
            {
                "frequent_itemsets":
                    frequent_itemsets,

                "rules":
                    rules,

                "support":
                    support,

                "confidence":
                    confidence,

                "lift":
                    min_lift,

                "runtime":
                    runtime,
            }
        )

        flash(
            "Apriori analysis completed successfully.",
            "success",
        )

        return redirect(
            url_for("results")
        )

    except Exception as exc:

        print()

        print("=" * 60)
        print("APRIORI ANALYSIS ERROR")
        print("=" * 60)

        print(
            type(exc).__name__,
            ":",
            str(exc),
        )

        print("=" * 60)

        print()

        flash(
            "Analysis could not be completed. Please check your dataset and parameters.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )


# ============================================================
# RESULTS
# ============================================================

@app.route("/results")
def results():

    if (
        not analysis_state["frequent_itemsets"]
        and not analysis_state["rules"]
    ):

        flash(
            "No analysis results available. Run Apriori first.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )

    itemsets, rules = (
        format_results(
            analysis_state.get(
                "lift"
            )
        )
    )

    return render_template(
        "results.html",

        analysis=
            analysis_state,

        itemsets=
            itemsets,

        rules=
            rules,
    )


# ============================================================
# DOWNLOAD RULES CSV
# ============================================================

@app.route("/download-csv")
def download_csv():

    if not analysis_state.get("rules"):
        flash("No association rules available to download.", "error")
        return redirect(url_for("dashboard"))

    _, rules = format_results(analysis_state.get("lift"))

    rows = []
    rows.append("Antecedent,Consequent,Support,Confidence,Lift")

    for rule in rules:
        antecedent = " + ".join(rule["antecedent"]).replace('"', '""')
        consequent = " + ".join(rule["consequent"]).replace('"', '""')
        support = f'{rule["support"]:.6f}'
        confidence = f'{rule["confidence"]:.6f}'
        lift = f'{rule["lift"]:.6f}'

        rows.append(f'"{antecedent}","{consequent}",{support},{confidence},{lift}')

    csv_content = "\n".join(rows)

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=association_rules.csv"
        },
    )


# ============================================================
# RESET
# ============================================================

@app.route(
    "/reset",
    methods=["POST"],
)
def reset():

    analysis_state.update(
        {
            "dataset_name":
                None,

            "transactions":
                [],

            "product_names":
                {},

            "frequent_itemsets":
                {},

            "rules":
                [],

            "support":
                None,

            "confidence":
                None,

            "lift":
                None,

            "runtime":
                None,

            "metrics":
                {},
        }
    )

    flash(
        "Analysis session cleared.",
        "success",
    )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# LEARN
# ============================================================

@app.route("/learn")
def learn():

    return render_template(
        "learn.html"
    )


# ============================================================
# BENCHMARKS
# ============================================================

@app.route("/benchmarks")
def benchmarks():

    return render_template(
        "benchmarks.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status":
            "ok",

        "project":
            "Scalable Association Rule Mining",

        "current_algorithm":
            "Apriori",

        "future_algorithms":
            [
                "FP-Growth",
                "Eclat",
            ],
    }


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return (
        render_template(
            "base.html"
        ),
        404,
    )


@app.errorhandler(500)
def internal_server_error(error):

    print(
        "Internal Server Error:",
        error,
    )

    return (
        render_template(
            "base.html"
        ),
        500,
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 60)
    print(
        "SCALABLE ASSOCIATION RULE MINING"
    )
    print("=" * 60)

    print(
        "Algorithm : Apriori"
    )

    print(
        "UI        : Flask"
    )

    print(
        "Project   :",
        BASE_DIR,
    )

    print(
        "URL       : http://127.0.0.1:5000"
    )

    print("=" * 60)

    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
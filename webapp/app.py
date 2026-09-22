import tempfile
from pathlib import Path
import sys
import time
import json
import uuid

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    Response,
    send_from_directory,
    session,
)


# ============================================================
# PROJECT PATH
# ============================================================

WEBAPP_DIR = Path(__file__).resolve().parent
BASE_DIR = WEBAPP_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"

def _get_upload_dir() -> Path:
    try:
        local_upload = DATA_DIR / "uploads"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        local_upload.mkdir(parents=True, exist_ok=True)
        # Test writability
        test_file = local_upload / ".write_test"
        test_file.touch()
        test_file.unlink()
        return local_upload
    except Exception:
        temp_upload = Path(tempfile.gettempdir()) / "scalable_arm_uploads"
        try:
            temp_upload.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return temp_upload

UPLOAD_DIR = _get_upload_dir()


# ============================================================
# MINING ENGINES
# ============================================================

from src.arm.apriori import apriori
from src.arm.fp_growth import fp_growth
from src.arm.rules import generate_rules
from src.arm.retail_loader import load_online_retail
from src.data.ingestion import process_dataset


# ============================================================
# FLASK APPLICATION
# ============================================================

template_dir = WEBAPP_DIR / "templates"
if not template_dir.exists():
    alt_template = Path.cwd() / "webapp" / "templates"
    if alt_template.exists():
        template_dir = alt_template

static_dir = WEBAPP_DIR / "static"
if not static_dir.exists():
    alt_static = Path.cwd() / "webapp" / "static"
    if alt_static.exists():
        static_dir = alt_static

app = Flask(
    __name__,
    template_folder=str(template_dir),
    static_folder=str(static_dir),
    static_url_path="/static",
)

app.secret_key = "scalable-association-rule-mining"



# ============================================================
# APPLICATION STATE
# ============================================================

_STATE_DIR = Path(tempfile.gettempdir()) / "arm_states"
try:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

_EMPTY_STATE = {
    "dataset_name": None,
    "file_path": None,
    "needs_mapping": False,
    "mapping_columns": [],
    "mapping_suggestion": {},
    "transactions": [],
    "product_names": {},
    "frequent_itemsets": {},
    "rules": [],
    "support": None,
    "confidence": None,
    "lift": None,
    "runtime": None,
    "algorithm": "Apriori",
    "metrics": {},
}

# In-memory fallback (works on localhost, may be empty on serverless cold-start)
analysis_state = dict(_EMPTY_STATE)


def _state_file(state_id: str) -> Path:
    return _STATE_DIR / f"arm_{state_id}.json"


def _serialize(state: dict) -> dict:
    """Convert analysis_state to a JSON-serialisable dict."""
    s = {}
    for key, val in state.items():
        if key == "transactions":
            # list[frozenset[str]] -> list[list[str]]
            s[key] = [sorted(list(t)) for t in val]
        elif key == "frequent_itemsets":
            # dict[frozenset[str], float] -> list[[sorted items, support]]
            s[key] = [
                [sorted(list(itemset)), float(sup)]
                for itemset, sup in val.items()
            ]
        elif key == "rules":
            # list[AssociationRule | dict] -> list[dict]
            rules_out = []
            for r in val:
                if isinstance(r, dict):
                    rules_out.append({
                        "antecedent": sorted(list(r.get("antecedent", []))),
                        "consequent": sorted(list(r.get("consequent", []))),
                        "support": float(r.get("support", 0)),
                        "confidence": float(r.get("confidence", 0)),
                        "lift": float(r.get("lift", 0)),
                    })
                else:
                    rules_out.append({
                        "antecedent": sorted(list(r.antecedent)),
                        "consequent": sorted(list(r.consequent)),
                        "support": float(r.support),
                        "confidence": float(r.confidence),
                        "lift": float(r.lift),
                    })
            s[key] = rules_out
        else:
            s[key] = val
    return s


def _deserialize(s: dict) -> dict:
    """Convert JSON-loaded dict back to analysis_state format."""
    state = dict(_EMPTY_STATE)
    for key, val in s.items():
        if key == "transactions":
            state[key] = [frozenset(t) for t in val]
        elif key == "frequent_itemsets":
            state[key] = {
                frozenset(items): float(sup)
                for items, sup in val
            }
        elif key == "rules":
            # Keep as dicts; format_results() already handles dict rules
            state[key] = val
        else:
            state[key] = val
    return state


import zlib
import base64

def _encode_dataset_payload(state: dict) -> str:
    """Compress transactions, product_names, dataset_name, metrics, algorithm to base64 string."""
    try:
        if not state.get("transactions"):
            return ""
        data = {
            "dataset_name": state.get("dataset_name"),
            "transactions": [sorted(list(t)) for t in state.get("transactions", [])],
            "product_names": state.get("product_names", {}),
            "metrics": state.get("metrics", {}),
            "algorithm": state.get("algorithm", "Apriori"),
        }
        compressed = zlib.compress(json.dumps(data).encode("utf-8"), level=6)
        return base64.b64encode(compressed).decode("ascii")
    except Exception as exc:
        print("encode_payload error:", exc)
        return ""


def _decode_and_apply_dataset_payload(payload_str: str) -> None:
    """Decode and apply compressed dataset payload to analysis_state."""
    global analysis_state
    try:
        if not payload_str or not payload_str.strip():
            return
        raw = zlib.decompress(base64.b64decode(payload_str.strip().encode("ascii"))).decode("utf-8")
        data = json.loads(raw)
        if data.get("transactions"):
            analysis_state["transactions"] = [frozenset(t) for t in data["transactions"]]
        if data.get("product_names") is not None:
            analysis_state["product_names"] = data["product_names"]
        if data.get("dataset_name") is not None:
            analysis_state["dataset_name"] = data["dataset_name"]
        if data.get("metrics") is not None:
            analysis_state["metrics"] = data["metrics"]
        if data.get("algorithm") is not None:
            analysis_state["algorithm"] = data["algorithm"]
    except Exception as exc:
        print("decode_payload error:", exc)


def save_state() -> None:
    """Persist analysis_state to /tmp and store the ID in the Flask session."""
    try:
        state_id = session.get("arm_state_id")
        if not state_id:
            state_id = str(uuid.uuid4())
            session["arm_state_id"] = state_id
        sf = _state_file(state_id)
        sf.write_text(json.dumps(_serialize(analysis_state)), encoding="utf-8")
        # Also store small dataset payload in session for instant cross-container recovery
        payload = _encode_dataset_payload(analysis_state)
        if payload and len(payload) < 3000:
            session["arm_payload"] = payload
    except Exception as exc:
        print("save_state error:", exc)


def load_state() -> None:
    """Load persisted state from /tmp or session payload into global analysis_state."""
    global analysis_state
    try:
        state_id = session.get("arm_state_id")
        if state_id:
            sf = _state_file(state_id)
            if sf.exists():
                raw = json.loads(sf.read_text(encoding="utf-8"))
                loaded = _deserialize(raw)
                analysis_state.update(loaded)
                return
        # Fallback to session payload if /tmp was wiped by cold start
        session_payload = session.get("arm_payload")
        if session_payload:
            _decode_and_apply_dataset_payload(session_payload)
    except Exception as exc:
        print("load_state error:", exc)



# ============================================================
# DATASET HELPERS
# ============================================================

def normalize_column_name(value):
    """
    Normalize column names so common variations
    are treated equally.

    Examples:
        TransactionID
        Transaction ID
        transaction_id
        Invoice No
        Invoice_No
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
    Clean CSV/Excel headers without changing data.
    """

    data = df.copy()

    data.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()
        for column in data.columns
    ]

    return data


def find_column(columns, candidates):
    """
    Robust column detection.

    Supports common variations such as:

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

    # Exact normalized matching
    for candidate in candidates:

        key = normalize_column_name(candidate)

        if key in normalized_columns:
            return normalized_columns[key]

    # Safe partial matching
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

    Also creates product-code -> product-name mapping.
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

        for _, row in mapping_data.iterrows():

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
            transactions.append(items)

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
    Calculate dataset statistics.
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
    Convert product ID into readable form.
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
        key=lambda x: (
            x["size"],
            -x["support"],
        )
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

        if isinstance(rule, dict):
            antecedent = rule.get("antecedent", [])
            consequent = rule.get("consequent", [])
            support = float(rule.get("support", 0.0))
            confidence = float(rule.get("confidence", 0.0))
            rule_lift = float(rule.get("lift", 0.0))
        else:
            antecedent = rule.antecedent
            consequent = rule.consequent
            support = float(rule.support)
            confidence = float(rule.confidence)
            rule_lift = float(rule.lift)

        if (
            min_lift is not None
            and rule_lift < min_lift
        ):
            continue

        rules.append(
            {
                "antecedent":
                    format_itemset(
                        antecedent,
                        product_names,
                    ),
 
                "consequent":
                    format_itemset(
                        consequent,
                        product_names,
                    ),
 
                "support":
                    support,
 
                "confidence":
                    confidence,
 
                "lift":
                    rule_lift,
            }
        )

    rules.sort(
        key=lambda x: x["lift"],
        reverse=True,
    )

    return (
        itemsets,
        rules,
    )


# ============================================================
# HOME / DASHBOARD & STATIC
# ============================================================

@app.route("/static/<path:filename>")
def serve_static(filename):
    for candidate in [
        static_dir,
        BASE_DIR / "static",
        WEBAPP_DIR / "static",
        Path.cwd() / "static",
        Path.cwd() / "webapp" / "static",
    ]:
        if candidate.exists() and (candidate / filename).is_file():
            return send_from_directory(str(candidate), filename)
    return send_from_directory(str(static_dir), filename)


@app.route("/api/index")
@app.route("/api")
@app.route("/index")
@app.route("/")
def dashboard():

    load_state()

    preview = []

    if analysis_state.get("transactions"):

        product_names = (
            analysis_state.get(
                "product_names",
                {},
            )
        )

        for transaction in (
            analysis_state[
                "transactions"
            ][:10]
        ):

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
        dataset_payload=_encode_dataset_payload(analysis_state),
    )


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"],
)
def upload():

    if request.method == "GET":
        return redirect(url_for("dashboard"))

    load_state()

    file = request.files.get(
        "dataset"
    )

    if file is None or file.filename == "":

        flash(
            "Please select a dataset first.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )

    filename = Path(
        file.filename
    ).name

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in {
        ".csv",
        ".xlsx",
        ".xls",
    }:

        flash(
            "Only CSV, XLSX and XLS files are supported.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )

    destination = (
        UPLOAD_DIR / filename
    )

    try:

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        file.save(
            destination
        )

        result = process_dataset(
            destination,
            filename=filename,
        )

        if result["status"] == "needs_mapping":

            analysis_state.update(
                {
                    "dataset_name":
                        filename,

                    "file_path":
                        str(destination),

                    "needs_mapping":
                        True,

                    "mapping_columns":
                        result["columns"],

                    "mapping_suggestion":
                        result.get(
                            "schema_suggestion",
                            {},
                        ),

                    "transactions":
                        [],

                    "product_names":
                        {},

                    "frequent_itemsets":
                        {},

                    "rules":
                        [],

                    "metrics":
                        {},

                    "algorithm":
                        "Apriori",
                }
            )

            save_state()

            flash(
                result["message"],
                "error",
            )

            return redirect(
                url_for("dashboard")
            )

        analysis_state.update(
            {
                "dataset_name":
                    filename,

                "file_path":
                    str(destination),

                "needs_mapping":
                    False,

                "transactions":
                    result["transactions"],

                "product_names":
                    result["product_names"],

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

                "algorithm":
                    analysis_state.get("algorithm", "Apriori"),

                "metrics":
                    result["metrics"],
            }
        )

        save_state()

        flash(
            f"{filename} uploaded and processed successfully.",
            "success",
        )

    except Exception as exc:

        print(
            "DATASET UPLOAD ERROR:",
            repr(exc),
        )

        flash(
            "Dataset could not be processed. "
            "Please check that your CSV contains "
            "valid transaction and product information.",
            "error",
        )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# MANUAL COLUMN MAPPING
# ============================================================

@app.route(
    "/map-columns",
    methods=["GET", "POST"],
)
def map_columns():

    if request.method == "GET":
        return redirect(url_for("dashboard"))

    load_state()

    file_path = analysis_state.get(
        "file_path"
    )

    filename = analysis_state.get(
        "dataset_name",
        "uploaded_dataset.csv",
    )

    if (
        not file_path
        or not Path(file_path).exists()
    ):

        flash(
            "Please upload a dataset first.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )

    txn_col = request.form.get(
        "transaction_col"
    )

    prod_col = request.form.get(
        "product_col"
    )

    qty_col = request.form.get(
        "quantity_col"
    )

    if not txn_col or not prod_col:

        flash(
            "Please select both a transaction column and a product column.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )

    manual_mapping = {
        "transaction":
            txn_col,

        "product":
            prod_col,

        "quantity":
            qty_col or "None",
    }

    try:

        result = process_dataset(
            Path(file_path),
            filename=filename,
            manual_mapping=manual_mapping,
        )

        analysis_state.update(
            {
                "needs_mapping":
                    False,

                "transactions":
                    result["transactions"],

                "product_names":
                    result["product_names"],

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

                "algorithm":
                    analysis_state.get("algorithm", "Apriori"),

                "metrics":
                    result["metrics"],
            }
        )

        save_state()

        flash(
            f"Columns mapped successfully for {filename}.",
            "success",
        )

    except Exception as exc:

        print(
            "COLUMN MAPPING ERROR:",
            repr(exc),
        )

        flash(
            "Could not process dataset with selected column mapping.",
            "error",
        )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# USE ONLINE RETAIL SAMPLE
# ============================================================

@app.route(
    "/use-sample",
    methods=["GET", "POST"],
)
def use_sample():

    if request.method == "GET":
        return redirect(url_for("dashboard"))

    load_state()

    sample_excel = (
        DATA_DIR / "Online Retail.xlsx"
    )

    sample_csv = (
        DATA_DIR / "online_retail_transactions.csv"
    )

    product_json = (
        DATA_DIR / "online_retail_product_names.json"
    )

    if (
        not sample_excel.exists()
        and not sample_csv.exists()
    ):

        flash(
            "Sample dataset could not be loaded. "
            "Please check the dataset file configuration.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )

    try:

        target_path = (
            sample_excel
            if sample_excel.exists()
            else sample_csv
        )

        transactions = load_online_retail(
            str(target_path)
        )

        product_names = {}

        if product_json.exists():

            try:

                with open(
                    product_json,
                    "r",
                    encoding="utf-8",
                ) as f:

                    product_names = json.load(f)

            except Exception:

                product_names = {}

        if (
            not product_names
            and sample_excel.exists()
        ):

            try:

                df = pd.read_excel(
                    sample_excel,
                    usecols=[
                        "StockCode",
                        "Description",
                    ],
                ).dropna()

                for _, row in df.iterrows():

                    code = str(
                        row["StockCode"]
                    ).strip()

                    name = str(
                        row["Description"]
                    ).strip()

                    if code and name:

                        product_names.setdefault(
                            code,
                            name,
                        )

                with open(
                    product_json,
                    "w",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        product_names,
                        f,
                    )

            except Exception:

                pass

        products = set()

        for transaction in transactions:

            products.update(
                transaction
            )

        transaction_count = len(
            transactions
        )

        average_items = (
            sum(
                len(t)
                for t in transactions
            )
            / transaction_count
            if transaction_count
            else 0
        )

        largest_transaction = (
            max(
                len(t)
                for t in transactions
            )
            if transactions
            else 0
        )

        statistics = {
            "rows":
                sum(
                    len(t)
                    for t in transactions
                ),

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
                2,
        }

        analysis_state.update(
            {
                "dataset_name":
                    "Online Retail Dataset",

                "file_path":
                    None,

                "transactions":
                    transactions,

                "product_names":
                    product_names,

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

                "algorithm":
                    analysis_state.get("algorithm", "Apriori"),

                "metrics":
                    statistics,
            }
        )

        save_state()

        flash(
            "Online Retail sample dataset loaded successfully.",
            "success",
        )

    except Exception as exc:

        print(
            "SAMPLE DATASET ERROR:",
            repr(exc),
        )

        flash(
            "Sample dataset could not be loaded. "
            "Please check the dataset file configuration.",
            "error",
        )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# ANALYSIS
# ============================================================

@app.route(
    "/analyze",
    methods=["GET", "POST"],
)
def analyze():

    if request.method == "GET":
        load_state()
        return redirect(url_for("results") if analysis_state.get("rules") else url_for("dashboard"))

    load_state()

    # Fallback to submitted dataset_payload if transactions were lost across serverless containers
    payload = request.form.get("dataset_payload")
    if payload:
        _decode_and_apply_dataset_payload(payload)

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
        # Algorithm Dispatch Mapping
        # ----------------------------------------------------

        algorithm_raw = request.form.get(
            "algorithm",
            "apriori",
        )

        algorithm_key = (
            str(algorithm_raw)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if algorithm_key in {
            "fp_growth",
            "fpgrowth",
            "fp",
        }:
            algorithm = "FP-Growth"
            mining_func = fp_growth

        elif algorithm_key == "eclat":
            flash(
                "Eclat is coming soon. Please select Apriori or FP-Growth.",
                "error",
            )
            return redirect(
                url_for("dashboard")
            )

        else:
            algorithm = "Apriori"
            mining_func = apriori

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        support = float(
            request.form.get(
                "support",
                "0.02",
            )
        )

        confidence = float(
            request.form.get(
                "confidence",
                "0.60",
            )
        )

        lift_raw = request.form.get(
            "lift",
            "",
        )

        min_lift = (
            float(lift_raw)
            if str(lift_raw).strip()
            else None
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not (
            0.001
            <= support
            <= 0.10
        ):

            raise ValueError(
                "Minimum support must be between 0.1% and 10%."
            )

        if not (
            0.10
            <= confidence
            <= 1.0
        ):

            raise ValueError(
                "Minimum confidence must be between 10% and 100%."
            )

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

        print()
        print("=" * 70)
        print(
            "STARTING ASSOCIATION RULE MINING"
        )
        print("=" * 70)

        print(
            "Dataset:",
            analysis_state[
                "dataset_name"
            ],
        )

        print(
            "Algorithm:",
            algorithm,
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

        print("=" * 70)

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Frequent itemset mining
        # ----------------------------------------------------

        print(
            f"Running {algorithm}..."
        )

        frequent_itemsets = mining_func(
            transactions,
            support,
        )

        print(
            "Frequent itemsets found:",
            len(frequent_itemsets),
        )

        # ----------------------------------------------------
        # Association rules
        # ----------------------------------------------------

        print(
            "Generating association rules..."
        )

        rules = generate_rules(
            frequent_itemsets,
            confidence,
        )

        print(
            "Rules generated before lift filter:",
            len(rules),
        )

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

        print("=" * 70)

        print(
            f"{algorithm.upper()} ANALYSIS COMPLETED"
        )

        print("=" * 70)
        print()

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

                "algorithm":
                    algorithm,
            }
        )

        save_state()

        flash(
            f"{algorithm} analysis completed successfully.",
            "success",
        )

        return redirect(
            url_for("results")
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "ASSOCIATION RULE MINING ERROR"
        )
        print("=" * 70)

        print(
            type(exc).__name__,
            ":",
            str(exc),
        )

        print("=" * 70)
        print()

        flash(
            "Analysis could not be completed. "
            "Please check your dataset and parameters.",
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

    load_state()

    if (
        not analysis_state[
            "frequent_itemsets"
        ]
        and not analysis_state[
            "rules"
        ]
    ):

        flash(
            "No analysis results available. "
            "Run an analysis first.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )

    itemsets, rules = format_results(
        analysis_state.get(
            "lift"
        )
    )

    return render_template(
        "results.html",
        analysis=analysis_state,
        itemsets=itemsets,
        rules=rules,
    )


# ============================================================
# DOWNLOAD RULES CSV
# ============================================================

@app.route("/download-csv")
def download_csv():

    load_state()

    if not analysis_state.get(
        "rules"
    ):

        flash(
            "No association rules available to download.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )

    _, rules = format_results(
        analysis_state.get(
            "lift"
        )
    )

    rows = [
        "Antecedent,Consequent,Support,Confidence,Lift"
    ]

    for rule in rules:

        antecedent = (
            " + ".join(
                rule["antecedent"]
            )
            .replace(
                '"',
                '""',
            )
        )

        consequent = (
            " + ".join(
                rule["consequent"]
            )
            .replace(
                '"',
                '""',
            )
        )

        rows.append(
            f'"{antecedent}",'
            f'"{consequent}",'
            f'{rule["support"]:.6f},'
            f'{rule["confidence"]:.6f},'
            f'{rule["lift"]:.6f}'
        )

    csv_content = "\n".join(
        rows
    )

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; "
                "filename=association_rules.csv"
        },
    )


# ============================================================
# RESET
# ============================================================

@app.route(
    "/reset",
    methods=["GET", "POST"],
)
def reset():

    load_state()

    analysis_state.update(
        {
            "dataset_name": None,
            "file_path": None,

            "needs_mapping": False,
            "mapping_columns": [],
            "mapping_suggestion": {},

            "transactions": [],
            "product_names": {},

            "frequent_itemsets": {},
            "rules": [],

            "support": None,
            "confidence": None,
            "lift": None,

            "runtime": None,

            "algorithm": "Apriori",

            "metrics": {},
        }
    )

    save_state()

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
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "ok",

        "project":
            "Scalable Association Rule Mining",

        "available_algorithms":
            [
                "Apriori",
                "FP-Growth",
            ],

        "future_algorithms":
            [
                "Eclat",
            ],

        "current_algorithm":
            analysis_state.get(
                "algorithm",
                "Apriori",
            ),
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

    print("=" * 70)

    print(
        "SCALABLE ASSOCIATION RULE MINING"
    )

    print("=" * 70)

    print(
        "Algorithms : Apriori + FP-Growth"
    )

    print(
        "UI         : Flask"
    )

    print(
        "Project    :",
        BASE_DIR,
    )

    print(
        "URL        : http://127.0.0.1:5000"
    )

    print("=" * 70)

    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
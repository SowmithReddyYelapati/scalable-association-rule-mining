"""
============================================================
SCALABLE ASSOCIATION RULE MINING
Flask Web Application
============================================================

Current algorithm:
    Apriori

Future algorithms:
    FP-Growth
    Eclat

Workflow:
    Upload Dataset
        ↓
    Dataset Validation
        ↓
    Transaction Conversion
        ↓
    Configure Support / Confidence
        ↓
    Apriori Frequent Itemset Mining
        ↓
    Association Rule Generation
        ↓
    Support / Confidence / Lift
        ↓
    Results

============================================================
"""

from __future__ import annotations

import csv
import inspect
import os
import sys
import time
from pathlib import Path
from itertools import combinations

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

app.secret_key = "scalable-association-rule-mining-dev-key"


# ============================================================
# OPTIONAL PANDAS
# ============================================================

try:
    import pandas as pd

    PANDAS_AVAILABLE = True

except ImportError:

    pd = None

    PANDAS_AVAILABLE = False


# ============================================================
# APRIORI IMPORT
# ============================================================

APRIORI_AVAILABLE = False
apriori_function = None


try:

    from src.arm.apriori import apriori as apriori_function

    APRIORI_AVAILABLE = True

except Exception as exc:

    print(
        "[WARNING] Could not import custom Apriori:",
        exc,
    )


# ============================================================
# RULE GENERATION IMPORT
# ============================================================

RULES_AVAILABLE = False
rules_module = None


try:

    import src.arm.rules as rules_module

    RULES_AVAILABLE = True

except Exception as exc:

    print(
        "[WARNING] Could not import custom rules module:",
        exc,
    )


# ============================================================
# APPLICATION STATE
# ============================================================

analysis_state = {

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    "dataset_name": None,

    "transactions": [],

    "product_names": {},

    "metrics": {},

    # --------------------------------------------------------
    # Mining
    # --------------------------------------------------------

    "frequent_itemsets": {},

    "itemsets": [],

    "rules": [],

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    "support": 0.02,

    "confidence": 0.60,

    "lift": 0.0,

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    "runtime": 0.0,

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    "analysis_complete": False,

}


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}


MAX_RESULT_ROWS = 100


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def allowed_file(filename: str) -> bool:

    if not filename:
        return False

    extension = Path(filename).suffix.lower()

    return extension in ALLOWED_EXTENSIONS


def reset_analysis():

    analysis_state.update({

        "dataset_name": None,

        "transactions": [],

        "product_names": {},

        "metrics": {},

        "frequent_itemsets": {},

        "itemsets": [],

        "rules": [],

        "support": 0.02,

        "confidence": 0.60,

        "lift": 0.0,

        "runtime": 0.0,

        "analysis_complete": False,

    })


def clean_product_name(value) -> str:

    if value is None:
        return ""

    value = str(value).strip()

    return value


def normalize_item(item):

    """
    Convert different item representations into a string.
    """

    if item is None:
        return ""

    return clean_product_name(item)


def normalize_itemset(itemset):

    """
    Convert an itemset into a sorted list.

    Handles:

        set
        frozenset
        tuple
        list
        single value
    """

    if itemset is None:
        return []

    if isinstance(
        itemset,
        (set, frozenset, list, tuple),
    ):

        items = [
            normalize_item(item)
            for item in itemset
        ]

    else:

        items = [
            normalize_item(itemset)
        ]

    items = [
        item
        for item in items
        if item
    ]

    return sorted(
        set(items),
        key=lambda x: str(x).lower(),
    )


def itemset_key(items):

    return frozenset(
        normalize_itemset(items)
    )


def calculate_support(
    itemset,
    transactions,
):

    if not transactions:
        return 0.0

    target = set(
        normalize_itemset(itemset)
    )

    if not target:
        return 0.0

    count = 0

    for transaction in transactions:

        transaction_set = set(
            normalize_itemset(transaction)
        )

        if target.issubset(
            transaction_set
        ):

            count += 1

    return count / len(transactions)


def calculate_confidence(
    antecedent,
    consequent,
    transactions,
):

    antecedent = set(
        normalize_itemset(antecedent)
    )

    consequent = set(
        normalize_itemset(consequent)
    )

    if not antecedent:
        return 0.0

    antecedent_support = calculate_support(
        antecedent,
        transactions,
    )

    if antecedent_support == 0:
        return 0.0

    combined = antecedent | consequent

    combined_support = calculate_support(
        combined,
        transactions,
    )

    return (
        combined_support
        / antecedent_support
    )


def calculate_lift(
    antecedent,
    consequent,
    transactions,
):

    antecedent = set(
        normalize_itemset(antecedent)
    )

    consequent = set(
        normalize_itemset(consequent)
    )

    if not antecedent or not consequent:
        return 0.0

    antecedent_support = calculate_support(
        antecedent,
        transactions,
    )

    consequent_support = calculate_support(
        consequent,
        transactions,
    )

    combined_support = calculate_support(
        antecedent | consequent,
        transactions,
    )

    if (
        antecedent_support == 0
        or consequent_support == 0
    ):
        return 0.0

    return (
        combined_support
        / (
            antecedent_support
            * consequent_support
        )
    )


# ============================================================
# DATASET LOADING
# ============================================================

def detect_column(
    columns,
    candidates,
):

    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:

        candidate_lower = candidate.lower()

        if candidate_lower in normalized:

            return normalized[
                candidate_lower
            ]

    # Partial matching

    for column in columns:

        column_lower = (
            str(column)
            .strip()
            .lower()
        )

        for candidate in candidates:

            if candidate.lower() in column_lower:

                return column

    return None


def read_csv_dataset(file_path):

    """
    Read CSV using Python's csv module.

    Detects common transaction/product columns.

    Supported examples:

        TransactionID, ProductName
        InvoiceNo, Description
        Transaction, Product
        OrderID, Product
    """

    transactions_map = {}

    product_names = {}

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:

            raise ValueError(
                "CSV file does not contain a header row."
            )

        columns = reader.fieldnames

        transaction_column = detect_column(
            columns,
            [
                "TransactionID",
                "Transaction",
                "Transaction_ID",
                "InvoiceNo",
                "Invoice",
                "OrderID",
                "Order_ID",
                "BasketID",
                "Basket",
                "BillNo",
            ],
        )

        product_column = detect_column(
            columns,
            [
                "ProductName",
                "Product",
                "Product_Name",
                "Description",
                "Item",
                "ItemName",
                "StockCode",
                "ProductID",
            ],
        )

        if transaction_column is None:

            raise ValueError(
                "Could not identify the transaction column. "
                "Expected something like TransactionID, "
                "InvoiceNo or OrderID."
            )

        if product_column is None:

            raise ValueError(
                "Could not identify the product column. "
                "Expected something like ProductName, "
                "Description, Product or Item."
            )

        for row in reader:

            transaction_id = clean_product_name(
                row.get(transaction_column)
            )

            product = clean_product_name(
                row.get(product_column)
            )

            if not transaction_id or not product:
                continue

            # ------------------------------------------------
            # Optional quantity
            # ------------------------------------------------

            quantity_column = detect_column(
                columns,
                [
                    "Quantity",
                    "Qty",
                ],
            )

            if quantity_column:

                quantity_value = row.get(
                    quantity_column
                )

                try:

                    quantity = float(
                        quantity_value
                    )

                    if quantity <= 0:
                        continue

                except (
                    ValueError,
                    TypeError,
                ):

                    pass

            if (
                transaction_id
                not in transactions_map
            ):

                transactions_map[
                    transaction_id
                ] = set()

            transactions_map[
                transaction_id
            ].add(product)

            # Product mapping

            if (
                product
                not in product_names
            ):

                product_names[
                    product
                ] = product

    transactions = list(
        transactions_map.values()
    )

    return (
        transactions,
        product_names,
    )


def read_excel_dataset(file_path):

    if not PANDAS_AVAILABLE:

        raise ValueError(
            "Pandas is required for Excel files. "
            "Install it using: pip install pandas openpyxl"
        )

    dataframe = pd.read_excel(
        file_path
    )

    if dataframe.empty:

        raise ValueError(
            "The uploaded Excel file is empty."
        )

    columns = list(
        dataframe.columns
    )

    transaction_column = detect_column(
        columns,
        [
            "TransactionID",
            "Transaction",
            "InvoiceNo",
            "Invoice",
            "OrderID",
            "Order_ID",
            "BasketID",
        ],
    )

    product_column = detect_column(
        columns,
        [
            "ProductName",
            "Product",
            "Description",
            "Item",
            "ItemName",
            "StockCode",
            "ProductID",
        ],
    )

    if transaction_column is None:

        raise ValueError(
            "Could not identify the transaction column."
        )

    if product_column is None:

        raise ValueError(
            "Could not identify the product column."
        )

    transactions_map = {}

    product_names = {}

    for _, row in dataframe.iterrows():

        transaction_id = clean_product_name(
            row[transaction_column]
        )

        product = clean_product_name(
            row[product_column]
        )

        if (
            not transaction_id
            or not product
        ):
            continue

        quantity_column = detect_column(
            columns,
            [
                "Quantity",
                "Qty",
            ],
        )

        if quantity_column:

            try:

                quantity = float(
                    row[quantity_column]
                )

                if quantity <= 0:
                    continue

            except (
                ValueError,
                TypeError,
            ):

                pass

        if (
            transaction_id
            not in transactions_map
        ):

            transactions_map[
                transaction_id
            ] = set()

        transactions_map[
            transaction_id
        ].add(product)

        product_names[
            product
        ] = product

    return (
        list(
            transactions_map.values()
        ),
        product_names,
    )


def load_dataset(
    file_path,
):

    extension = (
        Path(file_path)
        .suffix
        .lower()
    )

    if extension == ".csv":

        return read_csv_dataset(
            file_path
        )

    if extension in {
        ".xlsx",
        ".xls",
    }:

        return read_excel_dataset(
            file_path
        )

    raise ValueError(
        "Unsupported file type."
    )


# ============================================================
# DATASET METRICS
# ============================================================

def calculate_dataset_metrics(
    transactions,
):

    transaction_count = len(
        transactions
    )

    if transaction_count == 0:

        return {

            "transactions": 0,

            "products": 0,

            "average_items": 0.0,

            "largest_transaction": 0,

        }

    product_set = set()

    transaction_sizes = []

    for transaction in transactions:

        items = set(
            normalize_itemset(
                transaction
            )
        )

        product_set.update(
            items
        )

        transaction_sizes.append(
            len(items)
        )

    return {

        "transactions":
            transaction_count,

        "products":
            len(product_set),

        "average_items":
            sum(transaction_sizes)
            / transaction_count,

        "largest_transaction":
            max(transaction_sizes)
            if transaction_sizes
            else 0,

    }


# ============================================================
# APRIORI ENGINE
# ============================================================

def fallback_apriori(
    transactions,
    minimum_support,
):

    """
    Small pure-Python fallback Apriori implementation.

    This is used only if the project's custom Apriori
    function cannot be called successfully.

    The project's own Apriori remains the preferred engine.
    """

    transactions = [
        set(
            normalize_itemset(
                transaction
            )
        )
        for transaction in transactions
    ]

    transaction_count = len(
        transactions
    )

    if transaction_count == 0:

        return {}

    minimum_count = (
        minimum_support
        * transaction_count
    )

    frequent = {}

    # --------------------------------------------------------
    # 1-itemsets
    # --------------------------------------------------------

    item_counts = {}

    for transaction in transactions:

        for item in transaction:

            item_counts[item] = (
                item_counts.get(item, 0)
                + 1
            )

    current = {}

    for item, count in item_counts.items():

        support = (
            count
            / transaction_count
        )

        if count >= minimum_count:

            current[
                frozenset([item])
            ] = support

    frequent.update(current)

    # --------------------------------------------------------
    # Larger itemsets
    # --------------------------------------------------------

    k = 2

    while current:

        previous_items = list(
            current.keys()
        )

        candidates = set()

        for i in range(
            len(previous_items)
        ):

            for j in range(
                i + 1,
                len(previous_items),
            ):

                union = (
                    previous_items[i]
                    | previous_items[j]
                )

                if len(union) == k:

                    # Apriori pruning:
                    # every k-1 subset must be frequent

                    all_subsets_frequent = True

                    for subset in combinations(
                        union,
                        k - 1,
                    ):

                        if (
                            frozenset(subset)
                            not in current
                        ):

                            all_subsets_frequent = False
                            break

                    if all_subsets_frequent:

                        candidates.add(
                            frozenset(union)
                        )

        if not candidates:
            break

        next_level = {}

        for candidate in candidates:

            count = 0

            for transaction in transactions:

                if candidate.issubset(
                    transaction
                ):

                    count += 1

            support = (
                count
                / transaction_count
            )

            if count >= minimum_count:

                next_level[
                    candidate
                ] = support

        if not next_level:
            break

        frequent.update(
            next_level
        )

        current = next_level

        k += 1

    return frequent


def call_custom_apriori(
    transactions,
    minimum_support,
):

    """
    Try to call the project's existing Apriori
    implementation.

    Different project implementations often use
    slightly different parameter names/signatures.

    This adapter keeps the Flask UI independent of
    those details.
    """

    if not APRIORI_AVAILABLE:
        return None

    function = apriori_function

    try:

        signature = inspect.signature(
            function
        )

        parameters = list(
            signature.parameters.values()
        )

        kwargs = {}

        positional = []

        for parameter in parameters:

            name = parameter.name.lower()

            if name in {
                "transactions",
                "transaction",
                "data",
                "dataset",
                "baskets",
            }:

                positional.append(
                    transactions
                )

            elif name in {
                "min_support",
                "minimum_support",
                "support",
            }:

                positional.append(
                    minimum_support
                )

        # ----------------------------------------------------
        # First attempt using positional arguments
        # ----------------------------------------------------

        if len(positional) >= 2:

            return function(
                *positional[:2]
            )

        # ----------------------------------------------------
        # Common standard signature
        # ----------------------------------------------------

        try:

            return function(
                transactions,
                minimum_support,
            )

        except TypeError:

            pass

        # ----------------------------------------------------
        # Keyword fallback
        # ----------------------------------------------------

        kwargs = {}

        for parameter in parameters:

            name = parameter.name.lower()

            if name in {
                "transactions",
                "transaction",
                "data",
                "dataset",
                "baskets",
            }:

                kwargs[
                    parameter.name
                ] = transactions

            elif name in {
                "min_support",
                "minimum_support",
                "support",
            }:

                kwargs[
                    parameter.name
                ] = minimum_support

        if kwargs:

            return function(
                **kwargs
            )

    except Exception as exc:

        print(
            "[WARNING] Custom Apriori failed:",
            exc,
        )

    return None


def run_apriori(
    transactions,
    minimum_support,
):

    """
    Preferred order:

        1. Existing project Apriori
        2. Fallback Apriori

    """

    result = call_custom_apriori(
        transactions,
        minimum_support,
    )

    if result is not None:

        return result

    print(
        "[INFO] Using fallback Apriori implementation."
    )

    return fallback_apriori(
        transactions,
        minimum_support,
    )


# ============================================================
# NORMALIZE APRIORI OUTPUT
# ============================================================

def normalize_frequent_itemsets(
    result,
    transactions,
):

    """
    Convert different Apriori output formats into:

    {
        frozenset(...): support
    }
    """

    normalized = {}

    if result is None:
        return normalized

    # --------------------------------------------------------
    # Dictionary output
    # --------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        for key, value in result.items():

            # Nested dictionary
            if isinstance(
                value,
                dict,
            ):

                support = (
                    value.get("support")
                    or value.get("Support")
                    or value.get("frequency")
                    or value.get("frequency_count")
                )

                items = (
                    value.get("items")
                    or value.get("itemset")
                    or key
                )

            else:

                support = value
                items = key

            items = normalize_itemset(
                items
            )

            if not items:
                continue

            try:

                support = float(
                    support
                )

            except (
                TypeError,
                ValueError,
            ):

                support = calculate_support(
                    items,
                    transactions,
                )

            # Some implementations return
            # absolute counts instead of support.

            if support > 1:

                if len(transactions) > 0:

                    support = (
                        support
                        / len(transactions)
                    )

            normalized[
                frozenset(items)
            ] = support

        return normalized

    # --------------------------------------------------------
    # List / tuple output
    # --------------------------------------------------------

    if isinstance(
        result,
        (list, tuple),
    ):

        for entry in result:

            items = None
            support = None

            if isinstance(
                entry,
                dict,
            ):

                items = (
                    entry.get("items")
                    or entry.get("itemset")
                    or entry.get("items_set")
                )

                support = (
                    entry.get("support")
                    or entry.get("Support")
                )

            elif isinstance(
                entry,
                (list, tuple),
            ):

                if len(entry) >= 2:

                    items = entry[0]
                    support = entry[1]

            else:

                items = entry

            items = normalize_itemset(
                items
            )

            if not items:
                continue

            if support is None:

                support = calculate_support(
                    items,
                    transactions,
                )

            try:

                support = float(
                    support
                )

            except (
                TypeError,
                ValueError,
            ):

                support = calculate_support(
                    items,
                    transactions,
                )

            if support > 1:

                support = (
                    support
                    / len(transactions)
                )

            normalized[
                frozenset(items)
            ] = support

        return normalized

    return normalized


# ============================================================
# ASSOCIATION RULE GENERATION
# ============================================================

def generate_rules_fallback(
    frequent_itemsets,
    transactions,
    minimum_confidence,
):

    rules = []

    for itemset, itemset_support in (
        frequent_itemsets.items()
    ):

        itemset = set(itemset)

        if len(itemset) < 2:
            continue

        item_list = list(itemset)

        # Every non-empty proper subset
        # becomes a possible antecedent.

        for size in range(
            1,
            len(item_list),
        ):

            for antecedent_tuple in combinations(
                item_list,
                size,
            ):

                antecedent = frozenset(
                    antecedent_tuple
                )

                consequent = (
                    frozenset(itemset)
                    - antecedent
                )

                confidence = (
                    calculate_confidence(
                        antecedent,
                        consequent,
                        transactions,
                    )
                )

                if (
                    confidence
                    < minimum_confidence
                ):
                    continue

                lift = calculate_lift(
                    antecedent,
                    consequent,
                    transactions,
                )

                rules.append({

                    "antecedent":
                        sorted(
                            antecedent
                        ),

                    "consequent":
                        sorted(
                            consequent
                        ),

                    "support":
                        float(
                            itemset_support
                        ),

                    "confidence":
                        float(
                            confidence
                        ),

                    "lift":
                        float(
                            lift
                        ),

                })

    return rules


def call_custom_rules(
    frequent_itemsets,
    transactions,
    minimum_confidence,
):

    """
    Attempt to use the project's existing rules module.

    If the module doesn't expose a compatible generator,
    the Flask application falls back to its own correct
    rule-generation implementation.
    """

    if not RULES_AVAILABLE:

        return None

    possible_functions = [

        "generate_association_rules",

        "generate_rules",

        "association_rules",

        "generate_association_rule",

        "create_rules",

    ]

    function = None

    for name in possible_functions:

        candidate = getattr(
            rules_module,
            name,
            None,
        )

        if callable(candidate):

            function = candidate
            break

    if function is None:

        return None

    try:

        signature = inspect.signature(
            function
        )

        parameters = list(
            signature.parameters.values()
        )

        arguments = []

        for parameter in parameters:

            name = parameter.name.lower()

            if name in {
                "frequent_itemsets",
                "frequent_sets",
                "itemsets",
                "patterns",
                "frequent_patterns",
            }:

                arguments.append(
                    frequent_itemsets
                )

            elif name in {
                "transactions",
                "transaction",
                "data",
                "dataset",
            }:

                arguments.append(
                    transactions
                )

            elif name in {
                "min_confidence",
                "minimum_confidence",
                "confidence",
            }:

                arguments.append(
                    minimum_confidence
                )

        if arguments:

            return function(
                *arguments
            )

        # Common signature

        return function(
            frequent_itemsets,
            minimum_confidence,
        )

    except Exception as exc:

        print(
            "[WARNING] Custom rule generation failed:",
            exc,
        )

        return None


def normalize_rules(
    result,
    transactions,
):

    """
    Normalize rules into:

    {
        antecedent: [...],
        consequent: [...],
        support: float,
        confidence: float,
        lift: float
    }
    """

    normalized = []

    if result is None:

        return normalized

    if isinstance(
        result,
        dict,
    ):

        result = list(
            result.values()
        )

    for rule in result:

        antecedent = None
        consequent = None
        support = None
        confidence = None
        lift = None

        # ----------------------------------------------------
        # Dictionary rule
        # ----------------------------------------------------

        if isinstance(
            rule,
            dict,
        ):

            antecedent = (
                rule.get("antecedent")
                or rule.get("Antecedent")
                or rule.get("lhs")
                or rule.get("left")
                or rule.get("left_hand_side")
            )

            consequent = (
                rule.get("consequent")
                or rule.get("Consequent")
                or rule.get("rhs")
                or rule.get("right")
                or rule.get("right_hand_side")
            )

            support = (
                rule.get("support")
                or rule.get("Support")
            )

            confidence = (
                rule.get("confidence")
                or rule.get("Confidence")
            )

            lift = (
                rule.get("lift")
                or rule.get("Lift")
            )

        # ----------------------------------------------------
        # Tuple/list rule
        # ----------------------------------------------------

        elif isinstance(
            rule,
            (tuple, list),
        ):

            if len(rule) >= 5:

                antecedent = rule[0]
                consequent = rule[1]
                support = rule[2]
                confidence = rule[3]
                lift = rule[4]

            elif len(rule) == 4:

                antecedent = rule[0]
                consequent = rule[1]
                support = rule[2]
                confidence = rule[3]

            elif len(rule) == 3:

                antecedent = rule[0]
                consequent = rule[1]
                confidence = rule[2]

        if antecedent is None:
            continue

        if consequent is None:
            continue

        antecedent = normalize_itemset(
            antecedent
        )

        consequent = normalize_itemset(
            consequent
        )

        if not antecedent or not consequent:
            continue

        if support is None:

            support = calculate_support(
                set(antecedent)
                | set(consequent),
                transactions,
            )

        if confidence is None:

            confidence = calculate_confidence(
                antecedent,
                consequent,
                transactions,
            )

        if lift is None:

            lift = calculate_lift(
                antecedent,
                consequent,
                transactions,
            )

        try:

            support = float(
                support
            )

        except (
            ValueError,
            TypeError,
        ):

            support = calculate_support(
                set(antecedent)
                | set(consequent),
                transactions,
            )

        try:

            confidence = float(
                confidence
            )

        except (
            ValueError,
            TypeError,
        ):

            confidence = calculate_confidence(
                antecedent,
                consequent,
                transactions,
            )

        try:

            lift = float(
                lift
            )

        except (
            ValueError,
            TypeError,
        ):

            lift = calculate_lift(
                antecedent,
                consequent,
                transactions,
            )

        normalized.append({

            "antecedent":
                antecedent,

            "consequent":
                consequent,

            "support":
                support,

            "confidence":
                confidence,

            "lift":
                lift,

        })

    return normalized


# ============================================================
# RESULT PREPARATION
# ============================================================

def prepare_itemsets(
    frequent_itemsets,
):

    itemsets = []

    for items, support in (
        frequent_itemsets.items()
    ):

        items = normalize_itemset(
            items
        )

        try:

            support = float(
                support
            )

        except (
            ValueError,
            TypeError,
        ):

            support = 0.0

        itemsets.append({

            "items":
                items,

            "size":
                len(items),

            "support":
                support,

        })

    # Highest support first,
    # then smaller itemsets first.

    itemsets.sort(
        key=lambda x: (
            -x["support"],
            x["size"],
            x["items"],
        )
    )

    return itemsets


def prepare_rules(
    rules,
    minimum_confidence,
):

    cleaned = []

    for rule in rules:

        if (
            rule["confidence"]
            < minimum_confidence
        ):

            continue

        cleaned.append(rule)

    cleaned.sort(
        key=lambda x: (
            -x["lift"],
            -x["confidence"],
            -x["support"],
        )
    )

    return cleaned


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "dashboard.html",
        dataset_name=analysis_state[
            "dataset_name"
        ],
        metrics=analysis_state[
            "metrics"
        ],
    )


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.route(
    "/upload",
    methods=["POST"],
)
def upload():

    file = request.files.get(
        "dataset"
    )

    if file is None:

        flash(
            "Please select a dataset first.",
            "error",
        )

        return redirect(
            url_for("index")
        )

    if not file.filename:

        flash(
            "Please select a dataset first.",
            "error",
        )

        return redirect(
            url_for("index")
        )

    if not allowed_file(
        file.filename
    ):

        flash(
            "Only CSV, XLSX and XLS files are supported.",
            "error",
        )

        return redirect(
            url_for("index")
        )

    upload_directory = (
        PROJECT_ROOT
        / "webapp"
        / "uploads"
    )

    upload_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = Path(
        file.filename
    ).name

    file_path = (
        upload_directory
        / safe_filename
    )

    try:

        file.save(
            file_path
        )

        (
            transactions,
            product_names,
        ) = load_dataset(
            file_path
        )

        if not transactions:

            raise ValueError(
                "No valid transactions were found in the dataset."
            )

        metrics = (
            calculate_dataset_metrics(
                transactions
            )
        )

        # ----------------------------------------------------
        # Save dataset state
        # ----------------------------------------------------

        analysis_state[
            "dataset_name"
        ] = safe_filename

        analysis_state[
            "transactions"
        ] = transactions

        analysis_state[
            "product_names"
        ] = product_names

        analysis_state[
            "metrics"
        ] = metrics

        # Clear old mining results

        analysis_state[
            "frequent_itemsets"
        ] = {}

        analysis_state[
            "itemsets"
        ] = []

        analysis_state[
            "rules"
        ] = []

        analysis_state[
            "runtime"
        ] = 0.0

        analysis_state[
            "analysis_complete"
        ] = False

        flash(
            f"{safe_filename} uploaded successfully. "
            f"{metrics['transactions']:,} transactions "
            f"and {metrics['products']:,} products detected.",
            "success",
        )

    except Exception as exc:

        print(
            "[UPLOAD ERROR]",
            exc,
        )

        flash(
            f"Could not process the dataset: {exc}",
            "error",
        )

    return redirect(
        url_for("index")
    )


# ============================================================
# ANALYZE
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"],
)
def analyze():

    # ========================================================
    # CHECK DATASET
    # ========================================================

    transactions = analysis_state[
        "transactions"
    ]

    if not transactions:

        flash(
            "Please upload a dataset before running Apriori.",
            "error",
        )

        return redirect(
            url_for("index")
        )

    # ========================================================
    # READ PARAMETERS
    # ========================================================

    try:

        support = float(
            request.form.get(
                "support",
                "0.02",
            )
        )

    except (
        ValueError,
        TypeError,
    ):

        support = 0.02


    try:

        confidence = float(
            request.form.get(
                "confidence",
                "0.60",
            )
        )

    except (
        ValueError,
        TypeError,
    ):

        confidence = 0.60


    # ========================================================
    # VALIDATE SUPPORT
    # ========================================================

    if not (
        0.001
        <= support
        <= 1.0
    ):

        flash(
            "Minimum support must be between "
            "0.1% and 100%.",
            "error",
        )

        return redirect(
            url_for("index")
        )


    # ========================================================
    # VALIDATE CONFIDENCE
    # ========================================================

    if not (
        0.10
        <= confidence
        <= 1.0
    ):

        flash(
            "Minimum confidence must be between "
            "10% and 100%.",
            "error",
        )

        return redirect(
            url_for("index")
        )


    # ========================================================
    # CURRENT LIFT
    # ========================================================

    # Lift is calculated after rule generation.

    minimum_lift = 0.0


    # ========================================================
    # SAVE PARAMETERS
    # ========================================================

    analysis_state[
        "support"
    ] = support

    analysis_state[
        "confidence"
    ] = confidence

    analysis_state[
        "lift"
    ] = minimum_lift


    # ========================================================
    # RUN APRIORI
    # ========================================================

    print()
    print(
        "=" * 60
    )
    print(
        "APRiori ANALYSIS"
    )
    print(
        "=" * 60
    )
    print(
        f"Dataset    : "
        f"{analysis_state['dataset_name']}"
    )
    print(
        f"Transactions: "
        f"{len(transactions):,}"
    )
    print(
        f"Support    : "
        f"{support:.4f}"
    )
    print(
        f"Confidence : "
        f"{confidence:.4f}"
    )
    print(
        "=" * 60
    )


    start_time = time.perf_counter()


    try:

        # ----------------------------------------------------
        # STEP 1
        # Frequent Itemset Mining
        # ----------------------------------------------------

        raw_itemsets = run_apriori(
            transactions,
            support,
        )


        # ----------------------------------------------------
        # Normalize Apriori output
        # ----------------------------------------------------

        frequent_itemsets = (
            normalize_frequent_itemsets(
                raw_itemsets,
                transactions,
            )
        )


        # ----------------------------------------------------
        # Explicit support filtering
        #
        # This protects the UI from implementations that
        # return additional intermediate candidates.
        # ----------------------------------------------------

        frequent_itemsets = {

            itemset: itemset_support

            for itemset, itemset_support
            in frequent_itemsets.items()

            if itemset_support
            >= support

        }


        # ----------------------------------------------------
        # STEP 2
        # Generate association rules
        # ----------------------------------------------------

        raw_rules = call_custom_rules(
            frequent_itemsets,
            transactions,
            confidence,
        )


        # ----------------------------------------------------
        # If project's rule generator isn't compatible,
        # use our validated fallback.
        # ----------------------------------------------------

        if raw_rules is None:

            raw_rules = (
                generate_rules_fallback(
                    frequent_itemsets,
                    transactions,
                    confidence,
                )
            )


        # ----------------------------------------------------
        # Normalize rules
        # ----------------------------------------------------

        normalized_rules = normalize_rules(
            raw_rules,
            transactions,
        )


        # ----------------------------------------------------
        # Apply confidence threshold
        # ----------------------------------------------------

        normalized_rules = (
            prepare_rules(
                normalized_rules,
                confidence,
            )
        )


        # ----------------------------------------------------
        # Calculate runtime
        # ----------------------------------------------------

        runtime = (
            time.perf_counter()
            - start_time
        )


        # ----------------------------------------------------
        # Prepare UI structures
        # ----------------------------------------------------

        itemsets = (
            prepare_itemsets(
                frequent_itemsets
            )
        )


        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        analysis_state[
            "frequent_itemsets"
        ] = frequent_itemsets

        analysis_state[
            "itemsets"
        ] = itemsets

        analysis_state[
            "rules"
        ] = normalized_rules

        analysis_state[
            "runtime"
        ] = runtime

        analysis_state[
            "analysis_complete"
        ] = True


        print(
            f"Frequent itemsets: "
            f"{len(itemsets):,}"
        )

        print(
            f"Association rules: "
            f"{len(normalized_rules):,}"
        )

        print(
            f"Runtime: "
            f"{runtime:.4f} seconds"
        )

        print(
            "=" * 60
        )


        # ====================================================
        # REDIRECT TO RESULTS
        # ====================================================

        return redirect(
            url_for("results")
        )


    except Exception as exc:

        print()
        print(
            "[ANALYSIS ERROR]"
        )
        print(
            repr(exc)
        )
        print()

        flash(
            f"Apriori analysis failed: {exc}",
            "error",
        )

        analysis_state[
            "analysis_complete"
        ] = False

        return redirect(
            url_for("index")
        )


# ============================================================
# RESULTS
# ============================================================

@app.route("/results")
def results():

    if not analysis_state[
        "analysis_complete"
    ]:

        flash(
            "Run an Apriori analysis before viewing results.",
            "error",
        )

        return redirect(
            url_for("index")
        )


    analysis = {

        "dataset_name":
            analysis_state[
                "dataset_name"
            ],

        "support":
            analysis_state[
                "support"
            ],

        "confidence":
            analysis_state[
                "confidence"
            ],

        "lift":
            analysis_state[
                "lift"
            ],

        "runtime":
            analysis_state[
                "runtime"
            ],

        "metrics":
            analysis_state[
                "metrics"
            ],

    }


    return render_template(

        "results.html",

        analysis=analysis,

        itemsets=analysis_state[
            "itemsets"
        ],

        rules=analysis_state[
            "rules"
        ],

    )


# ============================================================
# DATASET PAGE
# ============================================================

@app.route("/dataset")
def dataset():

    return render_template(

        "dashboard.html",

        dataset_name=analysis_state[
            "dataset_name"
        ],

        metrics=analysis_state[
            "metrics"
        ],

    )


# ============================================================
# LEARN PAGE
# ============================================================

@app.route("/learn")
def learn():

    return render_template(
        "learn.html"
    )


# ============================================================
# BENCHMARKS PAGE
# ============================================================

@app.route("/benchmarks")
def benchmarks():

    benchmark_file = (
        PROJECT_ROOT
        / "benchmarks"
        / "real_world_benchmark.csv"
    )

    benchmark_rows = []

    if benchmark_file.exists():

        try:

            with open(
                benchmark_file,
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:

                reader = csv.DictReader(
                    file
                )

                benchmark_rows = list(
                    reader
                )

        except Exception as exc:

            print(
                "[BENCHMARK ERROR]",
                exc,
            )

    return render_template(

        "benchmarks.html",

        benchmark_rows=benchmark_rows,

    )


# ============================================================
# RESET
# ============================================================

@app.route(
    "/reset",
    methods=[
        "GET",
        "POST",
    ],
)
def reset():

    reset_analysis()

    flash(
        "Analysis state has been reset.",
        "success",
    )

    return redirect(
        url_for("index")
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "ok",

        "project":
            "Scalable Association Rule Mining",

        "ui":
            "Flask",

        "current_algorithm":
            "Apriori",

        "future_algorithms":
            [
                "FP-Growth",
                "Eclat",
            ],

        "apriori_available":
            APRIORI_AVAILABLE,

        "rules_module_available":
            RULES_AVAILABLE,

        "dataset_loaded":
            bool(
                analysis_state[
                    "transactions"
                ]
            ),

        "analysis_complete":
            analysis_state[
                "analysis_complete"
            ],

    })


# ============================================================
# API - CURRENT STATE
# ============================================================

@app.route("/api/state")
def api_state():

    metrics = (
        analysis_state[
            "metrics"
        ]
        or {}
    )

    return jsonify({

        "dataset":
            analysis_state[
                "dataset_name"
            ],

        "metrics":
            metrics,

        "support":
            analysis_state[
                "support"
            ],

        "confidence":
            analysis_state[
                "confidence"
            ],

        "lift":
            analysis_state[
                "lift"
            ],

        "frequent_itemsets":
            len(
                analysis_state[
                    "itemsets"
                ]
            ),

        "rules":
            len(
                analysis_state[
                    "rules"
                ]
            ),

        "runtime":
            analysis_state[
                "runtime"
            ],

        "analysis_complete":
            analysis_state[
                "analysis_complete"
            ],

    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "base.html",
    ), 404


@app.errorhandler(500)
def internal_server_error(error):

    print(
        "[500 ERROR]",
        error,
    )

    flash(
        "An unexpected server error occurred.",
        "error",
    )

    return redirect(
        url_for("index")
    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 60
    )
    print(
        "SCALABLE ASSOCIATION RULE MINING"
    )
    print(
        "=" * 60
    )
    print(
        f"Algorithm : Apriori"
    )
    print(
        f"UI        : Flask"
    )
    print(
        f"Project   : {PROJECT_ROOT}"
    )
    print(
        f"URL       : http://127.0.0.1:5000"
    )
    print(
        "=" * 60
    )
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
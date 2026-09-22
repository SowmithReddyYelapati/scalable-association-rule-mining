import io

import pytest

from webapp.app import (
    app,
    analysis_state,
)


# ============================================================
# TEST FIXTURE
# ============================================================

@pytest.fixture
def client():

    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


# ============================================================
# RESET GLOBAL STATE BEFORE / AFTER EVERY TEST
# ============================================================

@pytest.fixture(autouse=True)
def reset_analysis_state():

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

    yield

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


# ============================================================
# BASIC ROUTES
# ============================================================

def test_home_page_loads(client):

    response = client.get("/")

    assert response.status_code == 200


def test_learn_page_loads(client):

    response = client.get("/learn")

    assert response.status_code == 200


def test_benchmarks_page_loads(client):

    response = client.get("/benchmarks")

    assert response.status_code == 200


# ============================================================
# HEALTH ENDPOINT
# ============================================================

def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "ok"

    assert (
        data["current_algorithm"]
        == "Apriori"
    )

    assert (
        "Apriori"
        in data["available_algorithms"]
    )

    assert (
        "FP-Growth"
        in data["available_algorithms"]
    )

    assert (
        "Eclat"
        in data["future_algorithms"]
    )


def test_available_algorithms(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data[
        "available_algorithms"
    ] == [
        "Apriori",
        "FP-Growth",
    ]


def test_future_algorithms_are_visible(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert (
        "Apriori"
        in data["available_algorithms"]
    )

    assert (
        "FP-Growth"
        in data["available_algorithms"]
    )

    assert (
        "Eclat"
        in data["future_algorithms"]
    )


def test_eclat_is_not_available_yet(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert (
        "Eclat"
        not in data["available_algorithms"]
    )

    assert (
        "Eclat"
        in data["future_algorithms"]
    )


# ============================================================
# UPLOAD VALIDATION
# ============================================================

def test_upload_without_file(client):

    response = client.post(
        "/upload",
        data={},
    )

    assert response.status_code == 302


def test_upload_invalid_file_type(client):

    data = {
        "dataset": (
            io.BytesIO(
                b"invalid file"
            ),
            "test.txt",
        )
    }

    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 302


# ============================================================
# ANALYZE WITHOUT DATASET
# ============================================================

def test_analyze_without_dataset(client):

    response = client.post(
        "/analyze",
        data={
            "algorithm": "apriori",
            "support": "0.02",
            "confidence": "0.60",
            "lift": "",
        },
    )

    assert response.status_code == 302

    assert (
        response.headers["Location"]
        .endswith("/")
    )


def test_analyze_without_dataset_fp_growth(client):

    response = client.post(
        "/analyze",
        data={
            "algorithm": "fp-growth",
            "support": "0.02",
            "confidence": "0.60",
            "lift": "",
        },
    )

    assert response.status_code == 302

    assert (
        response.headers["Location"]
        .endswith("/")
    )


# ============================================================
# RESULTS WITHOUT ANALYSIS
# ============================================================

def test_results_without_analysis(client):

    response = client.get(
        "/results"
    )

    assert response.status_code == 302

    assert (
        response.headers["Location"]
        .endswith("/")
    )


# ============================================================
# DOWNLOAD WITHOUT RULES
# ============================================================

def test_download_without_rules(client):

    response = client.get(
        "/download-csv"
    )

    assert response.status_code == 302

    assert (
        response.headers["Location"]
        .endswith("/")
    )


# ============================================================
# RESET
# ============================================================

def test_reset_endpoint(client):

    analysis_state[
        "dataset_name"
    ] = "test.csv"

    analysis_state[
        "algorithm"
    ] = "FP-Growth"

    analysis_state[
        "transactions"
    ] = [
        frozenset(
            {"A", "B"}
        )
    ]

    analysis_state[
        "frequent_itemsets"
    ] = {
        frozenset(
            {"A"}
        ): 1.0
    }

    analysis_state[
        "rules"
    ] = []

    response = client.post(
        "/reset"
    )

    assert response.status_code == 302

    assert (
        analysis_state[
            "dataset_name"
        ]
        is None
    )

    assert (
        analysis_state[
            "transactions"
        ]
        == []
    )

    assert (
        analysis_state[
            "frequent_itemsets"
        ]
        == {}
    )

    assert (
        analysis_state[
            "rules"
        ]
        == []
    )

    assert (
        analysis_state[
            "algorithm"
        ]
        == "Apriori"
    )


# ============================================================
# DASHBOARD WITH DATASET
# ============================================================

def test_dashboard_with_dataset(client):

    analysis_state.update(
        {
            "dataset_name":
                "test.csv",

            "transactions": [
                frozenset(
                    {"A", "B"}
                ),

                frozenset(
                    {"A", "C"}
                ),
            ],

            "product_names": {},

            "frequent_itemsets": {},

            "rules": [],

            "metrics": {
                "rows": 2,
                "transactions": 2,
                "products": 3,
                "average_items": 2.0,
                "largest_transaction": 2,
                "columns": 2,
            },
        }
    )

    response = client.get("/")

    assert response.status_code == 200


# ============================================================
# RESULTS PAGE WITH ITEMSETS
# ============================================================

def test_results_page_with_itemsets(client):

    analysis_state.update(
        {
            "dataset_name":
                "test.csv",

            "transactions": [
                frozenset(
                    {"A", "B"}
                ),

                frozenset(
                    {"A", "C"}
                ),
            ],

            "product_names": {},

            "frequent_itemsets": {
                frozenset(
                    {"A"}
                ): 1.0,

                frozenset(
                    {"B"}
                ): 0.5,

                frozenset(
                    {"C"}
                ): 0.5,
            },

            "rules": [],

            "support": 0.5,

            "confidence": 0.6,

            "lift": None,

            "runtime": 0.01,

            "algorithm": "FP-Growth",

            "metrics": {
                "rows": 2,
                "transactions": 2,
                "products": 3,
                "average_items": 2.0,
                "largest_transaction": 2,
                "columns": 2,
            },
        }
    )

    response = client.get(
        "/results"
    )

    assert response.status_code == 200


# ============================================================
# RESULTS PAGE WITH RULES
# ============================================================

def test_results_page_with_rules(client):

    class DummyRule:

        antecedent = frozenset(
            {"A"}
        )

        consequent = frozenset(
            {"B"}
        )

        support = 0.5

        confidence = 0.8

        lift = 1.6

    analysis_state.update(
        {
            "dataset_name":
                "test.csv",

            "transactions": [
                frozenset(
                    {"A", "B"}
                ),

                frozenset(
                    {"A"}
                ),
            ],

            "product_names": {},

            "frequent_itemsets": {
                frozenset(
                    {"A"}
                ): 1.0,

                frozenset(
                    {"B"}
                ): 0.5,

                frozenset(
                    {"A", "B"}
                ): 0.5,
            },

            "rules": [
                DummyRule()
            ],

            "support": 0.5,

            "confidence": 0.6,

            "lift": None,

            "runtime": 0.01,

            "algorithm": "FP-Growth",

            "metrics": {
                "rows": 2,
                "transactions": 2,
                "products": 2,
                "average_items": 1.5,
                "largest_transaction": 2,
                "columns": 2,
            },
        }
    )

    response = client.get(
        "/results"
    )

    assert response.status_code == 200


# ============================================================
# DOWNLOAD CSV WITH RULES
# ============================================================

def test_download_csv_with_rules(client):

    class DummyRule:

        antecedent = frozenset(
            {"A"}
        )

        consequent = frozenset(
            {"B"}
        )

        support = 0.5

        confidence = 0.8

        lift = 1.6

    analysis_state.update(
        {
            "frequent_itemsets": {
                frozenset(
                    {"A"}
                ): 1.0,

                frozenset(
                    {"B"}
                ): 0.5,

                frozenset(
                    {"A", "B"}
                ): 0.5,
            },

            "rules": [
                DummyRule()
            ],

            "product_names": {},

            "lift": None,
        }
    )

    response = client.get(
        "/download-csv"
    )

    assert response.status_code == 200

    assert (
        response.mimetype
        == "text/csv"
    )

    assert (
        b"Antecedent"
        in response.data
    )

    assert (
        b"Consequent"
        in response.data
    )

    assert (
        b"Confidence"
        in response.data
    )

    assert (
        b"Lift"
        in response.data
    )


# ============================================================
# FP-GROWTH RESULT STATE
# ============================================================

def test_fp_growth_result_state(client):

    analysis_state.update(
        {
            "algorithm":
                "FP-Growth",

            "frequent_itemsets": {
                frozenset(
                    {"A"}
                ): 0.8
            },

            "rules": [],
        }
    )

    assert (
        analysis_state[
            "algorithm"
        ]
        == "FP-Growth"
    )

    assert (
        frozenset(
            {"A"}
        )
        in analysis_state[
            "frequent_itemsets"
        ]
    )


# ============================================================
# APRIORI RESULT STATE
# ============================================================

def test_apriori_result_state(client):

    analysis_state.update(
        {
            "algorithm":
                "Apriori",

            "frequent_itemsets": {
                frozenset(
                    {"A"}
                ): 0.8
            },

            "rules": [],
        }
    )

    assert (
        analysis_state[
            "algorithm"
        ]
        == "Apriori"
    )


# ============================================================
# INITIAL STATE VALIDATION
# ============================================================

def test_initial_analysis_state():

    assert (
        analysis_state[
            "algorithm"
        ]
        == "Apriori"
    )

    assert (
        analysis_state[
            "transactions"
        ]
        == []
    )

    assert (
        analysis_state[
            "frequent_itemsets"
        ]
        == {}
    )

    assert (
        analysis_state[
            "rules"
        ]
        == []
    )


# ============================================================
# METRICS VALIDATION
# ============================================================

def test_dashboard_metrics_are_complete():

    analysis_state["metrics"] = {
        "rows": 10,
        "transactions": 5,
        "products": 4,
        "average_items": 2.0,
        "largest_transaction": 4,
        "columns": 3,
    }

    metrics = analysis_state["metrics"]

    assert metrics["rows"] == 10

    assert metrics["transactions"] == 5

    assert metrics["products"] == 4

    assert metrics["average_items"] == 2.0

    assert metrics["largest_transaction"] == 4

    assert metrics["columns"] == 3


# ============================================================
# ALGORITHM SELECTION VALIDATION
# ============================================================

def test_default_algorithm_is_apriori():

    assert (
        analysis_state[
            "algorithm"
        ]
        == "Apriori"
    )


def test_fp_growth_can_be_selected():

    analysis_state[
        "algorithm"
    ] = "FP-Growth"

    assert (
        analysis_state[
            "algorithm"
        ]
        == "FP-Growth"
    )


def test_analyze_dispatch_fp_growth(client):
    analysis_state.update({
        "dataset_name": "test.csv",
        "transactions": [
            frozenset({"Milk", "Bread"}),
            frozenset({"Milk", "Bread", "Butter"}),
            frozenset({"Bread", "Butter"}),
        ],
        "product_names": {"Milk": "Milk", "Bread": "Bread", "Butter": "Butter"},
    })

    response = client.post(
        "/analyze",
        data={
            "algorithm": "fp_growth",
            "support": "0.02",
            "confidence": "0.60",
            "lift": "",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert analysis_state["algorithm"] == "FP-Growth"
    html = response.get_data(as_text=True)
    assert "FP-Growth" in html


def test_analyze_dispatch_apriori(client):
    analysis_state.update({
        "dataset_name": "test.csv",
        "transactions": [
            frozenset({"Milk", "Bread"}),
            frozenset({"Milk", "Bread", "Butter"}),
            frozenset({"Bread", "Butter"}),
        ],
        "product_names": {"Milk": "Milk", "Bread": "Bread", "Butter": "Butter"},
    })

    response = client.post(
        "/analyze",
        data={
            "algorithm": "apriori",
            "support": "0.02",
            "confidence": "0.60",
            "lift": "",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert analysis_state["algorithm"] == "Apriori"
    html = response.get_data(as_text=True)
    assert "Apriori" in html


def test_analyze_dispatch_fp_growth_legacy_slug(client):
    analysis_state.update({
        "dataset_name": "test.csv",
        "transactions": [
            frozenset({"Milk", "Bread"}),
            frozenset({"Milk", "Bread", "Butter"}),
        ],
        "product_names": {"Milk": "Milk", "Bread": "Bread", "Butter": "Butter"},
    })

    response = client.post(
        "/analyze",
        data={
            "algorithm": "fp-growth",
            "support": "0.02",
            "confidence": "0.60",
            "lift": "",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert analysis_state["algorithm"] == "FP-Growth"


def test_dashboard_retains_fp_growth_selection(client):
    analysis_state.update({
        "dataset_name": "test.csv",
        "transactions": [
            frozenset({"Milk", "Bread"}),
            frozenset({"Milk", "Bread", "Butter"}),
        ],
        "product_names": {"Milk": "Milk", "Bread": "Bread", "Butter": "Butter"},
        "algorithm": "FP-Growth",
    })

    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="algoCardFpGrowth"' in html
    assert 'id="algoCardFpGrowth"\n                data-algo="fp_growth"\n                role="radio"\n                aria-checked="true"' in html or 'data-algo="fp_growth"' in html
    assert 'value="fp_growth"' in html
    assert 'Run FP-Growth Analysis' in html


def test_dashboard_retains_apriori_selection(client):
    analysis_state.update({
        "dataset_name": "test.csv",
        "transactions": [
            frozenset({"Milk", "Bread"}),
            frozenset({"Milk", "Bread", "Butter"}),
        ],
        "product_names": {"Milk": "Milk", "Bread": "Bread", "Butter": "Butter"},
        "algorithm": "Apriori",
    })

    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'value="apriori"' in html
    assert 'Run Apriori Analysis' in html


def test_results_page_refresh_retains_fp_growth(client):
    analysis_state.update({
        "dataset_name": "test.csv",
        "transactions": [frozenset({"Milk", "Bread"})],
        "product_names": {},
        "frequent_itemsets": {frozenset({"Milk"}): 1.0},
        "rules": [],
        "support": 0.02,
        "confidence": 0.60,
        "lift": None,
        "runtime": 0.01,
        "algorithm": "FP-Growth",
    })

    r1 = client.get("/results")
    assert r1.status_code == 200
    assert "FP-Growth" in r1.get_data(as_text=True)

    r2 = client.get("/results")
    assert r2.status_code == 200
    assert "FP-Growth" in r2.get_data(as_text=True)
    assert analysis_state["algorithm"] == "FP-Growth"


# ============================================================
# TRANSACTION STATE VALIDATION
# ============================================================

def test_transactions_are_sets():

    analysis_state[
        "transactions"
    ] = [
        frozenset(
            {"Milk", "Bread"}
        ),

        frozenset(
            {"Bread", "Eggs"}
        ),
    ]

    transactions = (
        analysis_state[
            "transactions"
        ]
    )

    assert len(transactions) == 2

    assert isinstance(
        transactions[0],
        frozenset,
    )

    assert (
        "Milk"
        in transactions[0]
    )

    assert (
        "Bread"
        in transactions[0]
    )


# ============================================================
# PRODUCT NAME MAPPING
# ============================================================

def test_product_name_mapping():

    analysis_state[
        "product_names"
    ] = {
        "P001":
            "Milk",

        "P002":
            "Bread",
    }

    mapping = (
        analysis_state[
            "product_names"
        ]
    )

    assert (
        mapping["P001"]
        == "Milk"
    )

    assert (
        mapping["P002"]
        == "Bread"
    )


# ============================================================
# FREQUENT ITEMSET VALIDATION
# ============================================================

def test_frequent_itemsets_are_stored():

    analysis_state[
        "frequent_itemsets"
    ] = {
        frozenset(
            {"Milk"}
        ): 0.80,

        frozenset(
            {"Bread"}
        ): 0.70,

        frozenset(
            {"Milk", "Bread"}
        ): 0.60,
    }

    itemsets = (
        analysis_state[
            "frequent_itemsets"
        ]
    )

    assert len(itemsets) == 3

    assert (
        itemsets[
            frozenset({"Milk"})
        ]
        == 0.80
    )

    assert (
        itemsets[
            frozenset(
                {"Milk", "Bread"}
            )
        ]
        == 0.60
    )
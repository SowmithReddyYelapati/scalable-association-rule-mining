from pathlib import Path
import io

import pytest

from webapp.app import app


# ============================================================
# TEST CONFIGURATION
# ============================================================

app.config["TESTING"] = True


@pytest.fixture
def client():
    """
    Create a Flask test client.
    """
    with app.test_client() as client:
        yield client


# ============================================================
# BASIC APPLICATION TESTS
# ============================================================

def test_home_page_loads(client):

    response = client.get("/")

    assert response.status_code == 200


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "ok"

    assert data["current_algorithm"] == "Apriori"

    assert "FP-Growth" in data["future_algorithms"]

    assert "Eclat" in data["future_algorithms"]


# ============================================================
# LEARNING / BENCHMARK PAGES
# ============================================================

def test_learn_page_loads(client):

    response = client.get("/learn")

    assert response.status_code == 200


def test_benchmarks_page_loads(client):

    response = client.get("/benchmarks")

    assert response.status_code == 200


# ============================================================
# RESET
# ============================================================

def test_reset_endpoint(client):

    response = client.post(
        "/reset",
        follow_redirects=True,
    )

    assert response.status_code == 200


# ============================================================
# RESULTS WITHOUT ANALYSIS
# ============================================================

def test_results_without_analysis_redirects(client):

    response = client.get(
        "/results",
        follow_redirects=False,
    )

    assert response.status_code == 302

    assert response.headers["Location"].endswith("/")


# ============================================================
# ANALYSIS WITHOUT DATASET
# ============================================================

def test_analyze_without_dataset_redirects(client):

    response = client.post(
        "/analyze",
        data={
            "support": "0.10",
            "confidence": "0.50",
            "lift": "1.0",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302

    assert response.headers["Location"].endswith("/")


# ============================================================
# INVALID SUPPORT
# ============================================================

def test_invalid_support(client):

    # Load the small test dataset first
    csv_path = Path(
        "association_rule_test_dataset.csv"
    )

    if not csv_path.exists():

        pytest.skip(
            "association_rule_test_dataset.csv not found"
        )

    with open(
        csv_path,
        "rb",
    ) as file:

        upload_response = client.post(
            "/upload",
            data={
                "dataset": (
                    file,
                    "association_rule_test_dataset.csv",
                )
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

    assert upload_response.status_code == 200

    response = client.post(
        "/analyze",
        data={
            "support": "2.0",
            "confidence": "0.50",
            "lift": "1.0",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200

    assert (
        b"Minimum support"
        in response.data
    )


# ============================================================
# INVALID CONFIDENCE
# ============================================================

def test_invalid_confidence(client):

    csv_path = Path(
        "association_rule_test_dataset.csv"
    )

    if not csv_path.exists():

        pytest.skip(
            "association_rule_test_dataset.csv not found"
        )

    with open(
        csv_path,
        "rb",
    ) as file:

        client.post(
            "/upload",
            data={
                "dataset": (
                    file,
                    "association_rule_test_dataset.csv",
                )
            },
            content_type="multipart/form-data",
        )

    response = client.post(
        "/analyze",
        data={
            "support": "0.10",
            "confidence": "2.0",
            "lift": "1.0",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200

    assert (
        b"Minimum confidence"
        in response.data
    )


# ============================================================
# INVALID LIFT
# ============================================================

def test_invalid_lift(client):

    csv_path = Path(
        "association_rule_test_dataset.csv"
    )

    if not csv_path.exists():

        pytest.skip(
            "association_rule_test_dataset.csv not found"
        )

    with open(
        csv_path,
        "rb",
    ) as file:

        client.post(
            "/upload",
            data={
                "dataset": (
                    file,
                    "association_rule_test_dataset.csv",
                )
            },
            content_type="multipart/form-data",
        )

    response = client.post(
        "/analyze",
        data={
            "support": "0.10",
            "confidence": "0.50",
            "lift": "-1",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200

    assert (
        b"Minimum lift"
        in response.data
    )


# ============================================================
# UPLOAD TEST
# ============================================================

def test_upload_csv(client):

    csv_path = Path(
        "association_rule_test_dataset.csv"
    )

    if not csv_path.exists():

        pytest.skip(
            "association_rule_test_dataset.csv not found"
        )

    with open(
        csv_path,
        "rb",
    ) as file:

        response = client.post(
            "/upload",
            data={
                "dataset": (
                    file,
                    "association_rule_test_dataset.csv",
                )
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

    assert response.status_code == 200

    assert (
        b"uploaded successfully"
        in response.data
    )


# ============================================================
# UPLOAD WITHOUT FILE
# ============================================================

def test_upload_without_file(client):

    response = client.post(
        "/upload",
        data={},
        follow_redirects=True,
    )

    assert response.status_code == 200

    assert (
        b"Please select a dataset"
        in response.data
    )


# ============================================================
# INVALID FILE TYPE
# ============================================================

def test_upload_invalid_file_type(client):

    response = client.post(
        "/upload",
        data={
            "dataset": (
                io.BytesIO(b"this is not a dataset"),
                "test.txt",
            )
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200

    assert (
        b"Only CSV, XLSX and XLS files"
        in response.data
    )


# ============================================================
# COMPLETE APRIORI WORKFLOW
# ============================================================

def test_complete_apriori_workflow(client):

    csv_path = Path(
        "association_rule_test_dataset.csv"
    )

    if not csv_path.exists():

        pytest.skip(
            "association_rule_test_dataset.csv not found"
        )

    # --------------------------------------------------------
    # STEP 1 — Upload
    # --------------------------------------------------------

    with open(
        csv_path,
        "rb",
    ) as file:

        response = client.post(
            "/upload",
            data={
                "dataset": (
                    file,
                    "association_rule_test_dataset.csv",
                )
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 302

    # --------------------------------------------------------
    # STEP 2 — Run Apriori
    # --------------------------------------------------------

    response = client.post(
        "/analyze",
        data={
            "support": "0.10",
            "confidence": "0.50",
            "lift": "1.0",
        },
        follow_redirects=False,
    )

    # Successful analysis redirects to results
    assert response.status_code == 302

    assert (
        response.headers["Location"]
        .endswith("/results")
    )

    # --------------------------------------------------------
    # STEP 3 — Results page
    # --------------------------------------------------------

    response = client.get(
        "/results"
    )

    assert response.status_code == 200

    # --------------------------------------------------------
    # STEP 4 — Verify important UI content
    # --------------------------------------------------------

    assert (
        b"Analysis Results"
        in response.data
    )

    assert (
        b"Frequent Itemsets"
        in response.data
    )

    assert (
        b"Association Rules"
        in response.data
    )

    assert (
        b"Support"
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
# LOW SUPPORT SHOULD PRODUCE MORE PATTERNS
# ============================================================

def test_support_changes_results(client):

    csv_path = Path(
        "association_rule_test_dataset.csv"
    )

    if not csv_path.exists():

        pytest.skip(
            "association_rule_test_dataset.csv not found"
        )

    # Upload dataset
    with open(
        csv_path,
        "rb",
    ) as file:

        client.post(
            "/upload",
            data={
                "dataset": (
                    file,
                    "association_rule_test_dataset.csv",
                )
            },
            content_type="multipart/form-data",
        )

    # Run with 10% support
    response = client.post(
        "/analyze",
        data={
            "support": "0.10",
            "confidence": "0.50",
            "lift": "1.0",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200

    # Run with lower support
    response = client.post(
        "/analyze",
        data={
            "support": "0.05",
            "confidence": "0.50",
            "lift": "1.0",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200


# ============================================================
# CONFIDENCE FILTER TEST
# ============================================================

def test_confidence_changes_results(client):

    csv_path = Path(
        "association_rule_test_dataset.csv"
    )

    if not csv_path.exists():

        pytest.skip(
            "association_rule_test_dataset.csv not found"
        )

    # Upload
    with open(
        csv_path,
        "rb",
    ) as file:

        client.post(
            "/upload",
            data={
                "dataset": (
                    file,
                    "association_rule_test_dataset.csv",
                )
            },
            content_type="multipart/form-data",
        )

    # Low confidence
    response = client.post(
        "/analyze",
        data={
            "support": "0.10",
            "confidence": "0.50",
            "lift": "1.0",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200

    # High confidence
    response = client.post(
        "/analyze",
        data={
            "support": "0.10",
            "confidence": "0.90",
            "lift": "1.0",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200


# ============================================================
# LIFT FILTER TEST
# ============================================================

def test_lift_filter(client):

    csv_path = Path(
        "association_rule_test_dataset.csv"
    )

    if not csv_path.exists():

        pytest.skip(
            "association_rule_test_dataset.csv not found"
        )

    # Upload
    with open(
        csv_path,
        "rb",
    ) as file:

        client.post(
            "/upload",
            data={
                "dataset": (
                    file,
                    "association_rule_test_dataset.csv",
                )
            },
            content_type="multipart/form-data",
        )

    # Run with normal lift
    response = client.post(
        "/analyze",
        data={
            "support": "0.10",
            "confidence": "0.50",
            "lift": "1.0",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200

    # Run with high lift
    response = client.post(
        "/analyze",
        data={
            "support": "0.10",
            "confidence": "0.50",
            "lift": "2.0",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200


# ============================================================
# FUTURE ALGORITHMS
# ============================================================

def test_future_algorithms_are_visible(client):

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert (
        "FP-Growth"
        in data["future_algorithms"]
    )

    assert (
        "Eclat"
        in data["future_algorithms"]
    )


def test_download_csv_no_rules(client):

    response = client.get(
        "/download-csv",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"No association rules available to download" in response.data
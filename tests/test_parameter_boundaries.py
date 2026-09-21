import pytest
from webapp.app import app, analysis_state
from src.arm.apriori import apriori
from src.arm.fp_growth import fp_growth
from src.arm.rules import generate_rules


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_state():
    transactions = [
        {"apple", "banana", "cherry"},
        {"apple", "banana"},
        {"apple", "cherry"},
        {"banana", "date"},
        {"apple", "banana", "cherry", "date"},
        {"apple", "date"},
        {"banana", "cherry"},
        {"apple", "banana"},
        {"apple", "banana", "cherry"},
        {"banana", "cherry", "date"},
    ]
    product_names = {
        "apple": "Apple",
        "banana": "Banana",
        "cherry": "Cherry",
        "date": "Date",
    }
    analysis_state.update({
        "dataset_name": "boundary_test.csv",
        "transactions": transactions,
        "product_names": product_names,
        "frequent_itemsets": {},
        "rules": [],
    })
    yield transactions
    analysis_state.clear()


# ============================================================
# VALID SUPPORT BOUNDARIES (0.1% to 10%)
# ============================================================

@pytest.mark.parametrize("valid_support", [0.001, 0.01, 0.02, 0.05, 0.10])
def test_valid_support_boundaries_apriori(sample_state, valid_support):
    """Verify Apriori accepts valid support values between 0.1% and 10%."""
    result = apriori(sample_state, min_support=valid_support)
    assert isinstance(result, dict)
    assert len(result) > 0
    for itemset, supp in result.items():
        assert supp >= valid_support


@pytest.mark.parametrize("valid_support", [0.001, 0.01, 0.02, 0.05, 0.10])
def test_valid_support_boundaries_fp_growth(sample_state, valid_support):
    """Verify FP-Growth accepts valid support values between 0.1% and 10%."""
    result = fp_growth(sample_state, min_support=valid_support)
    assert isinstance(result, dict)
    assert len(result) > 0
    for itemset, supp in result.items():
        assert supp >= valid_support


@pytest.mark.parametrize("valid_support", [0.001, 0.01, 0.02, 0.05, 0.10])
def test_valid_support_boundaries_app(client, sample_state, valid_support):
    """Verify Flask endpoint accepts valid support values."""
    response = client.post(
        "/analyze",
        data={
            "algorithm": "apriori",
            "support": str(valid_support),
            "confidence": "0.60",
        },
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/results")
    assert analysis_state["support"] == valid_support


# ============================================================
# VALID CONFIDENCE BOUNDARIES (10% to 100%)
# ============================================================

@pytest.mark.parametrize("valid_confidence", [0.10, 0.50, 0.60, 0.90, 1.00])
def test_valid_confidence_boundaries_rules(sample_state, valid_confidence):
    """Verify rule generation accepts valid confidence values between 10% and 100%."""
    frequent_itemsets = apriori(sample_state, min_support=0.10)
    rules = generate_rules(frequent_itemsets, min_confidence=valid_confidence)
    assert isinstance(rules, list)
    for rule in rules:
        assert rule.confidence >= valid_confidence


@pytest.mark.parametrize("valid_confidence", [0.10, 0.50, 0.60, 0.90, 1.00])
def test_valid_confidence_boundaries_app(client, sample_state, valid_confidence):
    """Verify Flask endpoint accepts valid confidence values."""
    response = client.post(
        "/analyze",
        data={
            "algorithm": "apriori",
            "support": "0.05",
            "confidence": str(valid_confidence),
        },
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/results")
    assert analysis_state["confidence"] == valid_confidence


# ============================================================
# INVALID SUPPORT BOUNDARIES (<0.1% or >10%)
# ============================================================

@pytest.mark.parametrize("invalid_support", [0.0, 0.101, 0.15, -0.01, -0.10])
def test_invalid_support_rejected_by_app(client, sample_state, invalid_support):
    """Verify Flask endpoint rejects support values outside 0.1% - 10% range."""
    response = client.post(
        "/analyze",
        data={
            "algorithm": "apriori",
            "support": str(invalid_support),
            "confidence": "0.60",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    # Should redirect back to dashboard due to validation error
    assert analysis_state.get("support") != invalid_support


# ============================================================
# INVALID CONFIDENCE BOUNDARIES (<10% or >100%)
# ============================================================

@pytest.mark.parametrize("invalid_confidence", [0.0, 0.09, 1.001, 1.10, -0.05])
def test_invalid_confidence_rejected_by_app(client, sample_state, invalid_confidence):
    """Verify Flask endpoint rejects confidence values outside 10% - 100% range."""
    response = client.post(
        "/analyze",
        data={
            "algorithm": "apriori",
            "support": "0.05",
            "confidence": str(invalid_confidence),
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    # Should redirect back to dashboard due to validation error
    assert analysis_state.get("confidence") != invalid_confidence

import re
import pytest
from webapp.app import app, analysis_state


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def populated_state():
    transactions = [
        {"bread", "butter", "milk"},
        {"bread", "butter"},
        {"milk", "cookies"},
        {"bread", "butter", "cookies"},
        {"bread", "milk"},
    ] * 20  # 100 rows to ensure scrolling preview
    product_names = {
        "bread": "Bread",
        "butter": "Butter",
        "milk": "Milk",
        "cookies": "Cookies",
    }
    analysis_state.update({
        "dataset_name": "preview_test.csv",
        "transactions": transactions,
        "product_names": product_names,
        "metrics": {
            "transactions": len(transactions),
            "products": len(product_names),
            "average_items": 2.5,
            "largest_transaction": 3,
            "transaction_col_display": "TransactionID",
            "product_col_display": "ProductName",
            "quantity_col_display": "None",
            "rows_raw": len(transactions),
            "rows_removed": 0,
        },
        "frequent_itemsets": {
            frozenset(["bread"]): 0.8,
            frozenset(["butter"]): 0.6,
            frozenset(["bread", "butter"]): 0.6,
        },
        "rules": [
            {
                "antecedent": ["Bread"],
                "consequent": ["Butter"],
                "support": 0.6,
                "confidence": 0.75,
                "lift": 1.25,
            }
        ],
        "support": 0.02,
        "confidence": 0.60,
        "lift": 0.0,
        "runtime": 0.005,
        "algorithm": "Apriori",
        "analysis_complete": True,
    })
    yield
    analysis_state.clear()


def test_dashboard_loads(client):
    """Verify that the dashboard page loads with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Scalable Association Rule Mining" in html


def test_algorithm_selector_exists(client):
    """Verify algorithm selection controls exist with Apriori and FP-Growth options."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    # Hidden input or selector for algorithm
    assert 'name="algorithm"' in html
    # Apriori card / selector
    assert "APRIORI" in html
    assert 'data-algo="apriori"' in html or 'value="apriori"' in html
    # FP-Growth card / selector
    assert "FP-GROWTH" in html
    assert 'data-algo="fp_growth"' in html or 'value="fp_growth"' in html or 'data-algo="fp-growth"' in html or 'value="fp-growth"' in html


def test_support_input_contract(client):
    """Verify support input exists with valid boundaries (0.1% to 10.0%)."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert 'name="support"' in html
    assert 'min="0.001"' in html
    assert 'max="0.10"' in html
    assert 'step="0.001"' in html


def test_confidence_input_contract(client):
    """Verify confidence input exists with valid boundaries (10% to 100%)."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert 'name="confidence"' in html
    assert 'min="0.10"' in html
    assert 'max="1.00"' in html
    assert 'step="0.01"' in html


def test_dataset_preview_container_contract(client, populated_state):
    """
    Verify Dataset Preview container contract:
    - Container exists in rendered DOM
    - Has height: 300px and overflow-y: auto styling
    - Contains table rows for preview
    """
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    # Check Dataset Preview header and container
    assert "Dataset Preview" in html
    assert "dataset-preview" in html
    assert "dataset-preview-scroll" in html or "dataset-preview-body" in html

    # Check the explicit scroll styling contract
    assert "overflow-y: auto" in html
    assert "height: 300px" in html
    assert "max-height: 300px" in html

    # Verify rows are rendered
    assert '<div class="product-list">' in html
    assert '<span class="product-tag">' in html


def test_results_page_renders(client, populated_state):
    """Verify results page renders itemsets and association rules."""
    response = client.get("/results")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert "Frequent Itemsets" in html
    assert "Association Rules" in html
    assert "Bread" in html
    assert "Butter" in html


def test_browser_tooling_availability():
    """Check if headless browser runner (Playwright) is installed."""
    try:
        import playwright
        playwright_available = True
    except ImportError:
        playwright_available = False

    # Document whether browser runner is present
    if not playwright_available:
        pytest.skip("Playwright browser automation is not installed in the environment.")

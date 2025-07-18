"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "products": [
            {"id": 1, "name": "Product A", "category": "Electronics"},
            {"id": 2, "name": "Product B", "category": "Clothing"},
        ],
        "sales": [
            {"product_id": 1, "quantity": 10, "date": "2023-01-01"},
            {"product_id": 2, "quantity": 5, "date": "2023-01-01"},
        ],
    }


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    return {"connected": True, "data": []}

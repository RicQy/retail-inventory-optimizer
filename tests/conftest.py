"""Pytest configuration and fixtures."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
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


@pytest.fixture
def mock_ge_context():
    """Mock Great Expectations context for testing."""
    mock_context = MagicMock()

    # Mock expectation suite
    mock_suite = MagicMock()
    mock_context.get_expectation_suite.return_value = mock_suite
    mock_context.add_expectation_suite.return_value = mock_suite

    # Mock validator
    mock_validator = MagicMock()
    mock_context.get_validator.return_value = mock_validator

    # Mock validation results
    mock_results = MagicMock()
    mock_results.success = True
    mock_results.results = []
    mock_results.statistics = {
        "evaluated_expectations": 0,
        "successful_expectations": 0,
    }
    mock_validator.validate.return_value = mock_results

    return mock_context


@pytest.fixture
def mock_ge_validation_error():
    """Mock Great Expectations validation with errors."""
    mock_context = MagicMock()

    # Mock expectation suite
    mock_suite = MagicMock()
    mock_context.get_expectation_suite.return_value = mock_suite
    mock_context.add_expectation_suite.return_value = mock_suite

    # Mock validator
    mock_validator = MagicMock()
    mock_context.get_validator.return_value = mock_validator

    # Mock validation results with failures
    mock_results = MagicMock()
    mock_results.success = False

    mock_failed_result = MagicMock()
    mock_failed_result.success = False
    mock_failed_result.expectation_config.expectation_type = (
        "expect_column_values_to_be_between"
    )
    mock_failed_result.result = {"partial_unexpected_list": ["negative_value"]}

    mock_results.results = [mock_failed_result]
    mock_results.statistics = {
        "evaluated_expectations": 1,
        "successful_expectations": 0,
    }
    mock_validator.validate.return_value = mock_results

    return mock_context


@pytest.fixture
def valid_retail_data():
    """Create valid retail data for testing."""
    dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
    data = []

    for date in dates:
        for store in ["STORE_001", "STORE_002"]:
            for category in ["Electronics", "Clothing", "Food"]:
                data.append(
                    {
                        "date": date,
                        "sales": np.random.uniform(100, 1000),
                        "items": np.random.randint(1, 50),
                        "store_id": store,
                        "product_category": category,
                    }
                )

    return pd.DataFrame(data)


@pytest.fixture
def invalid_retail_data():
    """Create invalid retail data for testing."""
    data = {
        "date": ["2023-01-01", "2023-01-02", None],
        "sales": [100.0, -50.0, 200.0],  # Negative sales
        "items": [5, 10, 15],
        "store_id": ["STORE_001", "", "STORE_002"],  # Empty store_id
        "product_category": [
            "Electronics",
            "InvalidCategory",
            "Food",
        ],  # Invalid category
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    with patch("boto3.client") as mock_boto3:
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        yield mock_s3


@pytest.fixture
def mock_cloudwatch_client():
    """Mock CloudWatch client for testing."""
    with patch("boto3.client") as mock_boto3:
        mock_cloudwatch = MagicMock()
        mock_boto3.return_value = mock_cloudwatch
        yield mock_cloudwatch

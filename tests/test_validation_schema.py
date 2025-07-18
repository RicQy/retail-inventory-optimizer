import pandas as pd
import pytest
from datetime import datetime, timedelta
from pandera.errors import SchemaError
from app.etl.validation import RetailDataSchema
from unittest.mock import patch


def test_valid_dataframe():
    """Test that a valid DataFrame passes schema validation."""
    df = pd.DataFrame({
        "date": [datetime.now() - timedelta(days=1)],
        "sales": [100.0],
        "items": [5],
        "store_id": ["STORE_001"],
        "product_category": ["Electronics"],
    })
    schema = RetailDataSchema()
    # Should not raise any exception
    validated_df = schema.RETAIL_SALES_SCHEMA.validate(df)
    assert validated_df is not None


def test_invalid_dataframe():
    """Test that an invalid DataFrame raises a SchemaError."""
    df = pd.DataFrame({
        "date": [datetime.now() + timedelta(days=1)],  # Future date
        "sales": [-100.0],  # Negative sales
        "items": [5],
        "store_id": ["STORE_001"],
        "product_category": ["InvalidCategory"],  # Invalid category
    })
    schema = RetailDataSchema()
    with pytest.raises(SchemaError):
        schema.RETAIL_SALES_SCHEMA.validate(df)


def test_negative_sales():
    """Test that negative sales values raise SchemaError."""
    df = pd.DataFrame({
        "date": [datetime.now() - timedelta(days=1)],
        "sales": [-50.0],  # Negative sales
        "items": [5],
        "store_id": ["STORE_001"],
        "product_category": ["Electronics"],
    })
    schema = RetailDataSchema()
    with pytest.raises(SchemaError):
        schema.RETAIL_SALES_SCHEMA.validate(df)


def test_future_date():
    """Test that future dates raise SchemaError."""
    df = pd.DataFrame({
        "date": [datetime.now() + timedelta(days=1)],  # Future date
        "sales": [100.0],
        "items": [5],
        "store_id": ["STORE_001"],
        "product_category": ["Electronics"],
    })
    schema = RetailDataSchema()
    with pytest.raises(SchemaError):
        schema.RETAIL_SALES_SCHEMA.validate(df)


def test_invalid_product_category():
    """Test that invalid product categories raise SchemaError."""
    df = pd.DataFrame({
        "date": [datetime.now() - timedelta(days=1)],
        "sales": [100.0],
        "items": [5],
        "store_id": ["STORE_001"],
        "product_category": ["InvalidCategory"],  # Invalid category
    })
    schema = RetailDataSchema()
    with pytest.raises(SchemaError):
        schema.RETAIL_SALES_SCHEMA.validate(df)


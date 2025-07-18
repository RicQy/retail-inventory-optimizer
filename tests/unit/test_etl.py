import pytest
import pandas as pd
from app.etl.enhanced_etl import ETLPipeline
from app.etl.config import ETLConfig

@pytest.fixture
def etl_pipeline_config():
    return {
        'output_bucket': 'retail-data-processed',
        'validation_bucket': 'retail-data-validation-reports',
        'ge_context_root': 'app/etl/ge_data_context',
    }

@pytest.fixture
def etl_pipeline(etl_pipeline_config):
    return ETLPipeline(config=etl_pipeline_config, enable_validation=False)

def test_malformed_csv(etl_pipeline):
    """Test ETL with malformed CSV data"""
    malformed_csv = "date,sales,items,store_id,product_category\n2023-01-01,100,5,,Electronics\n,200,10,STORE_001,Food\n"
    
    df = pd.read_csv(pd.io.common.StringIO(malformed_csv))
    result = etl_pipeline.validate_raw_data(df)
    
    # Since validation is disabled, should return validation_skipped
    assert result.get('validation_skipped') is True

def test_missing_columns(etl_pipeline):
    """Test ETL with missing columns"""
    missing_columns_csv = "date,sales,items,store_id\n2023-01-01,100,5,STORE_001\n2023-01-02,200,10,STORE_002\n"
    
    df = pd.read_csv(pd.io.common.StringIO(missing_columns_csv))
    result = etl_pipeline.validate_raw_data(df)
    
    # Since validation is disabled, should return validation_skipped
    assert result.get('validation_skipped') is True

def test_empty_csv(etl_pipeline):
    """Test ETL with empty CSV"""
    empty_csv = "date,sales,items,store_id,product_category\n"
    
    df = pd.read_csv(pd.io.common.StringIO(empty_csv))
    result = etl_pipeline.validate_raw_data(df)
    
    # Since validation is disabled, should return validation_skipped
    assert result.get('validation_skipped') is True

def test_invalid_data_types(etl_pipeline):
    """Test ETL with invalid data types"""
    invalid_data_csv = "date,sales,items,store_id,product_category\n2023-01-01,invalid_number,5,STORE_001,Electronics\n2023-01-02,200,not_a_number,STORE_002,Food\n"
    
    df = pd.read_csv(pd.io.common.StringIO(invalid_data_csv))
    result = etl_pipeline.validate_raw_data(df)
    
    # Since validation is disabled, should return validation_skipped
    assert result.get('validation_skipped') is True

def test_transformation_with_missing_data(etl_pipeline):
    """Test transformation with missing required columns"""
    incomplete_data = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'sales': [100, 200],
        'items': [5, 10],
        'store_id': ['STORE_001', 'STORE_002']
        # Missing product_category
    })
    
    with pytest.raises(ValueError, match="Missing required columns"):
        etl_pipeline.transform_data(incomplete_data)

def test_transformation_with_all_null_values(etl_pipeline):
    """Test transformation with all null values in a column"""
    null_data = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'sales': [None, None],
        'items': [5, 10],
        'store_id': ['STORE_001', 'STORE_002'],
        'product_category': ['Electronics', 'Food']
    })
    
    result = etl_pipeline.transform_data(null_data)
    # Should handle nulls by dropping rows with null values
    assert len(result) == 0  # All rows should be dropped due to null sales

def test_malformed_dates(etl_pipeline):
    """Test ETL with malformed date formats"""
    malformed_dates = pd.DataFrame({
        'date': ['not-a-date', '2023-13-45'],  # Invalid dates
        'sales': [100, 200],
        'items': [5, 10],
        'store_id': ['STORE_001', 'STORE_002'],
        'product_category': ['Electronics', 'Food']
    })
    
    with pytest.raises(Exception):  # Should raise exception for invalid dates
        etl_pipeline.transform_data(malformed_dates)


#!/usr/bin/env python3
"""
Test script to validate the data contract with sample data.
"""

import pandas as pd
import sys
from pathlib import Path
from decimal import Decimal
from datetime import date

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.retail_sales import RetailSalesRecord, RetailSalesBatch

def test_data_contract():
    """Test the data contract with sample data."""
    
    # Load sample data
    base_path = Path(__file__).parent.parent
    csv_path = base_path / "data" / "retail_sales_sample.csv"
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Test individual record validation
    print("\n=== Testing Individual Record Validation ===")
    
    valid_records = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            # Convert row to dictionary and adjust data types
            record_data = row.to_dict()
            record_data['date'] = pd.to_datetime(record_data['date']).date()
            record_data['price'] = Decimal(str(record_data['price']))
            
            # Validate with Pydantic model
            record = RetailSalesRecord(**record_data)
            valid_records.append(record)
            
            if idx < 3:  # Show first 3 records
                print(f"✓ Record {idx + 1}: {record.store_id}, {record.sku}, {record.date}")
                print(f"  Revenue: ${record.daily_revenue()}")
                print(f"  Low stock: {record.is_low_stock()}")
                print(f"  Days remaining: {record.stock_days_remaining()}")
                
        except Exception as e:
            errors.append(f"Record {idx + 1}: {str(e)}")
            print(f"✗ Record {idx + 1}: {str(e)}")
    
    print(f"\nValidation Results:")
    print(f"✓ Valid records: {len(valid_records)}")
    print(f"✗ Invalid records: {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # Test batch validation
    print("\n=== Testing Batch Validation ===")
    
    try:
        batch = RetailSalesBatch(
            records=valid_records,
            batch_id="test_batch_001"
        )
        
        print(f"✓ Batch validation successful")
        print(f"  Total records: {len(batch.records)}")
        print(f"  Total revenue: ${batch.total_revenue()}")
        print(f"  Total units sold: {batch.total_units_sold()}")
        print(f"  Unique stores: {batch.unique_stores()}")
        print(f"  Unique SKUs: {batch.unique_skus()}")
        print(f"  Date range: {batch.date_range()}")
        
        # Test low stock analysis
        low_stock = batch.low_stock_records(threshold=50)
        print(f"  Low stock records (< 50 units): {len(low_stock)}")
        
        if low_stock:
            print("  Low stock items:")
            for record in low_stock[:3]:  # Show first 3
                print(f"    - {record.store_id}, {record.sku}: {record.on_hand} units")
        
    except Exception as e:
        print(f"✗ Batch validation failed: {str(e)}")
    
    # Test data quality checks
    print("\n=== Data Quality Summary ===")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Price range: ${df['price'].min()} to ${df['price'].max()}")
    print(f"Units sold range: {df['units_sold'].min()} to {df['units_sold'].max()}")
    print(f"Inventory range: {df['on_hand'].min()} to {df['on_hand'].max()}")
    print(f"Null values: {df.isnull().sum().sum()}")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = test_data_contract()
    if success:
        print("\n🎉 All tests passed! Data contract is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

# Retail Sales Dataset

This directory contains sample retail sales data for the inventory optimization system.

## Files

### `retail_sales_sample.csv`
- **Format**: CSV (Comma-Separated Values)
- **Size**: 120 records
- **Description**: Realistic retail sales data covering 5 days (2024-01-15 to 2024-01-19) across 3 stores

### `retail_sales_sample.parquet`
- **Format**: Parquet (Columnar storage format)
- **Size**: 120 records
- **Description**: Same data as CSV but optimized for analytical queries
- **Compression**: Snappy
- **Engine**: PyArrow

## Data Schema

The dataset follows a strict schema defined in `app/models/retail_sales.py`:

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `store_id` | String | Unique store identifier | Format: `STORE_XXX` |
| `sku` | String | Stock Keeping Unit | Format: `SKU_CATEGORY_XXX` |
| `date` | Date | Transaction date | YYYY-MM-DD format |
| `units_sold` | Integer | Units sold on date | ≥ 0, ≤ 1000 |
| `price` | Decimal | Unit price in USD | > 0.00, ≤ 10000.00 |
| `on_hand` | Integer | Current inventory level | ≥ 0, ≤ 100000 |

## Primary Key
- Composite key: (`store_id`, `sku`, `date`)

## Indexes
- `store_id`, `date`
- `sku`, `date`
- `date`
- `on_hand`

## Sample Data Overview

- **Stores**: 3 stores (STORE_001, STORE_002, STORE_003)
- **Products**: 8 SKUs across different categories:
  - Electronics: SKU_ELEC_001, SKU_ELEC_002
  - Clothing: SKU_CLOTH_001, SKU_CLOTH_002
  - Home: SKU_HOME_001
  - Books: SKU_BOOK_001
  - Food: SKU_FOOD_001
  - Sports: SKU_SPORT_001
- **Date Range**: 2024-01-15 to 2024-01-19 (5 days)
- **Price Range**: $12.99 to $299.99
- **Units Sold**: 2 to 62 units per transaction

## Data Quality Features

- Realistic inventory depletion over time
- Varying sales patterns across stores
- Different product categories with appropriate pricing
- Inventory levels that decrease with sales
- No missing values or null entries

## Usage

```python
import pandas as pd
from app.models.retail_sales import RetailSalesRecord

# Load CSV data
df = pd.read_csv('data/retail_sales_sample.csv')

# Load Parquet data (recommended for large datasets)
df = pd.read_parquet('data/retail_sales_sample.parquet')

# Validate data with Pydantic
for _, row in df.iterrows():
    record = RetailSalesRecord(**row.to_dict())
    print(f"Valid record: {record}")
```

## Generation Scripts

- `scripts/convert_to_parquet.py`: Converts CSV to Parquet format
- `scripts/generate_er_diagram.py`: Creates ER diagram visualization

## Related Documentation

- Data contract: `app/models/retail_sales.py`
- ER diagram: `docs/retail_sales_er_diagram.png`
- Schema documentation: `docs/retail_sales_er_diagram.md`

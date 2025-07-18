"""
Pydantic models for retail sales data contract.

This module defines the data contract for retail sales records, including
validation rules, field constraints, and data types.
"""

from datetime import date
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class RetailSalesRecord(BaseModel):
    """
    Data contract for a single retail sales record.

    This model defines the structure and validation rules for retail sales data,
    covering daily sales transactions, inventory levels, and pricing information.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    store_id: str
    sku: str
    date: date
    units_sold: int
    price: Decimal
    on_hand: int

    @field_validator("price")
    @classmethod
    def validate_price_range(cls, v):
        """Validate that price is within reasonable range."""
        if v <= 0:
            raise ValueError("Price must be positive")
        if v > 10000:
            raise ValueError("Price cannot exceed $10,000")
        return v

    @field_validator("units_sold")
    @classmethod
    def validate_units_sold(cls, v):
        """Validate units sold is reasonable."""
        if v < 0:
            raise ValueError("Units sold cannot be negative")
        if v > 1000:
            raise ValueError("Units sold cannot exceed 1,000 per day")
        return v

    @field_validator("on_hand")
    @classmethod
    def validate_on_hand(cls, v):
        """Validate on-hand inventory is reasonable."""
        if v < 0:
            raise ValueError("On-hand inventory cannot be negative")
        if v > 100000:
            raise ValueError("On-hand inventory cannot exceed 100,000 units")
        return v

    @field_validator("date")
    @classmethod
    def validate_date_not_future(cls, v):
        """Validate that date is not in the future."""
        from datetime import date as date_type

        if v > date_type.today():
            raise ValueError("Sales date cannot be in the future")
        return v

    def daily_revenue(self) -> Decimal:
        """Calculate daily revenue for this record."""
        return self.price * self.units_sold

    def is_low_stock(self, threshold: int = 50) -> bool:
        """Check if inventory is below threshold."""
        return self.on_hand < threshold

    def stock_days_remaining(self) -> Optional[float]:
        """Calculate days of stock remaining based on daily sales rate."""
        if self.units_sold == 0:
            return None
        return float(self.on_hand / self.units_sold)


class RetailSalesBatch(BaseModel):
    """
    Data contract for a batch of retail sales records.

    This model represents a collection of sales records, typically used for
    bulk data processing and validation.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    records: List[RetailSalesRecord]
    batch_id: Optional[str] = None

    @field_validator("records")
    @classmethod
    def validate_records_not_empty(cls, v):
        """Validate that records list is not empty."""
        if not v:
            raise ValueError("Records list cannot be empty")
        return v

    def total_revenue(self) -> Decimal:
        """Calculate total revenue for all records in batch."""
        return sum(record.daily_revenue() for record in self.records)

    def total_units_sold(self) -> int:
        """Calculate total units sold across all records."""
        return sum(record.units_sold for record in self.records)

    def unique_stores(self) -> set:
        """Get unique store IDs in the batch."""
        return {record.store_id for record in self.records}

    def unique_skus(self) -> set:
        """Get unique SKUs in the batch."""
        return {record.sku for record in self.records}

    def date_range(self) -> tuple:
        """Get date range (min, max) for the batch."""
        dates = [record.date for record in self.records]
        return min(dates), max(dates)

    def low_stock_records(self, threshold: int = 50) -> List[RetailSalesRecord]:
        """Get records with low stock levels."""
        return [record for record in self.records if record.is_low_stock(threshold)]


# Data schema documentation
RETAIL_SALES_SCHEMA = {
    "table_name": "retail_sales",
    "description": "Daily retail sales transactions with inventory data",
    "fields": {
        "store_id": {
            "type": "string",
            "description": "Unique store identifier (format: STORE_XXX)",
            "constraints": ["required", "pattern: ^STORE_\\d{3}$"],
        },
        "sku": {
            "type": "string",
            "description": "Stock Keeping Unit identifier (format: SKU_CATEGORY_XXX)",
            "constraints": ["required", "pattern: ^SKU_[A-Z]{2,10}_\\d{3}$"],
        },
        "date": {
            "type": "date",
            "description": "Transaction date (YYYY-MM-DD)",
            "constraints": ["required", "not_future"],
        },
        "units_sold": {
            "type": "integer",
            "description": "Number of units sold",
            "constraints": ["required", "min: 0", "max: 1000"],
        },
        "price": {
            "type": "decimal",
            "description": "Unit price in USD",
            "constraints": ["required", "min: 0.01", "max: 10000.00", "precision: 2"],
        },
        "on_hand": {
            "type": "integer",
            "description": "Current inventory level",
            "constraints": ["required", "min: 0", "max: 100000"],
        },
    },
    "primary_key": ["store_id", "sku", "date"],
    "indexes": [["store_id", "date"], ["sku", "date"], ["date"], ["on_hand"]],
}

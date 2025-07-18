"""
Example usage of the inventory optimization engine.

This script demonstrates how to use the optimize_inventory function
with sample data and configuration.
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimize import optimize_inventory


def create_sample_data():
    """Create sample forecast data for demonstration."""
    data = {
        "sku": ["SKU_ELEC_001", "SKU_ELEC_002", "SKU_CLOTH_001", "SKU_CLOTH_002"],
        "units_sold": [100, 80, 150, 120],  # Forecasted demand
        "price": [299.99, 89.99, 49.99, 29.99],  # Current price
        "on_hand": [50, 30, 80, 200],  # Current inventory
    }
    return pd.DataFrame(data)


def create_sample_config():
    """Create sample configuration for the optimization."""
    return {
        "budget": 15000,  # Available budget
        "shelf_space": 500,  # Available shelf space
        "supplier_moq": {  # Minimum order quantities per SKU
            "SKU_ELEC_001": 10,
            "SKU_ELEC_002": 5,
            "SKU_CLOTH_001": 20,
            "SKU_CLOTH_002": 15,
        },
        "holding_cost_per_unit": 0.5,  # Cost to hold one unit
        "stockout_cost_per_unit": 25.0,  # Cost per unit of stock-out
        "unit_cost": {  # Cost per unit (if different from price)
            "SKU_ELEC_001": 200.0,
            "SKU_ELEC_002": 60.0,
            "SKU_CLOTH_001": 30.0,
            "SKU_CLOTH_002": 20.0,
        },
        "shelf_space_per_unit": {  # Space per unit
            "SKU_ELEC_001": 2.0,
            "SKU_ELEC_002": 1.5,
            "SKU_CLOTH_001": 1.0,
            "SKU_CLOTH_002": 0.8,
        },
        "max_discount": 0.3,  # Maximum discount allowed
        "min_service_level": 0.9,  # Minimum service level required
    }


def main():
    """Main function to demonstrate the optimization engine."""
    print("Inventory Optimization Engine Demo")
    print("=" * 40)

    # Create sample data
    forecast_df = create_sample_data()
    costs_cfg = create_sample_config()

    print("Sample Forecast Data:")
    print(forecast_df.to_string(index=False))
    print()

    print("Configuration:")
    for key, value in costs_cfg.items():
        print(f"  {key}: {value}")
    print()

    # Run optimization
    print("Running optimization...")
    try:
        results = optimize_inventory(forecast_df, costs_cfg)

        print(f"Optimization Status: {results['status']}")
        print(f"Total Cost: ${results['total_cost']:.2f}")
        print()

        print("Recommended Orders:")
        for sku, quantity in results["orders"].items():
            print(f"  {sku}: {quantity} units")
        print()

        print("Recommended Discounts:")
        for sku, discount in results["discount_schedule"].items():
            print(f"  {sku}: {discount*100:.1f}% discount")
        print()

        print("Optimization Metrics:")
        metrics = results["metrics"]
        print(f"  Total Order Value: ${metrics['total_order_value']:.2f}")
        print(
            f"  Total Shelf Space Used: {metrics['total_shelf_space_used']:.1f} units"
        )
        print(f"  Total Discount Value: ${metrics['total_discount_value']:.2f}")
        print()

        print("Service Levels:")
        for sku, service_level in metrics["service_levels"].items():
            print(f"  {sku}: {service_level*100:.1f}%")

    except Exception as e:
        print(f"Optimization failed: {str(e)}")


if __name__ == "__main__":
    main()

"""
Simple test for the inventory optimization engine.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from optimize import CostsConfig, InventoryOptimizer, optimize_inventory


def test_basic_optimization():
    """Test basic optimization functionality."""
    print("Testing basic optimization...")

    # Create sample data
    forecast_df = pd.DataFrame(
        {
            "sku": ["SKU_001", "SKU_002"],
            "units_sold": [100, 80],
            "price": [299.99, 89.99],
            "on_hand": [50, 30],
        }
    )

    # Create configuration
    costs_cfg = {
        "budget": 15000,
        "shelf_space": 500,
        "supplier_moq": {"SKU_001": 10, "SKU_002": 5},
        "holding_cost_per_unit": 0.5,
        "stockout_cost_per_unit": 25.0,
        "unit_cost": {"SKU_001": 200.0, "SKU_002": 60.0},
        "shelf_space_per_unit": {"SKU_001": 2.0, "SKU_002": 1.5},
        "max_discount": 0.3,
        "min_service_level": 0.9,
    }

    # Run optimization
    results = optimize_inventory(forecast_df, costs_cfg)

    # Verify results
    assert results["status"] == "Optimal"
    assert "orders" in results
    assert "discount_schedule" in results
    assert "total_cost" in results
    assert "metrics" in results

    print("✓ Basic optimization test passed")
    return results


def test_advanced_optimization():
    """Test advanced optimization with class-based approach."""
    print("Testing advanced optimization...")

    # Create sample data
    forecast_df = pd.DataFrame(
        {
            "sku": ["SKU_A", "SKU_B", "SKU_C"],
            "units_sold": [150, 120, 80],
            "price": [49.99, 29.99, 19.99],
            "on_hand": [80, 200, 40],
        }
    )

    # Create configuration object
    config = CostsConfig(
        budget=10000,
        shelf_space=300,
        supplier_moq={"SKU_A": 20, "SKU_B": 15, "SKU_C": 10},
        holding_cost_per_unit=0.3,
        stockout_cost_per_unit=15.0,
        unit_cost={"SKU_A": 30.0, "SKU_B": 20.0, "SKU_C": 12.0},
        shelf_space_per_unit={"SKU_A": 1.0, "SKU_B": 0.8, "SKU_C": 0.5},
        max_discount=0.2,
        min_service_level=0.85,
    )

    # Create optimizer instance
    optimizer = InventoryOptimizer()

    # Run optimization
    results = optimizer.optimize_inventory(forecast_df, config)

    # Verify results
    assert results["status"] == "Optimal"
    assert len(results["orders"]) == 3
    assert len(results["discount_schedule"]) == 3
    assert results["total_cost"] is not None

    print("✓ Advanced optimization test passed")
    return results


def test_constraint_validation():
    """Test constraint validation and edge cases."""
    print("Testing constraint validation...")

    # Test empty DataFrame
    try:
        empty_df = pd.DataFrame()
        costs_cfg = {"budget": 1000, "shelf_space": 100}
        optimize_inventory(empty_df, costs_cfg)
        assert False, "Should have raised ValueError for empty DataFrame"
    except ValueError:
        print("✓ Empty DataFrame validation passed")

    # Test missing columns
    try:
        incomplete_df = pd.DataFrame({"sku": ["SKU_001"]})
        costs_cfg = {"budget": 1000, "shelf_space": 100}
        optimize_inventory(incomplete_df, costs_cfg)
        assert False, "Should have raised ValueError for missing columns"
    except ValueError:
        print("✓ Missing columns validation passed")

    # Test invalid budget
    try:
        valid_df = pd.DataFrame(
            {"sku": ["SKU_001"], "units_sold": [100], "price": [10.0], "on_hand": [50]}
        )
        costs_cfg = {"budget": -1000, "shelf_space": 100}
        optimize_inventory(valid_df, costs_cfg)
        assert False, "Should have raised ValueError for negative budget"
    except ValueError:
        print("✓ Invalid budget validation passed")


def test_metrics_calculation():
    """Test metrics calculation accuracy."""
    print("Testing metrics calculation...")

    # Create simple test case
    forecast_df = pd.DataFrame(
        {"sku": ["SKU_001"], "units_sold": [100], "price": [10.0], "on_hand": [50]}
    )

    costs_cfg = {
        "budget": 5000,
        "shelf_space": 100,
        "supplier_moq": {"SKU_001": 10},
        "holding_cost_per_unit": 1.0,
        "stockout_cost_per_unit": 5.0,
        "unit_cost": {"SKU_001": 8.0},
        "shelf_space_per_unit": {"SKU_001": 1.0},
        "max_discount": 0.1,
        "min_service_level": 0.9,
    }

    results = optimize_inventory(forecast_df, costs_cfg)

    # Check metrics
    assert "total_order_value" in results["metrics"]
    assert "total_shelf_space_used" in results["metrics"]
    assert "service_levels" in results["metrics"]
    assert "total_discount_value" in results["metrics"]

    print("✓ Metrics calculation test passed")
    return results


def main():
    """Run all tests."""
    print("Running inventory optimization tests...")
    print("=" * 50)

    try:
        # Run tests
        test_basic_optimization()
        test_advanced_optimization()
        test_constraint_validation()
        test_metrics_calculation()

        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("Inventory optimization engine is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

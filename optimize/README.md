# Inventory Optimization Engine

This module provides a comprehensive inventory optimization solution using linear programming (PuLP). It helps retailers optimize their inventory levels by minimizing stock-outs and holding costs while respecting various business constraints.

## Features

- **Linear Programming Optimization**: Uses PuLP for efficient mathematical optimization
- **Decision Variables**: 
  - Reorder quantities for each SKU
  - Clearance discount percentages
- **Objective Function**: Minimizes total costs (holding costs + stock-out costs)
- **Constraints**:
  - Budget limitations
  - Shelf space capacity
  - Supplier minimum order quantities (MOQ)
  - Service level requirements
  - Maximum discount limits

## Installation

Ensure you have PuLP installed:

```bash
pip install pulp
```

## Usage

### Basic Usage

```python
from optimize import optimize_inventory
import pandas as pd

# Sample forecast data
forecast_df = pd.DataFrame({
    'sku': ['SKU_001', 'SKU_002'],
    'units_sold': [100, 80],
    'price': [299.99, 89.99],
    'on_hand': [50, 30]
})

# Configuration
costs_cfg = {
    'budget': 15000,
    'shelf_space': 500,
    'supplier_moq': {'SKU_001': 10, 'SKU_002': 5},
    'holding_cost_per_unit': 0.5,
    'stockout_cost_per_unit': 25.0,
    'unit_cost': {'SKU_001': 200.0, 'SKU_002': 60.0},
    'shelf_space_per_unit': {'SKU_001': 2.0, 'SKU_002': 1.5},
    'max_discount': 0.3,
    'min_service_level': 0.9
}

# Run optimization
results = optimize_inventory(forecast_df, costs_cfg)
```

### Advanced Usage

```python
from optimize import InventoryOptimizer, CostsConfig

# Create configuration object
config = CostsConfig(
    budget=15000,
    shelf_space=500,
    supplier_moq={'SKU_001': 10, 'SKU_002': 5},
    holding_cost_per_unit=0.5,
    stockout_cost_per_unit=25.0,
    unit_cost={'SKU_001': 200.0, 'SKU_002': 60.0},
    shelf_space_per_unit={'SKU_001': 2.0, 'SKU_002': 1.5},
    max_discount=0.3,
    min_service_level=0.9
)

# Create optimizer instance
optimizer = InventoryOptimizer()

# Run optimization
results = optimizer.optimize_inventory(forecast_df, config)
```

## Input Data Requirements

### Forecast DataFrame
The forecast DataFrame must contain the following columns:
- `sku`: SKU identifier
- `units_sold`: Forecasted demand
- `price`: Current selling price
- `on_hand`: Current inventory level

### Configuration Parameters
- `budget`: Available budget for purchasing
- `shelf_space`: Total available shelf space
- `supplier_moq`: Dictionary of minimum order quantities per SKU
- `holding_cost_per_unit`: Cost to hold one unit in inventory
- `stockout_cost_per_unit`: Penalty cost for each unit of stock-out
- `unit_cost`: Dictionary of unit costs per SKU
- `shelf_space_per_unit`: Dictionary of space required per unit per SKU
- `max_discount`: Maximum discount percentage allowed (0-1)
- `min_service_level`: Minimum service level required (0-1)

## Output Format

The optimization returns a dictionary with the following structure:

```python
{
    'status': 'Optimal',  # Optimization status
    'orders': {           # Recommended order quantities
        'SKU_001': 40,
        'SKU_002': 42
    },
    'discount_schedule': {  # Recommended discount percentages
        'SKU_001': 0.0,
        'SKU_002': 0.0
    },
    'total_cost': 4326.0,  # Total optimized cost
    'metrics': {           # Additional metrics
        'total_order_value': 12470.0,
        'total_shelf_space_used': 210.0,
        'total_discount_value': 0.0,
        'service_levels': {
            'SKU_001': 0.9,
            'SKU_002': 0.9
        }
    }
}
```

## Example Output

```
Optimization Status: Optimal
Total Cost: $4326.00

Recommended Orders:
  SKU_ELEC_001: 40 units
  SKU_ELEC_002: 42 units
  SKU_CLOTH_001: 55 units
  SKU_CLOTH_002: 15 units

Recommended Discounts:
  SKU_ELEC_001: 0.0% discount
  SKU_ELEC_002: 0.0% discount
  SKU_CLOTH_001: 0.0% discount
  SKU_CLOTH_002: 0.0% discount

Optimization Metrics:
  Total Order Value: $12470.00
  Total Shelf Space Used: 210.0 units
  Total Discount Value: $0.00

Service Levels:
  SKU_ELEC_001: 90.0%
  SKU_ELEC_002: 90.0%
  SKU_CLOTH_001: 90.0%
  SKU_CLOTH_002: 100.0%
```

## Files

- `inventory_optimizer.py`: Main optimization engine
- `example_usage.py`: Example usage script
- `__init__.py`: Module initialization and exports
- `README.md`: This documentation

## Dependencies

- pandas: Data manipulation
- pulp: Linear programming solver
- numpy: Numerical computations
- logging: Logging functionality

## Error Handling

The optimizer includes comprehensive error handling for:
- Missing required columns in forecast data
- Invalid configuration parameters
- Infeasible optimization problems
- Solver failures

## Logging

The module uses Python's logging framework to provide detailed information about:
- Optimization progress
- Constraint additions
- Solution status
- Performance metrics

## Performance Considerations

- The optimizer uses integer programming for reorder quantities
- Continuous variables are used for discount percentages
- CBC solver is used by default (included with PuLP)
- Performance scales well with the number of SKUs

## Future Enhancements

Potential areas for improvement:
1. Multi-period optimization
2. Uncertainty handling with stochastic programming
3. Additional constraint types (supplier capacity, lead times)
4. Integration with forecasting uncertainty
5. Multi-objective optimization (cost vs. service level trade-offs)

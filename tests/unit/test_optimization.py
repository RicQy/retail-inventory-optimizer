"""
Tests for optimization module with edge cases: stockout vs overstock scenarios.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from optimize.inventory_optimizer import InventoryOptimizer, CostsConfig, optimize_inventory


@pytest.fixture
def sample_forecast_data():
    """Create sample forecast data for testing."""
    return pd.DataFrame({
        'sku': ['SKU_001', 'SKU_002', 'SKU_003'],
        'units_sold': [100, 50, 200],
        'price': [10.0, 25.0, 5.0],
        'on_hand': [30, 80, 10]
    })


@pytest.fixture
def stockout_scenario_data():
    """Create data representing stockout scenario."""
    return pd.DataFrame({
        'sku': ['SKU_001', 'SKU_002', 'SKU_003'],
        'units_sold': [100, 80, 150],  # High demand
        'price': [20.0, 30.0, 15.0],
        'on_hand': [10, 5, 8]  # Very low inventory
    })


@pytest.fixture
def overstock_scenario_data():
    """Create data representing overstock scenario."""
    return pd.DataFrame({
        'sku': ['SKU_001', 'SKU_002', 'SKU_003'],
        'units_sold': [20, 15, 30],  # Low demand
        'price': [25.0, 40.0, 18.0],
        'on_hand': [200, 300, 250]  # High inventory
    })


@pytest.fixture
def basic_costs_config():
    """Create basic costs configuration."""
    return CostsConfig(
        budget=10000,
        shelf_space=500,
        supplier_moq={'SKU_001': 10, 'SKU_002': 5, 'SKU_003': 15},
        holding_cost_per_unit=2.0,
        stockout_cost_per_unit=50.0,
        unit_cost={'SKU_001': 8.0, 'SKU_002': 20.0, 'SKU_003': 12.0},
        shelf_space_per_unit={'SKU_001': 1.0, 'SKU_002': 2.0, 'SKU_003': 0.5},
        max_discount=0.3,
        min_service_level=0.9
    )


@pytest.fixture
def tight_budget_config():
    """Create configuration with tight budget constraints."""
    return CostsConfig(
        budget=1000,  # Very tight budget
        shelf_space=500,
        supplier_moq={'SKU_001': 10, 'SKU_002': 5, 'SKU_003': 15},
        holding_cost_per_unit=2.0,
        stockout_cost_per_unit=50.0,
        unit_cost={'SKU_001': 8.0, 'SKU_002': 20.0, 'SKU_003': 12.0},
        shelf_space_per_unit={'SKU_001': 1.0, 'SKU_002': 2.0, 'SKU_003': 0.5},
        max_discount=0.3,
        min_service_level=0.9
    )


@pytest.fixture
def high_stockout_cost_config():
    """Create configuration with high stockout costs."""
    return CostsConfig(
        budget=10000,
        shelf_space=500,
        supplier_moq={'SKU_001': 10, 'SKU_002': 5, 'SKU_003': 15},
        holding_cost_per_unit=1.0,
        stockout_cost_per_unit=200.0,  # Very high stockout cost
        unit_cost={'SKU_001': 8.0, 'SKU_002': 20.0, 'SKU_003': 12.0},
        shelf_space_per_unit={'SKU_001': 1.0, 'SKU_002': 2.0, 'SKU_003': 0.5},
        max_discount=0.3,
        min_service_level=0.9
    )


class TestInventoryOptimizer:
    """Test class for InventoryOptimizer."""

    def test_stockout_scenario_optimization(self, stockout_scenario_data, high_stockout_cost_config):
        """Test optimization with stockout scenario."""
        optimizer = InventoryOptimizer()
        
        with patch('pulp.LpProblem') as mock_problem_class:
            mock_problem = Mock()
            mock_problem_class.return_value = mock_problem
            mock_problem.solve.return_value = None
            mock_problem.status = 1  # Optimal
            
            # Mock decision variables
            mock_reorder_vars = {}
            mock_discount_vars = {}
            for sku in stockout_scenario_data['sku']:
                mock_reorder_var = Mock()
                mock_reorder_var.varValue = 100  # High reorder quantities
                mock_reorder_vars[sku] = mock_reorder_var
                
                mock_discount_var = Mock()
                mock_discount_var.varValue = 0.0  # No discount needed
                mock_discount_vars[sku] = mock_discount_var
            
            optimizer.reorder_variables = mock_reorder_vars
            optimizer.discount_variables = mock_discount_vars
            
            with patch('pulp.LpVariable.dicts') as mock_lpvars:
                mock_lpvars.side_effect = [mock_reorder_vars, mock_discount_vars]
                
                with patch('pulp.value', return_value=1000.0):
                    results = optimizer.optimize_inventory(stockout_scenario_data, high_stockout_cost_config)
                    
                    assert results['status'] == 'Optimal'
                    assert len(results['orders']) == 3
                    assert all(order > 0 for order in results['orders'].values())

    def test_overstock_scenario_optimization(self, overstock_scenario_data, basic_costs_config):
        """Test optimization with overstock scenario."""
        optimizer = InventoryOptimizer()
        
        with patch('pulp.LpProblem') as mock_problem_class:
            mock_problem = Mock()
            mock_problem_class.return_value = mock_problem
            mock_problem.solve.return_value = None
            mock_problem.status = 1  # Optimal
            
            # Mock decision variables
            mock_reorder_vars = {}
            mock_discount_vars = {}
            for sku in overstock_scenario_data['sku']:
                mock_reorder_var = Mock()
                mock_reorder_var.varValue = 0  # No reorder needed
                mock_reorder_vars[sku] = mock_reorder_var
                
                mock_discount_var = Mock()
                mock_discount_var.varValue = 0.2  # Apply discount to clear inventory
                mock_discount_vars[sku] = mock_discount_var
            
            optimizer.reorder_variables = mock_reorder_vars
            optimizer.discount_variables = mock_discount_vars
            
            with patch('pulp.LpVariable.dicts') as mock_lpvars:
                mock_lpvars.side_effect = [mock_reorder_vars, mock_discount_vars]
                
                with patch('pulp.value', return_value=500.0):
                    results = optimizer.optimize_inventory(overstock_scenario_data, basic_costs_config)
                    
                    assert results['status'] == 'Optimal'
                    assert len(results['orders']) == 3
                    assert all(order == 0 for order in results['orders'].values())  # No reordering
                    assert all(discount > 0 for discount in results['discount_schedule'].values())

    def test_tight_budget_constraint(self, sample_forecast_data, tight_budget_config):
        """Test optimization with tight budget constraints."""
        optimizer = InventoryOptimizer()
        
        with patch('pulp.LpProblem') as mock_problem_class:
            mock_problem = Mock()
            mock_problem_class.return_value = mock_problem
            mock_problem.solve.return_value = None
            mock_problem.status = 1  # Optimal
            
            # Mock decision variables with limited ordering due to budget
            mock_reorder_vars = {}
            mock_discount_vars = {}
            for i, sku in enumerate(sample_forecast_data['sku']):
                mock_reorder_var = Mock()
                mock_reorder_var.varValue = 10 if i == 0 else 0  # Only order one SKU
                mock_reorder_vars[sku] = mock_reorder_var
                
                mock_discount_var = Mock()
                mock_discount_var.varValue = 0.1
                mock_discount_vars[sku] = mock_discount_var
            
            optimizer.reorder_variables = mock_reorder_vars
            optimizer.discount_variables = mock_discount_vars
            
            with patch('pulp.LpVariable.dicts') as mock_lpvars:
                mock_lpvars.side_effect = [mock_reorder_vars, mock_discount_vars]
                
                with patch('pulp.value', return_value=800.0):
                    results = optimizer.optimize_inventory(sample_forecast_data, tight_budget_config)
                    
                    assert results['status'] == 'Optimal'
                    assert sum(results['orders'].values()) <= 10  # Limited ordering

    def test_validation_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        optimizer = InventoryOptimizer()
        empty_df = pd.DataFrame()
        basic_config = CostsConfig(
            budget=1000, shelf_space=100, supplier_moq={}, 
            holding_cost_per_unit=1.0, stockout_cost_per_unit=10.0,
            unit_cost={}, shelf_space_per_unit={}
        )
        
        with pytest.raises(ValueError, match="Forecast DataFrame is empty"):
            optimizer.optimize_inventory(empty_df, basic_config)

    def test_validation_missing_columns(self):
        """Test validation with missing required columns."""
        optimizer = InventoryOptimizer()
        incomplete_df = pd.DataFrame({
            'sku': ['SKU_001'],
            'units_sold': [100]
            # Missing 'price' and 'on_hand' columns
        })
        basic_config = CostsConfig(
            budget=1000, shelf_space=100, supplier_moq={}, 
            holding_cost_per_unit=1.0, stockout_cost_per_unit=10.0,
            unit_cost={}, shelf_space_per_unit={}
        )
        
        with pytest.raises(ValueError, match="Missing required columns"):
            optimizer.optimize_inventory(incomplete_df, basic_config)

    def test_validation_negative_budget(self, sample_forecast_data):
        """Test validation with negative budget."""
        optimizer = InventoryOptimizer()
        invalid_config = CostsConfig(
            budget=-1000,  # Negative budget
            shelf_space=100, supplier_moq={}, 
            holding_cost_per_unit=1.0, stockout_cost_per_unit=10.0,
            unit_cost={}, shelf_space_per_unit={}
        )
        
        with pytest.raises(ValueError, match="Budget must be positive"):
            optimizer.optimize_inventory(sample_forecast_data, invalid_config)

    def test_validation_negative_shelf_space(self, sample_forecast_data):
        """Test validation with negative shelf space."""
        optimizer = InventoryOptimizer()
        invalid_config = CostsConfig(
            budget=1000,
            shelf_space=-100,  # Negative shelf space
            supplier_moq={}, 
            holding_cost_per_unit=1.0, stockout_cost_per_unit=10.0,
            unit_cost={}, shelf_space_per_unit={}
        )
        
        with pytest.raises(ValueError, match="Shelf space must be positive"):
            optimizer.optimize_inventory(sample_forecast_data, invalid_config)

    def test_service_level_constraints(self, sample_forecast_data, basic_costs_config):
        """Test service level constraints are applied."""
        optimizer = InventoryOptimizer()
        
        with patch('pulp.LpProblem') as mock_problem_class:
            mock_problem = Mock()
            mock_problem_class.return_value = mock_problem
            mock_problem.solve.return_value = None
            mock_problem.status = 1  # Optimal
            
            # Mock decision variables
            mock_reorder_vars = {}
            mock_discount_vars = {}
            for sku in sample_forecast_data['sku']:
                mock_reorder_var = Mock()
                mock_reorder_var.varValue = 50  # Moderate reorder
                mock_reorder_vars[sku] = mock_reorder_var
                
                mock_discount_var = Mock()
                mock_discount_var.varValue = 0.1
                mock_discount_vars[sku] = mock_discount_var
            
            optimizer.reorder_variables = mock_reorder_vars
            optimizer.discount_variables = mock_discount_vars
            
            with patch('pulp.LpVariable.dicts') as mock_lpvars:
                mock_lpvars.side_effect = [mock_reorder_vars, mock_discount_vars]
                
                with patch('pulp.value', return_value=2000.0):
                    results = optimizer.optimize_inventory(sample_forecast_data, basic_costs_config)
                    
                    # Check service levels are calculated
                    assert 'service_levels' in results['metrics']
                    for sku, service_level in results['metrics']['service_levels'].items():
                        assert 0 <= service_level <= 1

    def test_moq_constraints(self, sample_forecast_data, basic_costs_config):
        """Test MOQ constraints are respected."""
        optimizer = InventoryOptimizer()
        
        with patch('pulp.LpProblem') as mock_problem_class:
            mock_problem = Mock()
            mock_problem_class.return_value = mock_problem
            mock_problem.solve.return_value = None
            mock_problem.status = 1  # Optimal
            
            # Mock decision variables that respect MOQ
            mock_reorder_vars = {}
            mock_discount_vars = {}
            for sku in sample_forecast_data['sku']:
                mock_reorder_var = Mock()
                # Should be either 0 or >= MOQ
                moq = basic_costs_config.supplier_moq.get(sku, 0)
                mock_reorder_var.varValue = moq if moq > 0 else 0
                mock_reorder_vars[sku] = mock_reorder_var
                
                mock_discount_var = Mock()
                mock_discount_var.varValue = 0.0
                mock_discount_vars[sku] = mock_discount_var
            
            optimizer.reorder_variables = mock_reorder_vars
            optimizer.discount_variables = mock_discount_vars
            
            with patch('pulp.LpVariable.dicts') as mock_lpvars:
                mock_lpvars.side_effect = [mock_reorder_vars, mock_discount_vars]
                
                with patch('pulp.value', return_value=1500.0):
                    results = optimizer.optimize_inventory(sample_forecast_data, basic_costs_config)
                    
                    # Check MOQ constraints are respected
                    for sku, order_qty in results['orders'].items():
                        moq = basic_costs_config.supplier_moq.get(sku, 0)
                        if order_qty > 0:
                            assert order_qty >= moq

    def test_infeasible_solution(self, sample_forecast_data):
        """Test handling of infeasible optimization problem."""
        optimizer = InventoryOptimizer()
        
        # Create impossible constraints (zero budget but high demand)
        impossible_config = CostsConfig(
            budget=0,  # Zero budget
            shelf_space=1,  # Minimal space
            supplier_moq={'SKU_001': 1000, 'SKU_002': 1000, 'SKU_003': 1000},  # High MOQ
            holding_cost_per_unit=1.0, stockout_cost_per_unit=10.0,
            unit_cost={'SKU_001': 1000.0, 'SKU_002': 1000.0, 'SKU_003': 1000.0},  # High cost
            shelf_space_per_unit={'SKU_001': 100.0, 'SKU_002': 100.0, 'SKU_003': 100.0}
        )
        
        with patch('pulp.LpProblem') as mock_problem_class:
            mock_problem = Mock()
            mock_problem_class.return_value = mock_problem
            mock_problem.solve.return_value = None
            mock_problem.status = -1  # Infeasible
            
            with patch('pulp.LpVariable.dicts') as mock_lpvars:
                mock_lpvars.side_effect = [{}, {}]
                
                results = optimizer.optimize_inventory(sample_forecast_data, impossible_config)
                
                assert results['status'] == 'Infeasible'
                assert results['orders'] == {}
                assert results['total_cost'] is None

    def test_metrics_calculation(self, sample_forecast_data, basic_costs_config):
        """Test metrics calculation accuracy."""
        optimizer = InventoryOptimizer()
        
        with patch('pulp.LpProblem') as mock_problem_class:
            mock_problem = Mock()
            mock_problem_class.return_value = mock_problem
            mock_problem.solve.return_value = None
            mock_problem.status = 1  # Optimal
            
            # Mock decision variables
            mock_reorder_vars = {}
            mock_discount_vars = {}
            for sku in sample_forecast_data['sku']:
                mock_reorder_var = Mock()
                mock_reorder_var.varValue = 25
                mock_reorder_vars[sku] = mock_reorder_var
                
                mock_discount_var = Mock()
                mock_discount_var.varValue = 0.05
                mock_discount_vars[sku] = mock_discount_var
            
            optimizer.reorder_variables = mock_reorder_vars
            optimizer.discount_variables = mock_discount_vars
            
            with patch('pulp.LpVariable.dicts') as mock_lpvars:
                mock_lpvars.side_effect = [mock_reorder_vars, mock_discount_vars]
                
                with patch('pulp.value', return_value=1800.0):
                    results = optimizer.optimize_inventory(sample_forecast_data, basic_costs_config)
                    
                    # Verify metrics are calculated
                    metrics = results['metrics']
                    assert 'total_order_value' in metrics
                    assert 'total_shelf_space_used' in metrics
                    assert 'service_levels' in metrics
                    assert 'total_discount_value' in metrics
                    
                    # Check metrics are reasonable
                    assert metrics['total_order_value'] > 0
                    assert metrics['total_shelf_space_used'] > 0
                    assert len(metrics['service_levels']) == 3


class TestOptimizationEdgeCases:
    """Test edge cases for optimization."""

    def test_zero_demand_scenario(self, basic_costs_config):
        """Test optimization with zero demand scenario."""
        zero_demand_data = pd.DataFrame({
            'sku': ['SKU_001', 'SKU_002'],
            'units_sold': [0, 0],  # No demand
            'price': [10.0, 20.0],
            'on_hand': [100, 50]
        })
        
        optimizer = InventoryOptimizer()
        
        with patch('pulp.LpProblem') as mock_problem_class:
            mock_problem = Mock()
            mock_problem_class.return_value = mock_problem
            mock_problem.solve.return_value = None
            mock_problem.status = 1  # Optimal
            
            # With zero demand, should not order anything
            mock_reorder_vars = {}
            mock_discount_vars = {}
            for sku in zero_demand_data['sku']:
                mock_reorder_var = Mock()
                mock_reorder_var.varValue = 0
                mock_reorder_vars[sku] = mock_reorder_var
                
                mock_discount_var = Mock()
                mock_discount_var.varValue = 0.3  # High discount to clear inventory
                mock_discount_vars[sku] = mock_discount_var
            
            optimizer.reorder_variables = mock_reorder_vars
            optimizer.discount_variables = mock_discount_vars
            
            with patch('pulp.LpVariable.dicts') as mock_lpvars:
                mock_lpvars.side_effect = [mock_reorder_vars, mock_discount_vars]
                
                with patch('pulp.value', return_value=0.0):
                    results = optimizer.optimize_inventory(zero_demand_data, basic_costs_config)
                    
                    assert all(order == 0 for order in results['orders'].values())
                    assert all(discount > 0 for discount in results['discount_schedule'].values())

    def test_extreme_demand_scenario(self, basic_costs_config):
        """Test optimization with extreme demand scenario."""
        extreme_demand_data = pd.DataFrame({
            'sku': ['SKU_001', 'SKU_002'],
            'units_sold': [10000, 5000],  # Very high demand
            'price': [10.0, 20.0],
            'on_hand': [5, 3]  # Very low inventory
        })
        
        optimizer = InventoryOptimizer()
        
        with patch('pulp.LpProblem') as mock_problem_class:
            mock_problem = Mock()
            mock_problem_class.return_value = mock_problem
            mock_problem.solve.return_value = None
            mock_problem.status = 1  # Optimal
            
            # Should order as much as budget allows
            mock_reorder_vars = {}
            mock_discount_vars = {}
            for sku in extreme_demand_data['sku']:
                mock_reorder_var = Mock()
                mock_reorder_var.varValue = 500  # High reorder
                mock_reorder_vars[sku] = mock_reorder_var
                
                mock_discount_var = Mock()
                mock_discount_var.varValue = 0.0  # No discount needed
                mock_discount_vars[sku] = mock_discount_var
            
            optimizer.reorder_variables = mock_reorder_vars
            optimizer.discount_variables = mock_discount_vars
            
            with patch('pulp.LpVariable.dicts') as mock_lpvars:
                mock_lpvars.side_effect = [mock_reorder_vars, mock_discount_vars]
                
                with patch('pulp.value', return_value=8000.0):
                    results = optimizer.optimize_inventory(extreme_demand_data, basic_costs_config)
                    
                    assert all(order > 0 for order in results['orders'].values())
                    assert all(discount == 0 for discount in results['discount_schedule'].values())

    def test_single_sku_optimization(self, basic_costs_config):
        """Test optimization with single SKU."""
        single_sku_data = pd.DataFrame({
            'sku': ['SKU_001'],
            'units_sold': [100],
            'price': [15.0],
            'on_hand': [20]
        })
        
        optimizer = InventoryOptimizer()
        
        with patch('pulp.LpProblem') as mock_problem_class:
            mock_problem = Mock()
            mock_problem_class.return_value = mock_problem
            mock_problem.solve.return_value = None
            mock_problem.status = 1  # Optimal
            
            mock_reorder_vars = {'SKU_001': Mock()}
            mock_reorder_vars['SKU_001'].varValue = 80
            
            mock_discount_vars = {'SKU_001': Mock()}
            mock_discount_vars['SKU_001'].varValue = 0.1
            
            optimizer.reorder_variables = mock_reorder_vars
            optimizer.discount_variables = mock_discount_vars
            
            with patch('pulp.LpVariable.dicts') as mock_lpvars:
                mock_lpvars.side_effect = [mock_reorder_vars, mock_discount_vars]
                
                with patch('pulp.value', return_value=640.0):
                    results = optimizer.optimize_inventory(single_sku_data, basic_costs_config)
                    
                    assert len(results['orders']) == 1
                    assert 'SKU_001' in results['orders']
                    assert results['orders']['SKU_001'] == 80


class TestConvenienceFunction:
    """Test the convenience function for backward compatibility."""

    def test_convenience_function_with_dict_config(self, sample_forecast_data):
        """Test convenience function with dictionary configuration."""
        dict_config = {
            'budget': 5000,
            'shelf_space': 200,
            'supplier_moq': {'SKU_001': 10, 'SKU_002': 5, 'SKU_003': 15},
            'holding_cost_per_unit': 1.5,
            'stockout_cost_per_unit': 25.0,
            'unit_cost': {'SKU_001': 8.0, 'SKU_002': 20.0, 'SKU_003': 12.0},
            'shelf_space_per_unit': {'SKU_001': 1.0, 'SKU_002': 2.0, 'SKU_003': 0.5}
        }
        
        with patch('optimize.inventory_optimizer.InventoryOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            
            mock_results = {
                'status': 'Optimal',
                'orders': {'SKU_001': 50, 'SKU_002': 25, 'SKU_003': 75},
                'discount_schedule': {'SKU_001': 0.1, 'SKU_002': 0.05, 'SKU_003': 0.15},
                'total_cost': 3000.0,
                'metrics': {}
            }
            mock_optimizer.optimize_inventory.return_value = mock_results
            
            result = optimize_inventory(sample_forecast_data, dict_config)
            
            assert result == mock_results
            mock_optimizer.optimize_inventory.assert_called_once()

    def test_convenience_function_with_defaults(self, sample_forecast_data):
        """Test convenience function with default values."""
        minimal_config = {
            'budget': 1000
        }
        
        with patch('optimize.inventory_optimizer.InventoryOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            
            mock_results = {
                'status': 'Optimal',
                'orders': {'SKU_001': 10, 'SKU_002': 5, 'SKU_003': 15},
                'discount_schedule': {'SKU_001': 0.0, 'SKU_002': 0.0, 'SKU_003': 0.0},
                'total_cost': 800.0,
                'metrics': {}
            }
            mock_optimizer.optimize_inventory.return_value = mock_results
            
            result = optimize_inventory(sample_forecast_data, minimal_config)
            
            assert result == mock_results
            mock_optimizer.optimize_inventory.assert_called_once()
            
            # Check that defaults were applied
            call_args = mock_optimizer.optimize_inventory.call_args
            config_used = call_args[0][1]  # Second argument is the config
            assert config_used.shelf_space == 1000  # Default value
            assert config_used.min_service_level == 0.95  # Default value

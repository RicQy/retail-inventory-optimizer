"""
Inventory Optimization Engine using PuLP for Linear Programming

This module implements an inventory optimization system that minimizes stock-outs
and holding costs while respecting budget, shelf space, and supplier MOQ constraints.
"""

import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CostsConfig:
    """Configuration class for inventory optimization costs and constraints."""
    budget: float
    shelf_space: float
    supplier_moq: Dict[str, float]  # MOQ per SKU
    holding_cost_per_unit: float
    stockout_cost_per_unit: float
    unit_cost: Dict[str, float]  # Cost per unit per SKU
    shelf_space_per_unit: Dict[str, float]  # Space per unit per SKU
    max_discount: float = 0.5  # Maximum discount allowed
    min_service_level: float = 0.95  # Minimum service level


class InventoryOptimizer:
    """
    Inventory optimization engine using linear programming.
    
    This class provides functionality to optimize inventory levels by:
    - Minimizing stock-outs and holding costs
    - Respecting budget constraints
    - Managing shelf space limitations
    - Meeting supplier MOQ requirements
    - Determining optimal clearance discounts
    """
    
    def __init__(self):
        self.problem = None
        self.reorder_variables = {}
        self.discount_variables = {}
        self.results = {}
        
    def _validate_inputs(self, forecast_df: pd.DataFrame, costs_cfg: CostsConfig) -> None:
        """Validate input data and configuration."""
        required_columns = ['sku', 'units_sold', 'price', 'on_hand']
        missing_columns = [col for col in required_columns if col not in forecast_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if forecast_df.empty:
            raise ValueError("Forecast DataFrame is empty")
        
        if costs_cfg.budget <= 0:
            raise ValueError("Budget must be positive")
        
        if costs_cfg.shelf_space <= 0:
            raise ValueError("Shelf space must be positive")
    
    def _create_decision_variables(self, skus: List[str]) -> None:
        """Create decision variables for the optimization problem."""
        # Reorder quantity variables (integer, >= 0)
        self.reorder_variables = LpVariable.dicts(
            "ReorderQuantity", 
            skus, 
            lowBound=0, 
            cat='Integer'
        )
        
        # Clearance discount variables (continuous, 0 <= x <= max_discount)
        self.discount_variables = LpVariable.dicts(
            "ClearanceDiscount", 
            skus, 
            lowBound=0, 
            upBound=1.0, 
            cat='Continuous'
        )
        
        logger.info(f"Created decision variables for {len(skus)} SKUs")
    
    def _create_objective_function(self, forecast_df: pd.DataFrame, costs_cfg: CostsConfig) -> None:
        """Create the objective function to minimize total costs."""
        objective_terms = []
        
        for _, row in forecast_df.iterrows():
            sku = row['sku']
            forecasted_demand = row['units_sold']
            current_inventory = row['on_hand']
            
            # Holding cost component
            holding_cost = costs_cfg.holding_cost_per_unit * self.reorder_variables[sku]
            
            # Stock-out cost component (simplified)
            # For simplicity, we'll use a penalty for not meeting expected demand
            expected_shortage = max(0, forecasted_demand - current_inventory)
            stockout_cost = costs_cfg.stockout_cost_per_unit * expected_shortage
            
            objective_terms.append(holding_cost)
            if stockout_cost > 0:
                objective_terms.append(stockout_cost)
        
        self.problem += lpSum(objective_terms)
        logger.info("Objective function created")
    
    def _add_budget_constraint(self, forecast_df: pd.DataFrame, costs_cfg: CostsConfig) -> None:
        """Add budget constraint to the optimization problem."""
        cost_terms = []
        
        for _, row in forecast_df.iterrows():
            sku = row['sku']
            unit_cost = costs_cfg.unit_cost.get(sku, row['price'])
            
            # Simplified cost calculation (assuming no discount for now)
            cost_terms.append(unit_cost * self.reorder_variables[sku])
        
        self.problem += lpSum(cost_terms) <= costs_cfg.budget
        logger.info("Budget constraint added")
    
    def _add_shelf_space_constraint(self, forecast_df: pd.DataFrame, costs_cfg: CostsConfig) -> None:
        """Add shelf space constraint to the optimization problem."""
        total_space = 0
        
        for _, row in forecast_df.iterrows():
            sku = row['sku']
            space_per_unit = costs_cfg.shelf_space_per_unit.get(sku, 1.0)
            total_space += space_per_unit * self.reorder_variables[sku]
        
        self.problem += total_space <= costs_cfg.shelf_space
        logger.info("Shelf space constraint added")
    
    def _add_moq_constraints(self, forecast_df: pd.DataFrame, costs_cfg: CostsConfig) -> None:
        """Add supplier MOQ constraints to the optimization problem."""
        for _, row in forecast_df.iterrows():
            sku = row['sku']
            moq = costs_cfg.supplier_moq.get(sku, 0)
            
            # Either order nothing or order at least MOQ
            if moq > 0:
                self.problem += self.reorder_variables[sku] >= moq
        
        logger.info("MOQ constraints added")
    
    def _add_service_level_constraints(self, forecast_df: pd.DataFrame, costs_cfg: CostsConfig) -> None:
        """Add service level constraints to ensure minimum stock availability."""
        for _, row in forecast_df.iterrows():
            sku = row['sku']
            forecasted_demand = row['units_sold']
            current_inventory = row['on_hand']
            
            # Ensure we can meet at least min_service_level of demand
            required_stock = forecasted_demand * costs_cfg.min_service_level
            self.problem += (current_inventory + self.reorder_variables[sku]) >= required_stock
        
        logger.info("Service level constraints added")
    
    def _add_discount_constraints(self, forecast_df: pd.DataFrame, costs_cfg: CostsConfig) -> None:
        """Add constraints for discount variables."""
        for _, row in forecast_df.iterrows():
            sku = row['sku']
            # Discount should not exceed maximum allowed
            self.problem += self.discount_variables[sku] <= costs_cfg.max_discount
        
        logger.info("Discount constraints added")
    
    def optimize_inventory(self, forecast_df: pd.DataFrame, costs_cfg: CostsConfig) -> Dict:
        """
        Optimize inventory levels using linear programming.
        
        Args:
            forecast_df: DataFrame with columns ['sku', 'units_sold', 'price', 'on_hand']
            costs_cfg: Configuration object with cost parameters and constraints
            
        Returns:
            Dictionary with optimization results including orders and discount schedules
        """
        try:
            # Validate inputs
            self._validate_inputs(forecast_df, costs_cfg)
            
            # Initialize optimization problem
            self.problem = LpProblem("InventoryOptimization", LpMinimize)
            
            # Get unique SKUs
            skus = forecast_df['sku'].unique().tolist()
            
            # Create decision variables
            self._create_decision_variables(skus)
            
            # Create objective function
            self._create_objective_function(forecast_df, costs_cfg)
            
            # Add constraints
            self._add_budget_constraint(forecast_df, costs_cfg)
            self._add_shelf_space_constraint(forecast_df, costs_cfg)
            self._add_moq_constraints(forecast_df, costs_cfg)
            self._add_service_level_constraints(forecast_df, costs_cfg)
            self._add_discount_constraints(forecast_df, costs_cfg)
            
            # Solve the problem
            logger.info("Starting optimization...")
            self.problem.solve()
            
            # Check solution status
            status = LpStatus[self.problem.status]
            logger.info(f"Optimization status: {status}")
            
            if status != "Optimal":
                logger.warning(f"Optimization did not find optimal solution. Status: {status}")
                return {
                    'status': status,
                    'orders': {},
                    'discount_schedule': {},
                    'total_cost': None
                }
            
            # Extract results
            orders = {sku: int(self.reorder_variables[sku].varValue) 
                     for sku in skus}
            
            discount_schedule = {sku: float(self.discount_variables[sku].varValue) 
                               for sku in skus}
            
            total_cost = value(self.problem.objective)
            
            # Calculate additional metrics
            results = {
                'status': status,
                'orders': orders,
                'discount_schedule': discount_schedule,
                'total_cost': total_cost,
                'metrics': self._calculate_metrics(forecast_df, orders, discount_schedule, costs_cfg)
            }
            
            logger.info(f"Optimization completed successfully. Total cost: {total_cost}")
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
    
    def _calculate_metrics(self, forecast_df: pd.DataFrame, orders: Dict[str, int], 
                          discount_schedule: Dict[str, float], costs_cfg: CostsConfig) -> Dict:
        """Calculate additional metrics for the optimization results."""
        metrics = {
            'total_order_value': 0,
            'total_shelf_space_used': 0,
            'service_levels': {},
            'total_discount_value': 0
        }
        
        for _, row in forecast_df.iterrows():
            sku = row['sku']
            order_qty = orders.get(sku, 0)
            discount = discount_schedule.get(sku, 0)
            
            # Calculate order value
            unit_cost = costs_cfg.unit_cost.get(sku, row['price'])
            discounted_cost = unit_cost * (1 - discount)
            order_value = discounted_cost * order_qty
            metrics['total_order_value'] += order_value
            
            # Calculate shelf space used
            space_per_unit = costs_cfg.shelf_space_per_unit.get(sku, 1.0)
            metrics['total_shelf_space_used'] += space_per_unit * order_qty
            
            # Calculate service level
            current_inventory = row['on_hand']
            total_available = current_inventory + order_qty
            forecasted_demand = row['units_sold']
            service_level = min(1.0, total_available / max(forecasted_demand, 1))
            metrics['service_levels'][sku] = service_level
            
            # Calculate discount value
            discount_value = unit_cost * discount * order_qty
            metrics['total_discount_value'] += discount_value
        
        return metrics


# Convenience function for backward compatibility
def optimize_inventory(forecast_df: pd.DataFrame, costs_cfg: dict) -> Dict:
    """
    Convenience function to optimize inventory using the InventoryOptimizer class.
    
    Args:
        forecast_df: DataFrame with forecast data
        costs_cfg: Dictionary with configuration parameters
        
    Returns:
        Dictionary with optimization results
    """
    # Convert dict to CostsConfig object
    config = CostsConfig(
        budget=costs_cfg.get('budget', 10000),
        shelf_space=costs_cfg.get('shelf_space', 1000),
        supplier_moq=costs_cfg.get('supplier_moq', {}),
        holding_cost_per_unit=costs_cfg.get('holding_cost_per_unit', 1.0),
        stockout_cost_per_unit=costs_cfg.get('stockout_cost_per_unit', 10.0),
        unit_cost=costs_cfg.get('unit_cost', {}),
        shelf_space_per_unit=costs_cfg.get('shelf_space_per_unit', {}),
        max_discount=costs_cfg.get('max_discount', 0.5),
        min_service_level=costs_cfg.get('min_service_level', 0.95)
    )
    
    optimizer = InventoryOptimizer()
    return optimizer.optimize_inventory(forecast_df, config)

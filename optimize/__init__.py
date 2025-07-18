"""Optimization module for inventory optimization and resource allocation."""

from .inventory_optimizer import CostsConfig, InventoryOptimizer, optimize_inventory

# Expose the main optimization function
__all__ = ["optimize_inventory", "InventoryOptimizer", "CostsConfig"]

__version__ = "0.1.0"

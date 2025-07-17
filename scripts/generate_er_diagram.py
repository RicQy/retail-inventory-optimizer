#!/usr/bin/env python3
"""
Generate an ER diagram for the retail sales dataset.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_er_diagram():
    """Create and save an ER diagram for the retail sales dataset."""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Retail Sales Dataset - Entity Relationship Diagram', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Main entity rectangle
    entity_rect = patches.Rectangle((2, 2), 6, 4, linewidth=2, 
                                    edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(entity_rect)
    
    # Entity title
    ax.text(5, 5.5, 'RETAIL_SALES', fontsize=14, fontweight='bold', ha='center')
    
    # Primary key fields (marked with *)
    pk_fields = [
        '* store_id : VARCHAR(20)',
        '* sku : VARCHAR(50)', 
        '* date : DATE'
    ]
    
    # Regular fields
    regular_fields = [
        'units_sold : INTEGER',
        'price : DECIMAL(10,2)',
        'on_hand : INTEGER'
    ]
    
    # Draw primary key fields
    y_pos = 5.0
    for field in pk_fields:
        ax.text(2.2, y_pos, field, fontsize=10, fontweight='bold', va='center')
        y_pos -= 0.3
    
    # Draw separator line
    ax.plot([2.1, 7.9], [y_pos + 0.1, y_pos + 0.1], 'k-', linewidth=1)
    y_pos -= 0.2
    
    # Draw regular fields
    for field in regular_fields:
        ax.text(2.2, y_pos, field, fontsize=10, va='center')
        y_pos -= 0.3
    
    # Add constraints and notes
    ax.text(5, 1.5, 'PRIMARY KEY: (store_id, sku, date)', 
            fontsize=10, fontweight='bold', ha='center', style='italic')
    ax.text(5, 1.2, 'INDEXES: store_id, sku, date, on_hand', 
            fontsize=9, ha='center', style='italic')
    
    # Add field descriptions
    descriptions = [
        'store_id: Unique store identifier (STORE_XXX)',
        'sku: Stock Keeping Unit (SKU_CATEGORY_XXX)',
        'date: Transaction date (YYYY-MM-DD)',
        'units_sold: Number of units sold (≥ 0)',
        'price: Unit price in USD (> 0.00)',
        'on_hand: Current inventory level (≥ 0)'
    ]
    
    # Add description box
    desc_rect = patches.Rectangle((0.5, 0.2), 9, 0.8, linewidth=1, 
                                  edgecolor='gray', facecolor='lightyellow', alpha=0.5)
    ax.add_patch(desc_rect)
    
    ax.text(5, 0.9, 'Field Descriptions:', fontsize=10, fontweight='bold', ha='center')
    
    y_desc = 0.75
    for i, desc in enumerate(descriptions):
        if i < 3:  # First column
            ax.text(0.7, y_desc - (i * 0.1), desc, fontsize=8, va='center')
        else:  # Second column
            ax.text(5.2, y_desc - ((i-3) * 0.1), desc, fontsize=8, va='center')
    
    plt.tight_layout()
    
    # Save the diagram
    output_path = Path(__file__).parent.parent / "docs" / "retail_sales_er_diagram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ER diagram saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    create_er_diagram()

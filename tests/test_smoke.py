"""
Smoke tests to verify basic functionality without complex dependencies.
"""
import pytest


def test_basic_imports():
    """Test that we can import basic modules without errors."""
    try:
        import api
        import etl
        import forecast
        import optimize
        assert True  # If we get here, imports worked
    except ImportError:
        pytest.fail("Failed to import basic modules")


def test_pandas_numpy():
    """Test that pandas and numpy work correctly."""
    import pandas as pd
    import numpy as np
    
    # Create simple test data
    data = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })
    
    assert len(data) == 3
    assert data['a'].sum() == 6
    assert np.array([1, 2, 3]).sum() == 6


def test_project_structure():
    """Test that the project structure is intact."""
    import os
    
    # Check that main directories exist
    assert os.path.exists('api')
    assert os.path.exists('etl')
    assert os.path.exists('forecast')
    assert os.path.exists('optimize')
    assert os.path.exists('tests')
    
    # Check that main files exist
    assert os.path.exists('api/main.py')
    assert os.path.exists('pyproject.toml')
    assert os.path.exists('README.md')

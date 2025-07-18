"""
Minimal tests that should always pass in any environment.
"""


def test_python_basics():
    """Test basic Python functionality."""
    # Test basic operations
    assert 1 + 1 == 2
    assert "hello" + " world" == "hello world"
    assert len([1, 2, 3]) == 3
    
    # Test basic data structures
    data = {"a": 1, "b": 2}
    assert data["a"] == 1
    assert list(data.keys()) == ["a", "b"]


def test_list_operations():
    """Test list operations."""
    items = [1, 2, 3, 4, 5]
    assert sum(items) == 15
    assert max(items) == 5
    assert min(items) == 1
    assert len(items) == 5


def test_string_operations():
    """Test string operations."""
    text = "Retail Inventory Optimizer"
    assert text.lower().startswith("retail")
    assert "Inventory" in text
    assert text.split()[0] == "Retail"

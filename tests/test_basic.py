"""Basic tests to verify setup."""



def test_sample_data(sample_data):
    """Test that sample data fixture works."""
    assert "products" in sample_data
    assert "sales" in sample_data
    assert len(sample_data["products"]) == 2
    assert len(sample_data["sales"]) == 2


def test_mock_database(mock_database):
    """Test that mock database fixture works."""
    assert mock_database["connected"] is True
    assert isinstance(mock_database["data"], list)


def test_basic_math():
    """Test basic math operations."""
    assert 2 + 2 == 4
    assert 10 - 5 == 5
    assert 3 * 4 == 12
    assert 8 / 2 == 4

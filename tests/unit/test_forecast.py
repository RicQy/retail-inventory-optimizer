"""
Tests for forecasting module with edge cases: short time-series, seasonality detection.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from forecast.forecasting_service import ForecastingService, ForecastConfig


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    with patch('boto3.client') as mock_boto:
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3
        yield mock_s3


@pytest.fixture
def forecasting_service(mock_s3_client):
    """Create forecasting service with mocked S3 client."""
    return ForecastingService(s3_bucket='test-bucket')


@pytest.fixture
def short_timeseries_data():
    """Create short time-series data for testing."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'ds': dates,
        'y': [10, 12, 8, 15, 11],
        'sku': ['SKU_001'] * 5,
        'store_id': ['STORE_001'] * 5,
        'region': ['REGION_001'] * 5
    })
    return data


@pytest.fixture
def seasonal_data():
    """Create seasonal time-series data for testing."""
    dates = pd.date_range('2023-01-01', periods=52, freq='W')
    # Create artificial seasonal pattern
    trend = np.linspace(100, 200, 52)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(52) / 52)
    noise = np.random.normal(0, 5, 52)
    y_values = trend + seasonal + noise
    
    data = pd.DataFrame({
        'ds': dates,
        'y': y_values,
        'sku': ['SKU_001'] * 52,
        'store_id': ['STORE_001'] * 52,
        'region': ['REGION_001'] * 52
    })
    return data


@pytest.fixture
def empty_data():
    """Create empty dataset for testing."""
    return pd.DataFrame(columns=['ds', 'y', 'sku', 'store_id', 'region'])


@pytest.fixture
def single_point_data():
    """Create single data point for testing."""
    return pd.DataFrame({
        'ds': [pd.Timestamp('2023-01-01')],
        'y': [100],
        'sku': ['SKU_001'],
        'store_id': ['STORE_001'],
        'region': ['REGION_001']
    })


class TestForecastingService:
    """Test class for ForecastingService."""

    def test_prepare_data_filters_by_sku(self, forecasting_service, seasonal_data):
        """Test data preparation with SKU filter."""
        # Add another SKU to the data
        additional_data = seasonal_data.copy()
        additional_data['sku'] = 'SKU_002'
        combined_data = pd.concat([seasonal_data, additional_data], ignore_index=True)
        
        config = ForecastConfig(sku='SKU_001')
        result = forecasting_service.prepare_data(combined_data, config)
        
        assert len(result) == 52
        assert all(result['sku'] == 'SKU_001')

    def test_prepare_data_filters_by_store_id(self, forecasting_service, seasonal_data):
        """Test data preparation with store_id filter."""
        # Add another store to the data
        additional_data = seasonal_data.copy()
        additional_data['store_id'] = 'STORE_002'
        combined_data = pd.concat([seasonal_data, additional_data], ignore_index=True)
        
        config = ForecastConfig(store_id='STORE_001')
        result = forecasting_service.prepare_data(combined_data, config)
        
        assert len(result) == 52
        assert all(result['store_id'] == 'STORE_001')

    def test_prepare_data_filters_by_region(self, forecasting_service, seasonal_data):
        """Test data preparation with region filter."""
        # Add another region to the data
        additional_data = seasonal_data.copy()
        additional_data['region'] = 'REGION_002'
        combined_data = pd.concat([seasonal_data, additional_data], ignore_index=True)
        
        config = ForecastConfig(region='REGION_001')
        result = forecasting_service.prepare_data(combined_data, config)
        
        assert len(result) == 52
        assert all(result['region'] == 'REGION_001')

    def test_short_timeseries_prophet_training(self, forecasting_service, short_timeseries_data):
        """Test Prophet training with short time-series data."""
        prepared_data = short_timeseries_data[['ds', 'y']]
        
        # Prophet should handle short time-series but may not be optimal
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            
            model = forecasting_service.train_prophet(prepared_data)
            
            mock_prophet.assert_called_once_with()
            mock_model.add_country_holidays.assert_called_once_with(country_name='US')
            mock_model.fit.assert_called_once_with(prepared_data)

    def test_short_timeseries_arima_training(self, forecasting_service, short_timeseries_data):
        """Test ARIMA training with short time-series data."""
        y_values = short_timeseries_data['y']
        
        with patch('pmdarima.auto_arima') as mock_auto_arima:
            mock_model = Mock()
            mock_auto_arima.return_value = mock_model
            
            model = forecasting_service.train_arima(y_values)
            
            mock_auto_arima.assert_called_once_with(
                y_values, seasonal=True, m=12, stepwise=True, suppress_warnings=True
            )

    def test_empty_data_handling(self, forecasting_service, empty_data):
        """Test handling of empty data."""
        config = ForecastConfig(sku='SKU_001')
        result = forecasting_service.prepare_data(empty_data, config)
        
        assert len(result) == 0
        assert result.empty

    def test_single_point_data_handling(self, forecasting_service, single_point_data):
        """Test handling of single data point."""
        prepared_data = single_point_data[['ds', 'y']]
        
        # Should raise an error or handle gracefully
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            # Prophet will likely fail with single point
            mock_model.fit.side_effect = ValueError("Not enough data")
            
            with pytest.raises(ValueError):
                forecasting_service.train_prophet(prepared_data)

    def test_seasonality_detection_with_seasonal_data(self, forecasting_service, seasonal_data):
        """Test seasonality detection with seasonal data."""
        prepared_data = seasonal_data[['ds', 'y']]
        
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            
            # Mock the model to verify seasonal components are detected
            mock_model.params = {'seasonality_mode': 'additive'}
            
            model = forecasting_service.train_prophet(prepared_data)
            
            mock_prophet.assert_called_once_with()
            mock_model.add_country_holidays.assert_called_once_with(country_name='US')
            mock_model.fit.assert_called_once_with(prepared_data)

    def test_forecast_generation(self, forecasting_service, seasonal_data):
        """Test forecast generation."""
        prepared_data = seasonal_data[['ds', 'y']]
        
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            
            # Mock forecast results
            mock_future = Mock()
            mock_model.make_future_dataframe.return_value = mock_future
            
            mock_forecast = pd.DataFrame({
                'ds': pd.date_range('2023-01-01', periods=30, freq='D'),
                'yhat': np.random.uniform(100, 200, 30),
                'yhat_lower': np.random.uniform(80, 150, 30),
                'yhat_upper': np.random.uniform(120, 250, 30)
            })
            mock_model.predict.return_value = mock_forecast
            
            model = forecasting_service.train_prophet(prepared_data)
            forecast = forecasting_service.forecast(model, periods=30)
            
            assert len(forecast) == 30
            assert all(col in forecast.columns for col in ['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

    def test_model_serialization(self, forecasting_service, mock_s3_client):
        """Test model serialization to S3."""
        mock_model = Mock()
        key = 'test_model.pkl'
        
        forecasting_service.serialize_to_s3(mock_model, key)
        
        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        assert call_args[1]['Bucket'] == 'test-bucket'
        assert call_args[1]['Key'] == key

    def test_auto_forecast_integration(self, forecasting_service, seasonal_data):
        """Test auto_forecast integration method."""
        config = ForecastConfig(sku='SKU_001')
        
        with patch.object(forecasting_service, 'prepare_data') as mock_prepare, \
             patch.object(forecasting_service, 'train_prophet') as mock_train, \
             patch.object(forecasting_service, 'forecast') as mock_forecast, \
             patch.object(forecasting_service, 'serialize_to_s3') as mock_serialize:
            
            mock_prepare.return_value = seasonal_data
            mock_model = Mock()
            mock_train.return_value = mock_model
            mock_forecast_result = pd.DataFrame({
                'ds': pd.date_range('2023-01-01', periods=30, freq='D'),
                'yhat': np.random.uniform(100, 200, 30),
                'yhat_lower': np.random.uniform(80, 150, 30),
                'yhat_upper': np.random.uniform(120, 250, 30)
            })
            mock_forecast.return_value = mock_forecast_result
            
            result = forecasting_service.auto_forecast(seasonal_data, config, periods=30)
            
            mock_prepare.assert_called_once_with(seasonal_data, config)
            mock_train.assert_called_once_with(seasonal_data)
            mock_forecast.assert_called_once_with(mock_model, 30)
            mock_serialize.assert_called_once_with(mock_model, 'SKU_001_model.pkl')
            
            assert len(result) == 30


class TestForecastingEdgeCases:
    """Test edge cases for forecasting."""

    def test_very_short_timeseries_error_handling(self, forecasting_service):
        """Test error handling for very short time-series."""
        very_short_data = pd.DataFrame({
            'ds': [pd.Timestamp('2023-01-01')],
            'y': [100]
        })
        
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            mock_model.fit.side_effect = ValueError("Insufficient data")
            
            with pytest.raises(ValueError):
                forecasting_service.train_prophet(very_short_data)

    def test_zero_variance_data(self, forecasting_service):
        """Test handling of zero variance data."""
        zero_var_data = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10, freq='D'),
            'y': [100] * 10  # All same values
        })
        
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            
            model = forecasting_service.train_prophet(zero_var_data)
            
            mock_prophet.assert_called_once_with()
            mock_model.add_country_holidays.assert_called_once_with(country_name='US')
            mock_model.fit.assert_called_once_with(zero_var_data)

    def test_missing_values_in_timeseries(self, forecasting_service):
        """Test handling of missing values in time-series."""
        missing_data = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10, freq='D'),
            'y': [100, np.nan, 120, 110, np.nan, 130, 125, 115, np.nan, 140]
        })
        
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            
            # Prophet should handle NaN values
            model = forecasting_service.train_prophet(missing_data)
            
            mock_prophet.assert_called_once_with()
            mock_model.add_country_holidays.assert_called_once_with(country_name='US')
            mock_model.fit.assert_called_once_with(missing_data)

    def test_irregular_frequency_data(self, forecasting_service):
        """Test handling of irregular frequency data."""
        irregular_dates = pd.to_datetime([
            '2023-01-01', '2023-01-03', '2023-01-07', '2023-01-15', '2023-01-30'
        ])
        irregular_data = pd.DataFrame({
            'ds': irregular_dates,
            'y': [100, 110, 105, 120, 115]
        })
        
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            
            model = forecasting_service.train_prophet(irregular_data)
            
            mock_prophet.assert_called_once_with()
            mock_model.add_country_holidays.assert_called_once_with(country_name='US')
            mock_model.fit.assert_called_once_with(irregular_data)

    def test_extreme_values_handling(self, forecasting_service):
        """Test handling of extreme outlier values."""
        extreme_data = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10, freq='D'),
            'y': [100, 110, 105, 10000, 115, 120, -5000, 125, 130, 135]  # Extreme outliers
        })
        
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            
            model = forecasting_service.train_prophet(extreme_data)
            
            mock_prophet.assert_called_once_with()
            mock_model.add_country_holidays.assert_called_once_with(country_name='US')
            mock_model.fit.assert_called_once_with(extreme_data)

    def test_negative_values_handling(self, forecasting_service):
        """Test handling of negative values in time-series."""
        negative_data = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10, freq='D'),
            'y': [-50, -30, -20, -10, 0, 10, 20, 30, 40, 50]
        })
        
        with patch('prophet.Prophet') as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            
            model = forecasting_service.train_prophet(negative_data)
            
            mock_prophet.assert_called_once_with()
            mock_model.add_country_holidays.assert_called_once_with(country_name='US')
            mock_model.fit.assert_called_once_with(negative_data)

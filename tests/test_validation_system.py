"""
Test suite for the data validation system integration.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.etl.config import ETLConfig, get_environment_config
from app.etl.enhanced_etl import ETLPipeline, create_sample_data

# Import our validation modules
from app.etl.validation import DataQualityMetrics, DataValidator


class TestDataValidation:
    """Test cases for data validation functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.sample_data = self._create_test_data()
        self.invalid_data = self._create_invalid_data()

        # Mock AWS clients to avoid actual calls
        self.mock_config = {
            "output_bucket": "test-bucket",
            "validation_bucket": "test-validation-bucket",
            "ge_context_root": "test_ge_context",
        }

    def _create_test_data(self) -> pd.DataFrame:
        """Create valid test data"""
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        data = []

        for date in dates:
            for store in ["STORE_001", "STORE_002"]:
                for category in ["Electronics", "Clothing", "Food"]:
                    data.append(
                        {
                            "date": date,
                            "sales": np.random.uniform(100, 1000),
                            "items": np.random.randint(1, 50),
                            "store_id": store,
                            "product_category": category,
                        }
                    )

        return pd.DataFrame(data)

    def _create_invalid_data(self) -> pd.DataFrame:
        """Create invalid test data for testing error handling"""
        data = {
            "date": ["2023-01-01", "2023-01-02", None],
            "sales": [100.0, -50.0, 200.0],  # Negative sales
            "items": [5, 10, 15],
            "store_id": ["STORE_001", "", "STORE_002"],  # Empty store_id
            "product_category": [
                "Electronics",
                "InvalidCategory",
                "Food",
            ],  # Invalid category
        }
        return pd.DataFrame(data)

    @patch("great_expectations.get_context")
    @patch("boto3.client")
    def test_data_quality_metrics_calculation(self, mock_boto3, mock_gx_context):
        """Test data quality metrics calculation"""
        # Mock CloudWatch client
        mock_cloudwatch = MagicMock()
        mock_boto3.return_value = mock_cloudwatch

        # Mock Great Expectations context
        mock_context = MagicMock()
        mock_gx_context.return_value = mock_context

        DataQualityMetrics()
        validator = DataValidator(
            ge_data_context_root="test_context",
            s3_bucket="test-bucket",
            fail_on_validation_error=False,
        )

        # Test with valid data
        metrics = validator.calculate_data_quality_metrics(self.sample_data)

        # Assertions
        assert "total_rows" in metrics
        assert "total_columns" in metrics
        assert "overall_null_ratio" in metrics
        assert "duplicate_rows" in metrics
        assert metrics["total_rows"] == len(self.sample_data)
        assert metrics["overall_null_ratio"] >= 0
        assert metrics["duplicate_rows"] >= 0

        # Test with invalid data
        invalid_metrics = validator.calculate_data_quality_metrics(self.invalid_data)

        # Should detect null values
        assert invalid_metrics["date_null_count"] > 0
        assert invalid_metrics["date_null_ratio"] > 0

    @patch("great_expectations.get_context")
    @patch("boto3.client")
    def test_pandera_validation(self, mock_boto3, mock_gx_context):
        """Test Pandera schema validation"""
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3

        # Mock Great Expectations context
        mock_context = MagicMock()
        mock_gx_context.return_value = mock_context

        validator = DataValidator(
            ge_data_context_root="test_context",
            s3_bucket="test-bucket",
            fail_on_validation_error=False,
        )

        # Test with valid data
        result = validator.validate_with_pandera(
            self.sample_data, validator.schemas.RETAIL_SALES_SCHEMA
        )

        assert result["passed"] == True
        assert len(result["errors"]) == 0

        # Test with invalid data
        invalid_result = validator.validate_with_pandera(
            self.invalid_data, validator.schemas.RETAIL_SALES_SCHEMA
        )

        assert invalid_result["passed"] == False
        assert len(invalid_result["errors"]) > 0

    @patch("great_expectations.get_context")
    @patch("boto3.client")
    def test_schema_drift_detection(self, mock_boto3, mock_gx_context):
        """Test schema drift detection"""
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3

        # Mock Great Expectations context
        mock_context = MagicMock()
        mock_gx_context.return_value = mock_context

        validator = DataValidator(
            ge_data_context_root="test_context",
            s3_bucket="test-bucket",
            fail_on_validation_error=False,
        )

        # Reference schema
        reference_schema = {
            "columns": ["date", "sales", "items", "store_id", "product_category"],
            "dtypes": {
                "date": "datetime64[ns]",
                "sales": "float64",
                "items": "int64",
                "store_id": "object",
                "product_category": "object",
            },
        }

        # Test with matching schema
        drift_result = validator.detect_schema_drift(self.sample_data, reference_schema)
        assert drift_result["has_drift"] == False

        # Test with schema drift (extra column)
        drift_data = self.sample_data.copy()
        drift_data["extra_column"] = "test"

        drift_result = validator.detect_schema_drift(drift_data, reference_schema)
        assert drift_result["has_drift"] == True
        assert "extra_columns" in drift_result["drift_details"]
        assert "extra_column" in drift_result["drift_details"]["extra_columns"]

    @patch("great_expectations.get_context")
    @patch("boto3.client")
    def test_comprehensive_validation(self, mock_boto3, mock_gx_context):
        """Test comprehensive data validation"""
        mock_s3 = MagicMock()
        mock_cloudwatch = MagicMock()
        mock_boto3.side_effect = [mock_s3, mock_cloudwatch]

        # Mock Great Expectations context
        mock_context = MagicMock()
        mock_gx_context.return_value = mock_context

        validator = DataValidator(
            ge_data_context_root="test_context",
            s3_bucket="test-bucket",
            fail_on_validation_error=False,
        )

        # Test comprehensive validation
        result = validator.validate_data(self.sample_data, "test_dataset")

        assert "dataset_name" in result
        assert "timestamp" in result
        assert "data_shape" in result
        assert "overall_success" in result
        assert "quality_metrics" in result
        assert "pandera_results" in result
        assert "schema_drift" in result

        assert result["dataset_name"] == "test_dataset"
        assert result["data_shape"] == self.sample_data.shape


class TestETLPipeline:
    """Test cases for ETL pipeline with validation"""

    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "output_bucket": "test-output-bucket",
            "validation_bucket": "test-validation-bucket",
            "ge_context_root": "test_ge_context",
        }

    @patch("great_expectations.get_context")
    @patch("boto3.client")
    def test_etl_pipeline_initialization(self, mock_boto3, mock_gx_context):
        """Test ETL pipeline initialization"""
        mock_s3 = MagicMock()
        mock_cloudwatch = MagicMock()
        mock_s3_validator = MagicMock()
        mock_cloudwatch_metrics = MagicMock()
        mock_boto3.side_effect = [
            mock_s3,
            mock_cloudwatch,
            mock_s3_validator,
            mock_cloudwatch_metrics,
        ]

        # Mock Great Expectations context
        mock_context = MagicMock()
        mock_gx_context.return_value = mock_context

        pipeline = ETLPipeline(
            config=self.test_config, fail_fast=True, enable_validation=True
        )

        assert pipeline.config == self.test_config
        assert pipeline.fail_fast == True
        assert pipeline.enable_validation == True
        assert pipeline.validator is not None

    @patch("boto3.client")
    def test_data_transformation(self, mock_boto3):
        """Test data transformation with validation"""
        mock_s3 = MagicMock()
        mock_cloudwatch = MagicMock()
        mock_boto3.side_effect = [mock_s3, mock_cloudwatch]

        pipeline = ETLPipeline(
            config=self.test_config, fail_fast=False, enable_validation=False
        )

        # Create test data
        test_data = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "sales": [100.0, 200.0, 300.0],
                "items": [5, 10, 15],
                "store_id": ["STORE_001", "STORE_002", "STORE_001"],
                "product_category": ["Electronics", "Clothing", "Food"],
            }
        )

        # Transform data
        transformed_data = pipeline.transform_data(test_data)

        # Assertions
        assert "year" in transformed_data.columns
        assert "month" in transformed_data.columns
        assert "day" in transformed_data.columns
        assert "is_weekend" in transformed_data.columns
        assert "sales_normalized" in transformed_data.columns

        # Check data types
        assert transformed_data["date"].dtype == "datetime64[ns]"
        assert transformed_data["sales"].dtype == "float64"
        assert transformed_data["items"].dtype == "Int64"

    @patch("boto3.client")
    def test_metrics_emission(self, mock_boto3):
        """Test CloudWatch metrics emission"""
        mock_s3 = MagicMock()
        mock_cloudwatch = MagicMock()
        mock_boto3.side_effect = [mock_s3, mock_cloudwatch]

        pipeline = ETLPipeline(
            config=self.test_config, fail_fast=False, enable_validation=False
        )

        # Test metrics emission
        test_metrics = {"test_metric_1": 100, "test_metric_2": 0.5, "test_count": 10}

        pipeline.emit_etl_metrics(test_metrics, "test_stage")

        # Verify CloudWatch client was called
        mock_cloudwatch.put_metric_data.assert_called_once()

        # Verify the call parameters
        call_args = mock_cloudwatch.put_metric_data.call_args
        assert call_args[1]["Namespace"] == "ETL/Pipeline"
        assert len(call_args[1]["MetricData"]) == 3


class TestConfiguration:
    """Test cases for configuration management"""

    def test_etl_config_defaults(self):
        """Test ETL configuration defaults"""
        config = ETLConfig.get_etl_config()

        assert "output_bucket" in config
        assert "validation_bucket" in config
        assert "ge_context_root" in config
        assert "fail_fast_on_validation" in config
        assert "enable_cloudwatch_metrics" in config
        assert "retail_data_schema" in config

        # Check retail data schema
        schema = config["retail_data_schema"]
        assert "required_columns" in schema
        assert "data_types" in schema
        assert "business_rules" in schema

        required_columns = schema["required_columns"]
        assert "date" in required_columns
        assert "sales" in required_columns
        assert "items" in required_columns
        assert "store_id" in required_columns
        assert "product_category" in required_columns

    def test_environment_config(self):
        """Test environment-specific configuration"""
        dev_config = get_environment_config("development")
        get_environment_config("staging")
        prod_config = get_environment_config("production")

        # Development should have different settings
        assert dev_config["fail_fast_on_validation"] == False
        assert dev_config["enable_cloudwatch_metrics"] == False
        assert dev_config["log_level"] == "DEBUG"

        # Production should have strict settings
        assert prod_config["fail_fast_on_validation"] == True
        assert prod_config["enable_cloudwatch_metrics"] == True
        assert prod_config["log_level"] == "INFO"
        assert prod_config["max_retries"] == 3

    def test_validation_config(self):
        """Test validation-specific configuration"""
        validation_config = ETLConfig.get_validation_config()

        assert "ge_context_root" in validation_config
        assert "validation_bucket" in validation_config
        assert "fail_fast" in validation_config
        assert "max_null_ratio" in validation_config
        assert "max_duplicate_ratio" in validation_config
        assert "retail_data_schema" in validation_config


class TestIntegration:
    """Integration tests for the complete validation system"""

    @patch("great_expectations.get_context")
    @patch("boto3.client")
    def test_end_to_end_validation(self, mock_boto3, mock_gx_context):
        """Test end-to-end validation flow"""
        mock_s3 = MagicMock()
        mock_cloudwatch = MagicMock()
        mock_s3_validator = MagicMock()
        mock_cloudwatch_metrics = MagicMock()
        mock_boto3.side_effect = [
            mock_s3,
            mock_cloudwatch,
            mock_s3_validator,
            mock_cloudwatch_metrics,
        ]

        # Mock Great Expectations context
        mock_context = MagicMock()
        mock_gx_context.return_value = mock_context

        # Create test data
        test_df = create_sample_data()

        # Initialize pipeline
        pipeline = ETLPipeline(
            config={
                "output_bucket": "test-bucket",
                "validation_bucket": "test-validation-bucket",
                "ge_context_root": "test_ge_context",
            },
            fail_fast=False,
            enable_validation=True,
        )

        # Test raw data validation
        validation_result = pipeline.validate_raw_data(test_df, "integration_test")

        assert "overall_success" in validation_result
        assert "quality_metrics" in validation_result
        assert "pandera_results" in validation_result

        # Test data transformation
        transformed_df = pipeline.transform_data(test_df)

        # Test transformed data validation
        pipeline.validate_transformed_data(transformed_df)

        # Should have additional columns after transformation
        assert len(transformed_df.columns) > len(test_df.columns)
        assert "year" in transformed_df.columns
        assert "month" in transformed_df.columns
        assert "is_weekend" in transformed_df.columns

    @patch("great_expectations.get_context")
    @patch("boto3.client")
    def test_failure_handling(self, mock_boto3, mock_gx_context):
        """Test failure handling in validation"""
        mock_s3 = MagicMock()
        mock_cloudwatch = MagicMock()
        mock_s3_validator = MagicMock()
        mock_cloudwatch_metrics = MagicMock()
        mock_boto3.side_effect = [
            mock_s3,
            mock_cloudwatch,
            mock_s3_validator,
            mock_cloudwatch_metrics,
        ]

        # Mock Great Expectations context
        mock_context = MagicMock()
        mock_gx_context.return_value = mock_context

        # Create invalid data
        invalid_df = pd.DataFrame(
            {
                "date": [None, "2023-01-02"],
                "sales": [-100, 200],  # Negative sales
                "items": [5, 10],
                "store_id": ["", "STORE_002"],  # Empty store_id
                "product_category": ["InvalidCategory", "Food"],  # Invalid category
            }
        )

        # Test with fail_fast=False
        pipeline = ETLPipeline(
            config={
                "output_bucket": "test-bucket",
                "validation_bucket": "test-validation-bucket",
                "ge_context_root": "test_ge_context",
            },
            fail_fast=False,
            enable_validation=True,
        )

        validation_result = pipeline.validate_raw_data(invalid_df, "failure_test")

        # Should not fail but should report errors
        assert (
            "validation_failed" in validation_result
            or validation_result.get("overall_success") == False
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

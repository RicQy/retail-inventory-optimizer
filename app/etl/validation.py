"""
Data validation module integrating Great Expectations and Pandera for
comprehensive data quality checks.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

# CloudWatch metrics
import boto3

# Great Expectations imports
import great_expectations as gx
import pandas as pd

# Pandera imports
import pandera as pa
import polars as pl
import structlog
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.exceptions import DataContextError
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

logger = structlog.get_logger()


class DataValidationError(Exception):
    """Custom exception for data validation failures"""

    pass


class SchemaValidationError(DataValidationError):
    """Custom exception for schema validation failures"""

    pass


class DataQualityMetrics:
    """Class to handle data quality metrics and CloudWatch integration"""

    def __init__(self, cloudwatch_namespace: str = "ETL/DataQuality"):
        self.cloudwatch = boto3.client("cloudwatch")
        self.namespace = cloudwatch_namespace

    def emit_metrics(self, metrics: Dict[str, Any], dimensions: Dict[str, str] = None):
        """Emit metrics to CloudWatch"""
        try:
            if dimensions is None:
                dimensions = {}

            metric_data = []

            for metric_name, value in metrics.items():
                metric_data.append(
                    {
                        "MetricName": metric_name,
                        "Value": float(value),
                        "Unit": (
                            "Count" if "count" in metric_name.lower() else "Percent"
                        ),
                        "Dimensions": [
                            {"Name": k, "Value": v} for k, v in dimensions.items()
                        ],
                    }
                )

            self.cloudwatch.put_metric_data(
                Namespace=self.namespace, MetricData=metric_data
            )

            logger.info("Metrics emitted to CloudWatch", metrics=metrics)

        except Exception as e:
            logger.error("Failed to emit metrics to CloudWatch", error=str(e))


class RetailDataSchema:
    """Schema definitions for retail data validation"""

    # Pandera schema for retail sales data
    RETAIL_SALES_SCHEMA = DataFrameSchema(
        {
            "date": Column(
                pa.DateTime,
                checks=[
                    Check.greater_than_or_equal_to(pd.Timestamp("2020-01-01")),
                    Check.less_than_or_equal_to(pd.Timestamp.now()),
                ],
                nullable=False,
            ),
            "sales": Column(
                float,
                checks=[
                    Check.greater_than_or_equal_to(0),
                    Check.less_than(1000000),  # Reasonable upper bound
                ],
                nullable=False,
            ),
            "items": Column(
                int,
                checks=[
                    Check.greater_than_or_equal_to(0),
                    Check.less_than(10000),  # Reasonable upper bound
                ],
                nullable=False,
            ),
            "store_id": Column(
                str,
                checks=[
                    Check.str_length(min_value=1, max_value=50),
                    Check.str_matches(r"^[A-Za-z0-9_-]+$"),
                ],
                nullable=False,
            ),
            "product_category": Column(
                str,
                checks=[
                    Check.str_length(min_value=1, max_value=100),
                    Check.isin(
                        ["Electronics", "Clothing", "Food", "Home", "Beauty", "Sports"]
                    ),
                ],
                nullable=False,
            ),
        }
    )

    # Additional schema for processed data
    PROCESSED_DATA_SCHEMA = DataFrameSchema(
        {
            "date": Column(pa.DateTime, nullable=False),
            "sales": Column(float, nullable=False),
            "items": Column(int, nullable=False),
            "store_id": Column(str, nullable=False),
            "product_category": Column(str, nullable=False),
            "year": Column(int, nullable=False),
            "month": Column(int, nullable=False),
            "day": Column(int, nullable=False),
            "is_weekend": Column(bool, nullable=False),
            "sales_normalized": Column(float, nullable=False),
        }
    )


class DataValidator:
    """Main data validation class integrating Great Expectations and Pandera"""

    def __init__(
        self,
        ge_data_context_root: str = "app/etl/ge_data_context",
        s3_bucket: str = "retail-data-validation-reports",
        fail_on_validation_error: bool = True,
    ):

        self.ge_data_context_root = ge_data_context_root
        self.s3_bucket = s3_bucket
        self.fail_on_validation_error = fail_on_validation_error
        self.s3_client = boto3.client("s3")
        self.metrics = DataQualityMetrics()

        # Initialize Great Expectations context
        self._initialize_ge_context()

        # Initialize schemas
        self.schemas = RetailDataSchema()

    def _initialize_ge_context(self):
        """Initialize Great Expectations data context"""
        try:
            # Create GE context directory if it doesn't exist
            os.makedirs(self.ge_data_context_root, exist_ok=True)

            # Try to load existing context or create new one
            try:
                self.ge_context = gx.get_context(
                    context_root_dir=self.ge_data_context_root
                )
                logger.info("Loaded existing Great Expectations context")
            except DataContextError:
                # Create new context
                self.ge_context = gx.get_context(
                    context_root_dir=self.ge_data_context_root
                )
                logger.info("Created new Great Expectations context")

        except Exception as e:
            logger.error(
                "Failed to initialize Great Expectations context", error=str(e)
            )
            raise DataValidationError(f"Failed to initialize GE context: {str(e)}")

    def calculate_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics"""
        metrics = {}

        # Basic metrics
        metrics["total_rows"] = len(df)
        metrics["total_columns"] = len(df.columns)

        # Null value metrics
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_ratio = null_count / len(df) if len(df) > 0 else 0
            metrics[f"{col}_null_count"] = null_count
            metrics[f"{col}_null_ratio"] = null_ratio

        # Overall null metrics
        total_nulls = df.isnull().sum().sum()
        total_values = len(df) * len(df.columns)
        metrics["overall_null_ratio"] = (
            total_nulls / total_values if total_values > 0 else 0
        )

        # Duplicate rows
        metrics["duplicate_rows"] = df.duplicated().sum()
        metrics["duplicate_ratio"] = (
            metrics["duplicate_rows"] / len(df) if len(df) > 0 else 0
        )

        # Data type compliance
        expected_dtypes = {
            "date": "datetime64[ns]",
            "sales": "float64",
            "items": "int64",
            "store_id": "object",
            "product_category": "object",
        }

        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                metrics[f"{col}_dtype_match"] = (
                    1 if actual_dtype == expected_dtype else 0
                )

        # Business rule metrics
        if "sales" in df.columns:
            metrics["negative_sales_count"] = (df["sales"] < 0).sum()
            metrics["zero_sales_count"] = (df["sales"] == 0).sum()
            metrics["extreme_sales_count"] = (df["sales"] > 100000).sum()

        if "items" in df.columns:
            metrics["negative_items_count"] = (df["items"] < 0).sum()
            metrics["zero_items_count"] = (df["items"] == 0).sum()
            metrics["extreme_items_count"] = (df["items"] > 1000).sum()

        return metrics

    def validate_with_pandera(
        self, df: pd.DataFrame, schema: DataFrameSchema
    ) -> Dict[str, Any]:
        """Validate DataFrame using Pandera schema"""
        validation_result = {"passed": False, "errors": [], "metrics": {}}

        try:
            # Validate schema
            schema.validate(df, lazy=True)
            validation_result["passed"] = True
            logger.info("Pandera validation passed")

        except SchemaErrors as e:
            validation_result["errors"] = [str(error) for error in e.schema_errors]
            logger.error(
                "Pandera validation failed", errors=validation_result["errors"]
            )

            if self.fail_on_validation_error:
                raise SchemaValidationError(
                    f"Schema validation failed: {validation_result['errors']}"
                )

        except Exception as e:
            validation_result["errors"] = [str(e)]
            logger.error("Unexpected error during Pandera validation", error=str(e))

            if self.fail_on_validation_error:
                raise DataValidationError(f"Validation error: {str(e)}")

        return validation_result

    def validate_with_great_expectations(
        self, df: pd.DataFrame, suite_name: str = "retail_data_suite"
    ) -> Dict[str, Any]:
        """Validate DataFrame using Great Expectations"""
        validation_result = {
            "passed": False,
            "errors": [],
            "metrics": {},
            "ge_results": None,
        }

        try:
            # Create or get existing expectation suite
            try:
                suite = self.ge_context.get_expectation_suite(suite_name)
            except Exception:
                suite = self.ge_context.add_expectation_suite(suite_name)
                self._create_retail_expectations(suite)

            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="runtime_data_connector",
                data_asset_name="retail_data",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"timestamp": datetime.now().isoformat()},
            )

            # Get validator
            validator = self.ge_context.get_validator(
                batch_request=batch_request, expectation_suite=suite
            )

            # Run validation
            results = validator.validate()
            validation_result["ge_results"] = results
            validation_result["passed"] = results.success

            # Extract errors
            if not results.success:
                for result in results.results:
                    if not result.success:
                        validation_result["errors"].append(
                            f"Expectation {result.expectation_config.expectation_type} "
                            f"failed: {result.result}"
                        )

            logger.info(
                "Great Expectations validation completed",
                success=results.success,
                statistics=results.statistics,
            )

        except Exception as e:
            validation_result["errors"] = [str(e)]
            logger.error("Great Expectations validation failed", error=str(e))

            if self.fail_on_validation_error:
                raise DataValidationError(f"GE validation error: {str(e)}")

        return validation_result

    def _create_retail_expectations(self, suite):
        """Create Great Expectations for retail data"""
        expectations = [
            # Table expectations
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1, "max_value": 1000000},
            },
            {
                "expectation_type": "expect_table_columns_to_match_ordered_list",
                "kwargs": {
                    "column_list": [
                        "date",
                        "sales",
                        "items",
                        "store_id",
                        "product_category",
                    ]
                },
            },
            # Column existence expectations
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "date"},
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "sales"},
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "items"},
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "store_id"},
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "product_category"},
            },
            # Data type expectations
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "date"},
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "sales"},
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "items"},
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "store_id"},
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "product_category"},
            },
            # Business rule expectations
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "sales", "min_value": 0, "max_value": 1000000},
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "items", "min_value": 0, "max_value": 10000},
            },
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {
                    "column": "product_category",
                    "value_set": [
                        "Electronics",
                        "Clothing",
                        "Food",
                        "Home",
                        "Beauty",
                        "Sports",
                    ],
                },
            },
            {
                "expectation_type": "expect_column_values_to_match_regex",
                "kwargs": {"column": "store_id", "regex": "^[A-Za-z0-9_-]+$"},
            },
            # Statistical expectations
            {
                "expectation_type": "expect_column_mean_to_be_between",
                "kwargs": {"column": "sales", "min_value": 0, "max_value": 50000},
            },
            {
                "expectation_type": "expect_column_stdev_to_be_between",
                "kwargs": {"column": "sales", "min_value": 0, "max_value": 100000},
            },
        ]

        for expectation in expectations:
            suite.add_expectation(expectation)

    def detect_schema_drift(
        self, df: pd.DataFrame, reference_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect schema drift by comparing current data with reference schema"""
        drift_result = {"has_drift": False, "drift_details": {}, "current_schema": {}}

        # Get current schema
        current_schema = {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "shape": df.shape,
        }

        drift_result["current_schema"] = current_schema

        # Check for column differences
        reference_columns = set(reference_schema.get("columns", []))
        current_columns = set(current_schema["columns"])

        missing_columns = reference_columns - current_columns
        extra_columns = current_columns - reference_columns

        if missing_columns or extra_columns:
            drift_result["has_drift"] = True
            drift_result["drift_details"]["missing_columns"] = list(missing_columns)
            drift_result["drift_details"]["extra_columns"] = list(extra_columns)

        # Check for data type changes
        reference_dtypes = reference_schema.get("dtypes", {})
        dtype_changes = {}

        for col in current_columns.intersection(reference_columns):
            if col in reference_dtypes:
                if current_schema["dtypes"][col] != reference_dtypes[col]:
                    dtype_changes[col] = {
                        "old": reference_dtypes[col],
                        "new": current_schema["dtypes"][col],
                    }

        if dtype_changes:
            drift_result["has_drift"] = True
            drift_result["drift_details"]["dtype_changes"] = dtype_changes

        return drift_result

    def save_validation_report(
        self, validation_results: Dict[str, Any], report_name: str
    ):
        """Save validation report to S3"""
        try:
            timestamp = datetime.now().isoformat()
            report_key = f"validation_reports/{report_name}_{timestamp}.json"

            # Add metadata
            report_with_metadata = {
                "timestamp": timestamp,
                "report_name": report_name,
                "validation_results": validation_results,
            }

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=report_key,
                Body=json.dumps(report_with_metadata, indent=2, default=str),
                ContentType="application/json",
            )

            logger.info(
                "Validation report saved to S3", bucket=self.s3_bucket, key=report_key
            )

        except Exception as e:
            logger.error("Failed to save validation report to S3", error=str(e))

    def validate_data(
        self, df: pd.DataFrame, dataset_name: str = "retail_data"
    ) -> Dict[str, Any]:
        """Comprehensive data validation using both Pandera and Great Expectations"""
        logger.info(
            "Starting comprehensive data validation",
            dataset=dataset_name,
            shape=df.shape,
        )

        # Convert to pandas if it's a Polars DataFrame
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        validation_results = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "data_shape": df.shape,
            "overall_success": True,
            "pandera_results": {},
            "ge_results": {},
            "quality_metrics": {},
            "schema_drift": {},
        }

        try:
            # Calculate quality metrics
            quality_metrics = self.calculate_data_quality_metrics(df)
            validation_results["quality_metrics"] = quality_metrics

            # Emit metrics to CloudWatch
            self.metrics.emit_metrics(
                quality_metrics, dimensions={"dataset": dataset_name}
            )

            # Pandera validation
            pandera_results = self.validate_with_pandera(
                df, self.schemas.RETAIL_SALES_SCHEMA
            )
            validation_results["pandera_results"] = pandera_results

            if not pandera_results["passed"]:
                validation_results["overall_success"] = False

            # Great Expectations validation
            ge_results = self.validate_with_great_expectations(
                df, f"{dataset_name}_suite"
            )
            validation_results["ge_results"] = ge_results

            if not ge_results["passed"]:
                validation_results["overall_success"] = False

            # Schema drift detection (using a reference schema)
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

            drift_results = self.detect_schema_drift(df, reference_schema)
            validation_results["schema_drift"] = drift_results

            if drift_results["has_drift"]:
                validation_results["overall_success"] = False
                logger.warning(
                    "Schema drift detected",
                    drift_details=drift_results["drift_details"],
                )

                if self.fail_on_validation_error:
                    raise SchemaValidationError(
                        f"Schema drift detected: {drift_results['drift_details']}"
                    )

            # Save validation report
            self.save_validation_report(validation_results, dataset_name)

            logger.info(
                "Data validation completed",
                success=validation_results["overall_success"],
                dataset=dataset_name,
            )

        except Exception as e:
            validation_results["overall_success"] = False
            validation_results["error"] = str(e)
            logger.error("Data validation failed", error=str(e), dataset=dataset_name)

            if self.fail_on_validation_error:
                raise

        return validation_results


def setup_logging():
    """Setup structured logging with structlog"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

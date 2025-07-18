"""
Enhanced ETL Pipeline with integrated data validation, quality checks, and logging.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict
from urllib.request import urlopen

import boto3
import pandas as pd
import structlog

from .validation import (
    DataValidationError,
    DataValidator,
    setup_logging,
)

# Initialize structured logging
setup_logging()
logger = structlog.get_logger()


class ETLPipeline:
    """Enhanced ETL Pipeline with comprehensive validation and monitoring"""

    def __init__(
        self,
        config: Dict[str, Any],
        fail_fast: bool = True,
        enable_validation: bool = True,
    ):
        """
        Initialize ETL Pipeline

        Args:
            config: Configuration dictionary containing S3 buckets, paths, etc.
            fail_fast: Whether to fail immediately on validation errors
            enable_validation: Whether to enable data validation
        """
        self.config = config
        self.fail_fast = fail_fast
        self.enable_validation = enable_validation

        # Initialize AWS clients
        self.s3_client = boto3.client("s3")
        self.cloudwatch = boto3.client("cloudwatch")

        # Initialize validator if enabled
        if self.enable_validation:
            self.validator = DataValidator(
                ge_data_context_root=config.get(
                    "ge_context_root", "app/etl/ge_data_context"
                ),
                s3_bucket=config.get(
                    "validation_bucket", "retail-data-validation-reports"
                ),
                fail_on_validation_error=fail_fast,
            )

        # ETL metrics
        self.etl_metrics = {}

        logger.info(
            "ETL Pipeline initialized",
            fail_fast=fail_fast,
            enable_validation=enable_validation,
        )

    def emit_etl_metrics(self, metrics: Dict[str, Any], stage: str):
        """Emit ETL metrics to CloudWatch"""
        try:
            metric_data = []

            for metric_name, value in metrics.items():
                metric_data.append(
                    {
                        "MetricName": metric_name,
                        "Value": float(value),
                        "Unit": "Count"
                        if "count" in metric_name.lower()
                        else "Seconds"
                        if "time" in metric_name.lower()
                        else "None",
                        "Dimensions": [
                            {"Name": "ETL_Stage", "Value": stage},
                            {"Name": "Pipeline", "Value": "RetailDataPipeline"},
                        ],
                    }
                )

            self.cloudwatch.put_metric_data(
                Namespace="ETL/Pipeline", MetricData=metric_data
            )

            logger.info("ETL metrics emitted", stage=stage, metrics=metrics)

        except Exception as e:
            logger.error("Failed to emit ETL metrics", error=str(e), stage=stage)

    def download_data(self, url: str) -> str:
        """Download data from URL with monitoring"""
        stage = "download"
        start_time = time.time()

        try:
            logger.info("Starting data download", url=url, stage=stage)

            response = urlopen(url)
            data = response.read().decode("utf-8")

            # Calculate metrics
            download_time = time.time() - start_time
            data_size = len(data.encode("utf-8"))

            metrics = {
                "download_time_seconds": download_time,
                "data_size_bytes": data_size,
                "download_success": 1,
            }

            self.emit_etl_metrics(metrics, stage)

            logger.info(
                "Data download completed",
                stage=stage,
                download_time=download_time,
                data_size=data_size,
            )

            return data

        except Exception as e:
            # Emit failure metrics
            failure_metrics = {
                "download_time_seconds": time.time() - start_time,
                "download_success": 0,
                "download_failures": 1,
            }

            self.emit_etl_metrics(failure_metrics, stage)

            logger.error("Data download failed", error=str(e), stage=stage)
            raise

    def validate_raw_data(
        self, df: pd.DataFrame, dataset_name: str = "raw_data"
    ) -> Dict[str, Any]:
        """Validate raw data with comprehensive checks"""
        stage = "validation"
        start_time = time.time()

        try:
            logger.info("Starting data validation", dataset=dataset_name, stage=stage)

            if not self.enable_validation:
                logger.info("Validation disabled, skipping", stage=stage)
                return {"validation_skipped": True}

            # Run comprehensive validation
            validation_results = self.validator.validate_data(df, dataset_name)

            # Calculate validation metrics
            validation_time = time.time() - start_time

            metrics = {
                "validation_time_seconds": validation_time,
                "validation_success": 1 if validation_results["overall_success"] else 0,
                "validation_failures": 0
                if validation_results["overall_success"]
                else 1,
                "total_rows_validated": validation_results["data_shape"][0],
                "total_columns_validated": validation_results["data_shape"][1],
            }

            # Add quality metrics
            quality_metrics = validation_results.get("quality_metrics", {})
            for metric_name, value in quality_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"quality_{metric_name}"] = value

            self.emit_etl_metrics(metrics, stage)

            logger.info(
                "Data validation completed",
                success=validation_results["overall_success"],
                validation_time=validation_time,
                stage=stage,
            )

            return validation_results

        except Exception as e:
            # Emit failure metrics
            failure_metrics = {
                "validation_time_seconds": time.time() - start_time,
                "validation_success": 0,
                "validation_failures": 1,
            }

            self.emit_etl_metrics(failure_metrics, stage)

            logger.error("Data validation failed", error=str(e), stage=stage)

            if self.fail_fast:
                raise

            return {"validation_failed": True, "error": str(e)}

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data with monitoring and validation"""
        stage = "transformation"
        start_time = time.time()

        try:
            logger.info(
                "Starting data transformation", input_shape=df.shape, stage=stage
            )

            # Ensure required columns exist
            required_columns = [
                "date",
                "sales",
                "items",
                "store_id",
                "product_category",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Data type conversions with error handling
            try:
                df["date"] = pd.to_datetime(df["date"])
                df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
                df["items"] = pd.to_numeric(df["items"], errors="coerce").astype(
                    "Int64"
                )
                df["store_id"] = df["store_id"].astype(str)
                df["product_category"] = df["product_category"].astype(str)
            except Exception as e:
                logger.error("Data type conversion failed", error=str(e), stage=stage)
                raise

            # Handle missing values
            initial_rows = len(df)
            df = df.dropna(
                subset=["date", "sales", "items", "store_id", "product_category"]
            )
            rows_after_cleaning = len(df)

            # Create additional features
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6])

            # Normalize sales values
            if df["sales"].std() > 0:
                df["sales_normalized"] = (df["sales"] - df["sales"].mean()) / df[
                    "sales"
                ].std()
            else:
                df["sales_normalized"] = 0.0

            # Calendar completion (fill missing dates)
            # First remove any duplicate dates by grouping and aggregating
            df_grouped = (
                df.groupby(["date", "store_id", "product_category"])
                .agg(
                    {
                        "sales": "sum",
                        "items": "sum",
                        "year": "first",
                        "month": "first",
                        "day": "first",
                        "is_weekend": "first",
                        "sales_normalized": "mean",
                    }
                )
                .reset_index()
            )

            # Set date as index for calendar completion
            df_grouped.set_index("date", inplace=True)

            # Only perform calendar completion if we have a good date range
            try:
                df_grouped = df_grouped.asfreq("D", method="ffill")
                df_grouped.reset_index(inplace=True)
                df = df_grouped
            except ValueError as e:
                # If calendar completion fails, just use the grouped data
                logger.warning(
                    f"Calendar completion failed: {str(e)}, using grouped data"
                )
                df_grouped.reset_index(inplace=True)
                df = df_grouped

            # Calculate transformation metrics
            transformation_time = time.time() - start_time

            metrics = {
                "transformation_time_seconds": transformation_time,
                "input_rows": initial_rows,
                "output_rows": len(df),
                "rows_dropped": initial_rows - rows_after_cleaning,
                "transformation_success": 1,
                "features_added": 5,  # year, month, day, is_weekend, sales_normalized
            }

            self.emit_etl_metrics(metrics, stage)

            logger.info(
                "Data transformation completed",
                input_shape=(initial_rows, len(df.columns) - 5),
                output_shape=df.shape,
                transformation_time=transformation_time,
                stage=stage,
            )

            return df

        except Exception as e:
            # Emit failure metrics
            failure_metrics = {
                "transformation_time_seconds": time.time() - start_time,
                "transformation_success": 0,
                "transformation_failures": 1,
            }

            self.emit_etl_metrics(failure_metrics, stage)

            logger.error("Data transformation failed", error=str(e), stage=stage)
            raise

    def validate_transformed_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate transformed data"""
        stage = "post_transformation_validation"
        start_time = time.time()

        try:
            logger.info("Starting transformed data validation", stage=stage)

            if not self.enable_validation:
                logger.info("Validation disabled, skipping", stage=stage)
                return {"validation_skipped": True}

            # Use processed data schema for validation
            validation_results = self.validator.validate_with_pandera(
                df, self.validator.schemas.PROCESSED_DATA_SCHEMA
            )

            # Calculate quality metrics
            quality_metrics = self.validator.calculate_data_quality_metrics(df)

            # Emit metrics
            validation_time = time.time() - start_time

            metrics = {
                "post_transform_validation_time_seconds": validation_time,
                "post_transform_validation_success": 1
                if validation_results["passed"]
                else 0,
                "post_transform_validation_failures": 0
                if validation_results["passed"]
                else 1,
            }

            # Add quality metrics
            for metric_name, value in quality_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"post_transform_quality_{metric_name}"] = value

            self.emit_etl_metrics(metrics, stage)

            logger.info(
                "Transformed data validation completed",
                success=validation_results["passed"],
                validation_time=validation_time,
                stage=stage,
            )

            return validation_results

        except Exception as e:
            # Emit failure metrics
            failure_metrics = {
                "post_transform_validation_time_seconds": time.time() - start_time,
                "post_transform_validation_success": 0,
                "post_transform_validation_failures": 1,
            }

            self.emit_etl_metrics(failure_metrics, stage)

            logger.error(
                "Transformed data validation failed", error=str(e), stage=stage
            )

            if self.fail_fast:
                raise

            return {"validation_failed": True, "error": str(e)}

    def save_data(
        self, df: pd.DataFrame, output_path: str, s3_key: str
    ) -> Dict[str, Any]:
        """Save data to local storage and S3 with monitoring"""
        stage = "save"
        start_time = time.time()

        try:
            logger.info(
                "Starting data save",
                output_path=output_path,
                s3_key=s3_key,
                stage=stage,
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save to local parquet
            df.to_parquet(output_path, index=False)

            # Get file size
            file_size = os.path.getsize(output_path)

            # Upload to S3
            self.s3_client.upload_file(
                output_path, self.config["output_bucket"], s3_key
            )

            # Calculate save metrics
            save_time = time.time() - start_time

            metrics = {
                "save_time_seconds": save_time,
                "file_size_bytes": file_size,
                "save_success": 1,
                "rows_saved": len(df),
                "columns_saved": len(df.columns),
            }

            self.emit_etl_metrics(metrics, stage)

            logger.info(
                "Data save completed",
                output_path=output_path,
                s3_key=s3_key,
                file_size=file_size,
                save_time=save_time,
                stage=stage,
            )

            return {
                "local_path": output_path,
                "s3_key": s3_key,
                "file_size": file_size,
                "rows_saved": len(df),
                "success": True,
            }

        except Exception as e:
            # Emit failure metrics
            failure_metrics = {
                "save_time_seconds": time.time() - start_time,
                "save_success": 0,
                "save_failures": 1,
            }

            self.emit_etl_metrics(failure_metrics, stage)

            logger.error("Data save failed", error=str(e), stage=stage)
            raise

    def run_pipeline(
        self, data_url: str, output_path: str, s3_key: str
    ) -> Dict[str, Any]:
        """Run the complete ETL pipeline with validation and monitoring"""
        pipeline_start_time = time.time()

        logger.info(
            "Starting ETL pipeline",
            data_url=data_url,
            output_path=output_path,
            s3_key=s3_key,
        )

        pipeline_results = {
            "pipeline_start_time": datetime.now().isoformat(),
            "stages": {},
            "overall_success": True,
            "pipeline_metrics": {},
        }

        try:
            # Stage 1: Download data
            raw_data = self.download_data(data_url)
            pipeline_results["stages"]["download"] = {"success": True}

            # Stage 2: Parse data
            df = pd.read_csv(pd.io.common.StringIO(raw_data))

            # Stage 3: Validate raw data
            validation_results = self.validate_raw_data(df, "raw_retail_data")
            pipeline_results["stages"]["raw_validation"] = validation_results

            if validation_results.get("validation_failed") and self.fail_fast:
                raise DataValidationError("Raw data validation failed")

            # Stage 4: Transform data
            transformed_df = self.transform_data(df)
            pipeline_results["stages"]["transformation"] = {"success": True}

            # Stage 5: Validate transformed data
            transform_validation = self.validate_transformed_data(transformed_df)
            pipeline_results["stages"]["transform_validation"] = transform_validation

            if transform_validation.get("validation_failed") and self.fail_fast:
                raise DataValidationError("Transformed data validation failed")

            # Stage 6: Save data
            save_results = self.save_data(transformed_df, output_path, s3_key)
            pipeline_results["stages"]["save"] = save_results

            # Calculate overall pipeline metrics
            pipeline_time = time.time() - pipeline_start_time

            pipeline_metrics = {
                "pipeline_total_time_seconds": pipeline_time,
                "pipeline_success": 1,
                "pipeline_failures": 0,
                "total_rows_processed": len(transformed_df),
                "total_columns_processed": len(transformed_df.columns),
            }

            self.emit_etl_metrics(pipeline_metrics, "pipeline_overall")

            pipeline_results["pipeline_metrics"] = pipeline_metrics
            pipeline_results["pipeline_end_time"] = datetime.now().isoformat()

            logger.info(
                "ETL pipeline completed successfully",
                pipeline_time=pipeline_time,
                total_rows=len(transformed_df),
            )

            return pipeline_results

        except Exception as e:
            # Emit failure metrics
            pipeline_time = time.time() - pipeline_start_time
            failure_metrics = {
                "pipeline_total_time_seconds": pipeline_time,
                "pipeline_success": 0,
                "pipeline_failures": 1,
            }

            self.emit_etl_metrics(failure_metrics, "pipeline_overall")

            pipeline_results["overall_success"] = False
            pipeline_results["error"] = str(e)
            pipeline_results["pipeline_end_time"] = datetime.now().isoformat()

            logger.error(
                "ETL pipeline failed", error=str(e), pipeline_time=pipeline_time
            )

            if self.fail_fast:
                raise

            return pipeline_results


def create_sample_data():
    """Create sample retail data for testing"""
    import random
    from datetime import datetime, timedelta

    # Generate sample data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    data = []
    stores = ["STORE_001", "STORE_002", "STORE_003", "STORE_004", "STORE_005"]
    categories = ["Electronics", "Clothing", "Food", "Home", "Beauty", "Sports"]

    current_date = start_date
    while current_date <= end_date:
        for store in stores:
            for category in categories:
                if random.random() > 0.3:  # 70% chance of having data
                    data.append(
                        {
                            "date": current_date.strftime("%Y-%m-%d"),
                            "sales": round(random.uniform(100, 5000), 2),
                            "items": random.randint(1, 100),
                            "store_id": store,
                            "product_category": category,
                        }
                    )
        current_date += timedelta(days=1)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv("sample_retail_data.csv", index=False)

    print(f"Created sample data with {len(df)} rows")
    return df


if __name__ == "__main__":
    # Configuration
    config = {
        "output_bucket": "retail-data-processed",
        "validation_bucket": "retail-data-validation-reports",
        "ge_context_root": "app/etl/ge_data_context",
    }

    # Create sample data if it doesn't exist
    if not os.path.exists("sample_retail_data.csv"):
        create_sample_data()

    # Initialize pipeline
    pipeline = ETLPipeline(config=config, fail_fast=True, enable_validation=True)

    # Run pipeline
    try:
        results = pipeline.run_pipeline(
            data_url="file://sample_retail_data.csv",  # Use local file for testing
            output_path="data/processed/retail_data.parquet",
            s3_key="processed/retail_data.parquet",
        )

        print("Pipeline completed successfully!")
        print(f"Results: {json.dumps(results, indent=2, default=str)}")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise

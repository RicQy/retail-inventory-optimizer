"""
Configuration settings for ETL pipeline validation and logging.
"""

import os
from typing import Dict, Any

class ETLConfig:
    """Configuration class for ETL pipeline settings"""
    
    # AWS Configuration
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    
    # S3 Buckets
    OUTPUT_BUCKET = os.getenv('ETL_OUTPUT_BUCKET', 'retail-data-processed')
    VALIDATION_BUCKET = os.getenv('ETL_VALIDATION_BUCKET', 'retail-data-validation-reports')
    BRONZE_BUCKET = os.getenv('ETL_BRONZE_BUCKET', 'retail-data-bronze')
    SILVER_BUCKET = os.getenv('ETL_SILVER_BUCKET', 'retail-data-silver')
    GOLD_BUCKET = os.getenv('ETL_GOLD_BUCKET', 'retail-data-gold')
    
    # Great Expectations Configuration
    GE_CONTEXT_ROOT = os.getenv('GE_CONTEXT_ROOT', 'app/etl/ge_data_context')
    GE_ENABLE_VALIDATION = os.getenv('GE_ENABLE_VALIDATION', 'true').lower() == 'true'
    
    # Validation Configuration
    FAIL_FAST_ON_VALIDATION = os.getenv('FAIL_FAST_ON_VALIDATION', 'true').lower() == 'true'
    ENABLE_SCHEMA_DRIFT_DETECTION = os.getenv('ENABLE_SCHEMA_DRIFT_DETECTION', 'true').lower() == 'true'
    
    # CloudWatch Configuration
    CLOUDWATCH_NAMESPACE_ETL = os.getenv('CLOUDWATCH_NAMESPACE_ETL', 'ETL/Pipeline')
    CLOUDWATCH_NAMESPACE_QUALITY = os.getenv('CLOUDWATCH_NAMESPACE_QUALITY', 'ETL/DataQuality')
    ENABLE_CLOUDWATCH_METRICS = os.getenv('ENABLE_CLOUDWATCH_METRICS', 'true').lower() == 'true'
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', 'json')  # json or text
    
    # Data Quality Thresholds
    MAX_NULL_RATIO = float(os.getenv('MAX_NULL_RATIO', '0.1'))  # 10% max null values
    MAX_DUPLICATE_RATIO = float(os.getenv('MAX_DUPLICATE_RATIO', '0.05'))  # 5% max duplicates
    MIN_ROWS_THRESHOLD = int(os.getenv('MIN_ROWS_THRESHOLD', '1'))
    MAX_ROWS_THRESHOLD = int(os.getenv('MAX_ROWS_THRESHOLD', '1000000'))
    
    # Retail Data Specific Configuration
    RETAIL_DATA_SCHEMA = {
        'required_columns': ['date', 'sales', 'items', 'store_id', 'product_category'],
        'data_types': {
            'date': 'datetime64[ns]',
            'sales': 'float64',
            'items': 'int64',
            'store_id': 'object',
            'product_category': 'object'
        },
        'business_rules': {
            'sales_min': 0,
            'sales_max': 1000000,
            'items_min': 0,
            'items_max': 10000,
            'valid_categories': ['Electronics', 'Clothing', 'Food', 'Home', 'Beauty', 'Sports']
        }
    }
    
    # File Processing Configuration
    OUTPUT_FORMAT = os.getenv('OUTPUT_FORMAT', 'parquet')  # parquet, csv, json
    ENABLE_COMPRESSION = os.getenv('ENABLE_COMPRESSION', 'true').lower() == 'true'
    COMPRESSION_TYPE = os.getenv('COMPRESSION_TYPE', 'snappy')
    
    # Retry Configuration
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))  # seconds
    
    # Performance Configuration
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10000'))
    PARALLEL_PROCESSING = os.getenv('PARALLEL_PROCESSING', 'true').lower() == 'true'
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    
    @classmethod
    def get_etl_config(cls) -> Dict[str, Any]:
        """Get ETL configuration as dictionary"""
        return {
            'aws_region': cls.AWS_REGION,
            'output_bucket': cls.OUTPUT_BUCKET,
            'validation_bucket': cls.VALIDATION_BUCKET,
            'bronze_bucket': cls.BRONZE_BUCKET,
            'silver_bucket': cls.SILVER_BUCKET,
            'gold_bucket': cls.GOLD_BUCKET,
            'ge_context_root': cls.GE_CONTEXT_ROOT,
            'ge_enable_validation': cls.GE_ENABLE_VALIDATION,
            'fail_fast_on_validation': cls.FAIL_FAST_ON_VALIDATION,
            'enable_schema_drift_detection': cls.ENABLE_SCHEMA_DRIFT_DETECTION,
            'cloudwatch_namespace_etl': cls.CLOUDWATCH_NAMESPACE_ETL,
            'cloudwatch_namespace_quality': cls.CLOUDWATCH_NAMESPACE_QUALITY,
            'enable_cloudwatch_metrics': cls.ENABLE_CLOUDWATCH_METRICS,
            'log_level': cls.LOG_LEVEL,
            'log_format': cls.LOG_FORMAT,
            'max_null_ratio': cls.MAX_NULL_RATIO,
            'max_duplicate_ratio': cls.MAX_DUPLICATE_RATIO,
            'min_rows_threshold': cls.MIN_ROWS_THRESHOLD,
            'max_rows_threshold': cls.MAX_ROWS_THRESHOLD,
            'retail_data_schema': cls.RETAIL_DATA_SCHEMA,
            'output_format': cls.OUTPUT_FORMAT,
            'enable_compression': cls.ENABLE_COMPRESSION,
            'compression_type': cls.COMPRESSION_TYPE,
            'max_retries': cls.MAX_RETRIES,
            'retry_delay': cls.RETRY_DELAY,
            'batch_size': cls.BATCH_SIZE,
            'parallel_processing': cls.PARALLEL_PROCESSING,
            'max_workers': cls.MAX_WORKERS
        }
    
    @classmethod
    def get_validation_config(cls) -> Dict[str, Any]:
        """Get validation-specific configuration"""
        return {
            'ge_context_root': cls.GE_CONTEXT_ROOT,
            'validation_bucket': cls.VALIDATION_BUCKET,
            'fail_fast': cls.FAIL_FAST_ON_VALIDATION,
            'enable_schema_drift_detection': cls.ENABLE_SCHEMA_DRIFT_DETECTION,
            'max_null_ratio': cls.MAX_NULL_RATIO,
            'max_duplicate_ratio': cls.MAX_DUPLICATE_RATIO,
            'min_rows_threshold': cls.MIN_ROWS_THRESHOLD,
            'max_rows_threshold': cls.MAX_ROWS_THRESHOLD,
            'retail_data_schema': cls.RETAIL_DATA_SCHEMA
        }
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'log_level': cls.LOG_LEVEL,
            'log_format': cls.LOG_FORMAT,
            'cloudwatch_namespace_etl': cls.CLOUDWATCH_NAMESPACE_ETL,
            'cloudwatch_namespace_quality': cls.CLOUDWATCH_NAMESPACE_QUALITY,
            'enable_cloudwatch_metrics': cls.ENABLE_CLOUDWATCH_METRICS
        }

# Environment-specific configurations
ENVIRONMENTS = {
    'development': {
        'fail_fast_on_validation': False,
        'enable_cloudwatch_metrics': False,
        'log_level': 'DEBUG',
        'max_retries': 1,
        'output_bucket': 'retail-data-processed-dev',
        'validation_bucket': 'retail-data-validation-reports-dev'
    },
    'staging': {
        'fail_fast_on_validation': True,
        'enable_cloudwatch_metrics': True,
        'log_level': 'INFO',
        'max_retries': 2,
        'output_bucket': 'retail-data-processed-staging',
        'validation_bucket': 'retail-data-validation-reports-staging'
    },
    'production': {
        'fail_fast_on_validation': True,
        'enable_cloudwatch_metrics': True,
        'log_level': 'INFO',
        'max_retries': 3,
        'output_bucket': 'retail-data-processed-prod',
        'validation_bucket': 'retail-data-validation-reports-prod'
    }
}

def get_environment_config(env: str = None) -> Dict[str, Any]:
    """Get configuration for specific environment"""
    if env is None:
        env = os.getenv('ENVIRONMENT', 'development')
    
    base_config = ETLConfig.get_etl_config()
    env_config = ENVIRONMENTS.get(env, {})
    
    # Merge configurations
    base_config.update(env_config)
    return base_config

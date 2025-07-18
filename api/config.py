"""
Configuration module for FastAPI application.

This module handles environment variables, settings, and configuration
for the retail inventory optimization API.
"""

from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application settings
    app_name: str = "Retail Inventory Optimization API"
    app_version: str = "1.0.0"
    debug: bool = False
    app_env: str = "development"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Database settings
    database_url: Optional[str] = None

    # S3 settings
    s3_bucket: str = "default-bucket"
    s3_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Security settings
    secret_key: str = "your-secret-key-here"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # CORS settings
    allowed_origins: List[str] = ["*"]

    # Logging settings
    log_level: str = "INFO"

    # Feature flags
    enable_auth: bool = False
    enable_rate_limiting: bool = True

    # Performance settings
    max_workers: int = 4
    request_timeout: int = 300

    # Forecasting settings
    default_forecast_horizon: int = 30
    max_forecast_horizon: int = 365

    # Optimization settings
    max_optimization_time: int = 300

    model_config = {"env_file": ".env", "case_sensitive": False}


# Global settings instance
settings = Settings()

"""
Configuration module for FastAPI application.

This module handles environment variables, settings, and configuration
for the retail inventory optimization API.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application settings
    app_name: str = Field(default="Retail Inventory Optimization API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    app_env: str = Field(default="development", env="APP_ENV")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database settings
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # S3 settings
    s3_bucket: str = Field(default="default-bucket", env="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", env="S3_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # CORS settings
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Feature flags
    enable_auth: bool = Field(default=False, env="ENABLE_AUTH")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    
    # Performance settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")
    
    # Forecasting settings
    default_forecast_horizon: int = Field(default=30, env="DEFAULT_FORECAST_HORIZON")
    max_forecast_horizon: int = Field(default=365, env="MAX_FORECAST_HORIZON")
    
    # Optimization settings
    max_optimization_time: int = Field(default=300, env="MAX_OPTIMIZATION_TIME")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }


# Global settings instance
settings = Settings()

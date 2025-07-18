import logging
from dataclasses import dataclass
from typing import Any, Optional

import boto3
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet


@dataclass
class ForecastConfig:
    sku: Optional[str] = None
    store_id: Optional[str] = None
    region: Optional[str] = None


class ForecastingService:
    def __init__(self, s3_bucket: str):
        self.s3_client = boto3.client("s3")
        self.s3_bucket = s3_bucket
        logging.basicConfig(level=logging.DEBUG)

    def prepare_data(self, data: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
        # Filter and prepare data
        logging.info("Preparing data based on provided configuration...")
        if config.sku:
            data = data[data["sku"] == config.sku]
        if config.store_id:
            data = data[data["store_id"] == config.store_id]
        if config.region:
            data = data[data["region"] == config.region]
        return data

    def train_prophet(self, data: pd.DataFrame) -> Prophet:
        # Model initialization and fitting
        model = Prophet(country_holidays="US")
        model.fit(data)
        logging.info("Prophet model trained successfully.")
        return model

    def train_arima(self, data: pd.DataFrame) -> auto_arima:
        # ARIMA model training
        model = auto_arima(
            data, seasonal=True, m=12, stepwise=True, suppress_warnings=True
        )
        logging.info("ARIMA model trained successfully.")
        return model

    def forecast(self, model: Prophet, periods: int = 30) -> pd.DataFrame:
        # Generate forecasts
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        logging.info("Forecasting completed.")
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def serialize_to_s3(self, model: Any, key: str) -> None:
        # Serialize and upload model
        import pickle

        serialized_model = pickle.dumps(model)
        self.s3_client.put_object(Bucket=self.s3_bucket, Key=key, Body=serialized_model)
        logging.info(f"Model serialized and saved to S3 with key: {key}")

    def auto_forecast(
        self, data: pd.DataFrame, config: ForecastConfig, periods: int = 30
    ) -> pd.DataFrame:
        data_prepared = self.prepare_data(data, config)
        model = self.train_prophet(data_prepared)
        forecast = self.forecast(model, periods)
        self.serialize_to_s3(model, key=f"{config.sku or 'global'}_model.pkl")
        return forecast

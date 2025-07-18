import os
from urllib.request import urlopen

import boto3
import polars as pl

# Constants
CSV_URL = "https://datahub.io/core/retail-sales/r/retail-sales.csv"
OUTPUT_DIR = "data/clean"
BUCKET_NAME = "<YOUR_BUCKET_NAME>"  # Replace with your actual S3 bucket name

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Step 1: Download the CSV
def download_csv() -> str:
    print("Downloading CSV...")
    response = urlopen(CSV_URL)
    csv_data = response.read().decode("utf-8")
    return csv_data


# Step 2: Validate and Clean the Data
def process_data(csv_data: str) -> pl.DataFrame:
    print("Processing data with Polars...")
    df = pl.read_csv(csv_data)
    # Validate schema
    expected_schema = {"Date": pl.Date, "Sales": pl.Float64}
    for col, dtype in expected_schema.items():
        if df[col].dtype != dtype:
            df = df.with_column(df[col].cast(dtype))
    # Handle nulls
    df = df.fill_null(strategy="zero")
    return df


# Step 3: Save to Parquet and upload to S3
def save_to_parquet_and_s3(df: pl.DataFrame):
    print("Saving to Parquet and uploading to S3...")
    file_path = os.path.join(OUTPUT_DIR, "retail_sales.parquet")
    df.write_parquet(file_path)

    # Upload to S3
    s3 = boto3.client("s3")
    s3.upload_file(file_path, BUCKET_NAME, "retail_sales/retail_sales.parquet")


if __name__ == "__main__":
    csv_data = download_csv()
    df = process_data(csv_data)
    save_to_parquet_and_s3(df)

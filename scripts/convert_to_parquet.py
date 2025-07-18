#!/usr/bin/env python3
"""
Convert CSV retail sales data to Parquet format
"""

from pathlib import Path

import pandas as pd


def convert_csv_to_parquet():
    # Define paths
    base_path = Path(__file__).parent.parent
    csv_path = base_path / "data" / "retail_sales_sample.csv"
    parquet_path = base_path / "data" / "retail_sales_sample.parquet"

    # Read CSV
    df = pd.read_csv(csv_path)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Ensure proper data types
    df["store_id"] = df["store_id"].astype("string")
    df["sku"] = df["sku"].astype("string")
    df["units_sold"] = df["units_sold"].astype("int32")
    df["price"] = df["price"].astype("float64")
    df["on_hand"] = df["on_hand"].astype("int32")

    # Save as Parquet
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

    print(f"Successfully converted {csv_path} to {parquet_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")


if __name__ == "__main__":
    convert_csv_to_parquet()

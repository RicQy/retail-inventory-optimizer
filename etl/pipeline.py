import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import boto3
from datetime import datetime

# Initialize S3 client
s3 = boto3.client('s3')

# Configuration
RAW_DATA_PATH = 'sample_retail_sales.csv'
S3_BUCKET_BRONZE = 'my-bronze-bucket'
S3_BUCKET_SILVER = 'my-silver-bucket'

# Step 1: Ingestion
def ingest_data():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=['date'])
    return df

# Step 2: Transformation
def transform_data(df):
    # Type coercion
    df['sales'] = df['sales'].astype(float)
    df['items'] = df['items'].astype(int)

    # Calendar completion (adding missing dates)
    df.set_index('date', inplace=True)
    df = df.asfreq('D').fillna(method='ffill')

    # Normalize currency (example transformation)
    df['sales'] = df['sales'] * 1.2  # assuming conversion rate

    return df

# Step 3: Writing to Parquet

def write_to_parquet(df, path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, partition_cols=['year', 'month'])

# Step 4: Upload to S3
def upload_to_s3(file_path, bucket, key):
    s3.upload_file(file_path, bucket, key)

# Main pipeline execution
def main():
    df = ingest_data()
    df = transform_data(df)

    # Add year and month columns for partitioning
    df['year'] = df.index.year
    df['month'] = df.index.month

    # File paths
    parquet_path = 'output/cleaned_data.parquet'

    # Write to Parquet
    write_to_parquet(df, parquet_path)

    # Upload to S3
    upload_to_s3(parquet_path, S3_BUCKET_BRONZE, 'cleaned_data/bronze/')
    upload_to_s3(parquet_path, S3_BUCKET_SILVER, 'cleaned_data/silver/')

if __name__ == "__main__":
    main()


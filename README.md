# Retail Inventory Clearance Optimizer

A FastAPI-based system for retail inventory optimization using demand forecasting and optimization algorithms.

## Quick Start

### Local Development

1. **Setup Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the Application**
   
   **Using the convenience script:**
   ```bash
   # On Unix/Linux/Mac:
   chmod +x scripts/launch_local.sh
   ./scripts/launch_local.sh
   
   # On Windows:
   scripts\launch_local.bat
   ```
   
   **Or manually:**
   ```bash
   export APP_ENV=test  # On Windows: set APP_ENV=test
   uvicorn api.main:app --port 8000 --reload
   ```

3. **Access the Application**
   - API Documentation: http://127.0.0.1:8000/docs
   - Health Check: http://127.0.0.1:8000/health
   - Readiness Check: http://127.0.0.1:8000/health/readiness

## API Endpoints

### Health Monitoring
- `GET /health` - Overall health status
- `GET /health/readiness` - Readiness probe
- `GET /health/liveness` - Liveness probe

### Data Management
- `POST /upload-data` - Upload data for ETL processing
- `GET /upload-data/status/{task_id}` - Check ETL job status

### Forecasting
- `GET /forecast` - Generate demand forecasts

### Optimization
- `POST /optimize` - Optimize inventory levels

### Job Management
- `POST /jobs` - Create asynchronous jobs
- `GET /jobs` - List jobs
- `GET /jobs/{job_id}` - Get job status
- `POST /jobs/{job_id}/cancel` - Cancel job
- `POST /jobs/{job_id}/retry` - Retry failed job

## Configuration

The application supports configuration through environment variables:

- `APP_ENV` - Environment mode (development, test, production)
- `APP_NAME` - Application name
- `DEBUG` - Enable debug mode
- `PORT` - Server port (default: 8000)
- `S3_BUCKET` - S3 bucket for data storage
- `S3_REGION` - AWS region (default: us-east-1)
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key

### Test Mode

When `APP_ENV=test`, the application runs in test mode with:
- Dummy AWS credentials
- Mocked AWS services
- Local processing without external dependencies

## Architecture

- **FastAPI**: Web framework for API endpoints
- **AWS S3**: Data storage
- **AWS DynamoDB**: Job metadata storage
- **AWS SQS**: Job queue management
- **AWS Lambda**: Async task processing
- **Prophet**: Time series forecasting
- **Optimization Engine**: Inventory optimization algorithms

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Linting
flake8 .

# Formatting
black .

# Type checking
mypy .
```

### Project Structure
```
retail-inventory-optimizer/
├── api/                    # FastAPI application
│   ├── main.py            # Main application file
│   ├── config.py          # Configuration settings
│   ├── aws_services.py    # AWS service integrations
│   └── ...
├── etl/                   # ETL pipeline
├── forecast/              # Forecasting models
├── optimize/              # Optimization algorithms
├── scripts/               # Utility scripts
│   ├── launch_local.sh    # Unix launch script
│   └── launch_local.bat   # Windows launch script
├── tests/                 # Test suite
└── README.md             # This file
```

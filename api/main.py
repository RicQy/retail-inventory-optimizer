"""
FastAPI application for retail inventory optimization system.

This API provides endpoints for:
- Data upload and ETL pipeline triggering
- Demand forecasting
- Inventory optimization
- Health monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import os
import tempfile
import json
from datetime import datetime, timedelta
import io
import base64

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import structlog
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import our modules
from etl.pipeline import main as etl_main
from forecast.forecasting_service import ForecastingService, ForecastConfig
from optimize.inventory_optimizer import InventoryOptimizer, CostsConfig

# Import API modules
from api.config import settings
from api.pagination import PaginationParams, PaginatedResponse, pagination_params, paginate_list
from api.middleware import (
    RateLimitingMiddleware,
    RequestTrackingMiddleware,
    SecurityHeadersMiddleware,
    CacheControlMiddleware,
    ErrorHandlingMiddleware,
    MetricsMiddleware,
    get_metrics,
    metrics_middleware
)
from api.aws_services import aws_service_manager, JobStatus, JobType
from api.job_orchestration import job_orchestrator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Security
security = HTTPBearer()

# Global variables for services
forecasting_service = None
inventory_optimizer = None

# Background task status tracking
background_tasks_status = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global forecasting_service, inventory_optimizer
    
# Initialize services
    forecasting_service = ForecastingService(settings.s3_bucket)
    inventory_optimizer = InventoryOptimizer()
    
    logger.info("Application started successfully")
    yield
    
    # Cleanup
    logger.info("Application shutting down")

# Create FastAPI app
app = FastAPI(
    title="Retail Inventory Optimization API",
    description="API for demand forecasting and inventory optimization",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "data",
            "description": "Data upload and ETL operations"
        },
        {
            "name": "forecast",
            "description": "Demand forecasting operations"
        },
        {
            "name": "optimization",
            "description": "Inventory optimization operations"
        },
        {
            "name": "jobs",
            "description": "Asynchronous job orchestration and management"
        },
        {
            "name": "health",
            "description": "Health monitoring endpoints"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UploadDataRequest(BaseModel):
    """Request model for data upload via S3 key."""
    s3_key: str = Field(..., description="S3 key for the data file")
    file_type: str = Field(default="csv", description="File type (csv, parquet, etc.)")
    
class OptimizationRequest(BaseModel):
    """Request model for inventory optimization."""
    horizon: int = Field(default=30, description="Forecast horizon in days", ge=1, le=365)
    budget: float = Field(..., description="Available budget", gt=0)
    shelf_space: float = Field(..., description="Available shelf space", gt=0)
    supplier_moq: Dict[str, float] = Field(default_factory=dict, description="Minimum order quantities per SKU")
    holding_cost_per_unit: float = Field(default=1.0, description="Holding cost per unit", ge=0)
    stockout_cost_per_unit: float = Field(default=10.0, description="Stockout cost per unit", ge=0)
    unit_cost: Dict[str, float] = Field(default_factory=dict, description="Unit cost per SKU")
    shelf_space_per_unit: Dict[str, float] = Field(default_factory=dict, description="Shelf space per unit per SKU")
    max_discount: float = Field(default=0.5, description="Maximum discount allowed", ge=0, le=1)
    min_service_level: float = Field(default=0.95, description="Minimum service level", ge=0, le=1)
    
    @validator('horizon')
    def validate_horizon(cls, v):
        if v <= 0:
            raise ValueError('Horizon must be positive')
        return v

class ForecastResponse(BaseModel):
    """Response model for forecast endpoint."""
    forecast_data: List[Dict[str, Any]]
    plot_base64: Optional[str] = None
    metadata: Dict[str, Any]

class OptimizationResponse(BaseModel):
    """Response model for optimization endpoint."""
    status: str
    orders: Dict[str, int]
    discount_schedule: Dict[str, float]
    total_cost: float
    metrics: Dict[str, Any]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    services: Dict[str, str]
    version: str

class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AsyncJobRequest(BaseModel):
    """Request model for creating asynchronous jobs."""
    job_type: str = Field(..., description="Type of job (etl, forecast, optimization, batch_processing)")
    input_data: Dict[str, Any] = Field(..., description="Input data for the job")
    priority: int = Field(default=1, description="Job priority (1-5, higher is more urgent)", ge=1, le=5)

class AsyncJobResponse(BaseModel):
    """Response model for async job creation."""
    job_id: str
    status: str
    created_at: str
    estimated_completion: Optional[str] = None

class JobMetadataResponse(BaseModel):
    """Response model for job metadata."""
    job_id: str
    job_type: str
    status: str
    progress: float
    created_at: str
    updated_at: str
    user_id: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int
    priority: int

class JobListResponse(BaseModel):
    """Response model for job listing."""
    jobs: List[JobMetadataResponse]
    total_count: int
    has_more: bool

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "File not found", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Dependency for authentication (placeholder)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication placeholder."""
    # In production, implement proper JWT validation
    return {"user_id": "default_user"}

# Background task functions
async def run_etl_job(task_id: str, file_path: str = None, s3_key: str = None):
    """Background task for ETL job execution."""
    try:
        background_tasks_status[task_id] = {
            "status": "running",
            "progress": 0.0,
            "result": None,
            "error": None
        }
        
        logger.info(f"Starting ETL job {task_id}")
        
        # Update progress
        background_tasks_status[task_id]["progress"] = 0.1
        
        # Run ETL pipeline
        if file_path:
            # Process uploaded file
            background_tasks_status[task_id]["progress"] = 0.5
            # Mock ETL processing for uploaded file
            await asyncio.sleep(2)  # Simulate processing time
        elif s3_key:
            # Process S3 file
            background_tasks_status[task_id]["progress"] = 0.5
            # Mock ETL processing for S3 file
            await asyncio.sleep(2)  # Simulate processing time
        
        # Complete the task
        background_tasks_status[task_id].update({
            "status": "completed",
            "progress": 1.0,
            "result": {"message": "ETL job completed successfully"}
        })
        
        logger.info(f"ETL job {task_id} completed successfully")
        
    except Exception as e:
        background_tasks_status[task_id].update({
            "status": "failed",
            "progress": 0.0,
            "error": str(e)
        })
        logger.error(f"ETL job {task_id} failed", exc_info=e)

def generate_plot(forecast_data: List[Dict[str, Any]], title: str = "Forecast Plot") -> str:
    """Generate a base64-encoded plot from forecast data."""
    try:
        # Extract data for plotting
        dates = [item['ds'] for item in forecast_data]
        values = [item['yhat'] for item in forecast_data]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, label='Forecast', color='blue')
        
        # Add confidence intervals if available
        if 'yhat_lower' in forecast_data[0] and 'yhat_upper' in forecast_data[0]:
            lower = [item['yhat_lower'] for item in forecast_data]
            upper = [item['yhat_upper'] for item in forecast_data]
            plt.fill_between(dates, lower, upper, alpha=0.3, color='blue')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Forecasted Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return plot_base64
        
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")
        return None

# API Endpoints

@app.post("/upload-data", tags=["data"])
async def upload_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    s3_key: str = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload data and trigger ETL job.
    
    Supports both multipart file upload and S3 key reference.
    """
    if not file and not s3_key:
        raise HTTPException(
            status_code=400,
            detail="Either file upload or S3 key must be provided"
        )
    
    # Generate task ID
    task_id = f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(current_user))}"
    
    try:
        if file:
            # Handle file upload
            if not file.filename.endswith(('.csv', '.parquet', '.json')):
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Only CSV, Parquet, and JSON are supported."
                )
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Start background ETL job
            background_tasks.add_task(run_etl_job, task_id, file_path=temp_file_path)
            
        elif s3_key:
            # Handle S3 key
            background_tasks.add_task(run_etl_job, task_id, s3_key=s3_key)
        
        return {
            "task_id": task_id,
            "status": "initiated",
            "message": "ETL job started successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to upload data: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate ETL job")

@app.get("/upload-data/status/{task_id}", tags=["data"])
async def get_etl_status(task_id: str):
    """Get the status of an ETL job."""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatusResponse(
        task_id=task_id,
        **background_tasks_status[task_id]
    )

@app.get("/forecast", tags=["forecast"], response_model=ForecastResponse)
async def get_forecast(
    store_id: str = Query(..., description="Store ID"),
    sku: str = Query(..., description="SKU"),
    horizon: int = Query(default=30, description="Forecast horizon in days", ge=1, le=365),
    include_plot: bool = Query(default=True, description="Include base64-encoded plot"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get demand forecast for a specific store and SKU.
    
    Returns forecast data and optionally a plot.
    """
    try:
        # Create forecast config
        config = ForecastConfig(
            sku=sku,
            store_id=store_id
        )
        
        # Mock data for demonstration (in production, load from database/S3)
        mock_data = pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'y': [100 + i * 0.5 + (i % 7) * 10 for i in range(100)],
            'sku': [sku] * 100,
            'store_id': [store_id] * 100
        })
        
        # Generate forecast
        forecast_result = forecasting_service.auto_forecast(
            mock_data, 
            config, 
            periods=horizon
        )
        
        # Convert to list of dictionaries
        forecast_data = forecast_result.to_dict('records')
        
        # Generate plot if requested
        plot_base64 = None
        if include_plot:
            plot_base64 = generate_plot(
                forecast_data, 
                title=f"Demand Forecast - Store: {store_id}, SKU: {sku}"
            )
        
        return ForecastResponse(
            forecast_data=forecast_data,
            plot_base64=plot_base64,
            metadata={
                "store_id": store_id,
                "sku": sku,
                "horizon": horizon,
                "generated_at": datetime.now().isoformat(),
                "model_type": "prophet"
            }
        )
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate forecast")

@app.post("/optimize", tags=["optimization"], response_model=OptimizationResponse)
async def optimize_inventory(
    request: OptimizationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Optimize inventory levels based on forecast and constraints.
    
    Accepts optimization parameters and returns recommended orders and discounts.
    """
    try:
        # Create costs configuration
        costs_config = CostsConfig(
            budget=request.budget,
            shelf_space=request.shelf_space,
            supplier_moq=request.supplier_moq,
            holding_cost_per_unit=request.holding_cost_per_unit,
            stockout_cost_per_unit=request.stockout_cost_per_unit,
            unit_cost=request.unit_cost,
            shelf_space_per_unit=request.shelf_space_per_unit,
            max_discount=request.max_discount,
            min_service_level=request.min_service_level
        )
        
        # Mock forecast data for demonstration
        mock_forecast_data = pd.DataFrame({
            'sku': ['SKU001', 'SKU002', 'SKU003'],
            'units_sold': [100, 150, 80],
            'price': [10.0, 15.0, 8.0],
            'on_hand': [50, 30, 120]
        })
        
        # Run optimization
        optimization_result = inventory_optimizer.optimize_inventory(
            mock_forecast_data,
            costs_config
        )
        
        return OptimizationResponse(**optimization_result)
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize inventory")

@app.get("/health", tags=["health"], response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for liveness and readiness probes.
    """
    # Check AWS services health
    aws_health = await aws_service_manager.health_check()
    
    services_status = {
        "forecasting_service": "healthy" if forecasting_service else "unavailable",
        "inventory_optimizer": "healthy" if inventory_optimizer else "unavailable",
        "job_orchestrator": "healthy",  # Assume healthy if no errors
        **aws_health  # Include S3, DynamoDB, SQS, Lambda status
    }
    
    overall_status = "healthy" if all(
        status == "healthy" for status in services_status.values()
    ) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        services=services_status,
        version="1.0.0"
    )

@app.get("/health/liveness", tags=["health"])
async def liveness_check():
    """Liveness probe endpoint."""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

@app.get("/health/readiness", tags=["health"])
async def readiness_check():
    """Readiness probe endpoint."""
    # Check if all required services are initialized
    if not forecasting_service or not inventory_optimizer:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": datetime.now().isoformat()}

# Additional utility endpoints

@app.get("/", tags=["health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Retail Inventory Optimization API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/tasks", tags=["data"])
async def list_tasks(
    limit: int = Query(default=50, description="Maximum number of tasks to return"),
    status: Optional[str] = Query(None, description="Filter by task status")
):
    """List background tasks with optional filtering."""
    tasks = []
    for task_id, task_info in background_tasks_status.items():
        if status and task_info['status'] != status:
            continue
        tasks.append({
            "task_id": task_id,
            **task_info
        })
    
    # Sort by most recent first and apply limit
    tasks.sort(key=lambda x: x['task_id'], reverse=True)
    return {"tasks": tasks[:limit]}

# New Asynchronous Job Orchestration Endpoints

@app.post("/jobs", tags=["jobs"], response_model=AsyncJobResponse)
async def create_async_job(
    request: AsyncJobRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new asynchronous job for processing.
    
    Supports ETL, forecasting, optimization, and batch processing jobs.
    """
    try:
        # Validate job type
        try:
            job_type = JobType(request.job_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid job type: {request.job_type}. Valid types: {[t.value for t in JobType]}"
            )
        
        # Create the job
        job_id = await job_orchestrator.create_job(
            job_type=job_type,
            input_data=request.input_data,
            user_id=current_user["user_id"],
            priority=request.priority
        )
        
        # Get job metadata to return
        job_metadata = await job_orchestrator.get_job_status(job_id)
        
        return AsyncJobResponse(
            job_id=job_id,
            status=job_metadata.status.value,
            created_at=job_metadata.created_at.isoformat(),
            estimated_completion=job_metadata.estimated_completion.isoformat() if job_metadata.estimated_completion else None
        )
        
    except Exception as e:
        logger.error(f"Failed to create async job: {e}")
        raise HTTPException(status_code=500, detail="Failed to create job")

@app.get("/jobs/{job_id}", tags=["jobs"], response_model=JobMetadataResponse)
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get the status and metadata of a specific job.
    """
    try:
        job_metadata = await job_orchestrator.get_job_status(job_id)
        
        if not job_metadata:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check if user has access to this job
        if job_metadata.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return JobMetadataResponse(
            job_id=job_metadata.job_id,
            job_type=job_metadata.job_type.value,
            status=job_metadata.status.value,
            progress=job_metadata.progress,
            created_at=job_metadata.created_at.isoformat(),
            updated_at=job_metadata.updated_at.isoformat(),
            user_id=job_metadata.user_id,
            input_data=job_metadata.input_data,
            output_data=job_metadata.output_data,
            error_message=job_metadata.error_message,
            retry_count=job_metadata.retry_count,
            priority=job_metadata.priority
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job status")

@app.get("/jobs", tags=["jobs"], response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    limit: int = Query(default=50, description="Maximum number of jobs to return", ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """
    List jobs for the current user with optional filtering.
    """
    try:
        # Convert string parameters to enums if provided
        status_filter = None
        if status:
            try:
                status_filter = JobStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid statuses: {[s.value for s in JobStatus]}"
                )
        
        job_type_filter = None
        if job_type:
            try:
                job_type_filter = JobType(job_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid job type: {job_type}. Valid types: {[t.value for t in JobType]}"
                )
        
        # List jobs
        jobs = await job_orchestrator.list_jobs(
            user_id=current_user["user_id"],
            status=status_filter,
            job_type=job_type_filter,
            limit=limit
        )
        
        # Convert to response format
        job_responses = []
        for job in jobs:
            job_responses.append(JobMetadataResponse(
                job_id=job.job_id,
                job_type=job.job_type.value,
                status=job.status.value,
                progress=job.progress,
                created_at=job.created_at.isoformat(),
                updated_at=job.updated_at.isoformat(),
                user_id=job.user_id,
                input_data=job.input_data,
                output_data=job.output_data,
                error_message=job.error_message,
                retry_count=job.retry_count,
                priority=job.priority
            ))
        
        return JobListResponse(
            jobs=job_responses,
            total_count=len(job_responses),
            has_more=len(job_responses) == limit
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to list jobs")

@app.post("/jobs/{job_id}/cancel", tags=["jobs"])
async def cancel_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Cancel a running or queued job.
    """
    try:
        # Check if job exists and user has access
        job_metadata = await job_orchestrator.get_job_status(job_id)
        
        if not job_metadata:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_metadata.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Cancel the job
        success = await job_orchestrator.cancel_job(job_id)
        
        if success:
            return {"message": "Job cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel job")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")

@app.post("/jobs/{job_id}/retry", tags=["jobs"])
async def retry_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retry a failed job.
    """
    try:
        # Check if job exists and user has access
        job_metadata = await job_orchestrator.get_job_status(job_id)
        
        if not job_metadata:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_metadata.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Retry the job
        success = await job_orchestrator.retry_job(job_id)
        
        if success:
            return {"message": "Job retried successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to retry job")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry job: {e}")
        raise HTTPException(status_code=500, detail="Failed to retry job")

@app.get("/jobs/{job_id}/logs", tags=["jobs"])
async def get_job_logs(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get logs for a specific job.
    """
    try:
        # Check if job exists and user has access
        job_metadata = await job_orchestrator.get_job_status(job_id)
        
        if not job_metadata:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_metadata.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get job logs
        logs = await job_orchestrator.get_job_logs(job_id)
        
        return {"job_id": job_id, "logs": logs}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job logs")

@app.get("/jobs/{job_id}/metrics", tags=["jobs"])
async def get_job_metrics(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get metrics for a specific job.
    """
    try:
        # Check if job exists and user has access
        job_metadata = await job_orchestrator.get_job_status(job_id)
        
        if not job_metadata:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_metadata.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get job metrics
        metrics = await job_orchestrator.get_job_metrics(job_id)
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job metrics")

@app.get("/jobs/queue/statistics", tags=["jobs"])
async def get_queue_statistics(
    current_user: dict = Depends(get_current_user)
):
    """
    Get statistics about the job queue.
    """
    try:
        statistics = await job_orchestrator.get_queue_statistics()
        return statistics
        
    except Exception as e:
        logger.error(f"Failed to get queue statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queue statistics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use structlog configuration
    )

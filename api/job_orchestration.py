"""
Job orchestration service for managing asynchronous job processing.

This service handles:
- Job creation and queuing
- Job status tracking
- Error handling and retries
- Lambda function invocation
- Progress monitoring
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import structlog

from api.aws_services import (
    AWSServiceManager,
    JobMetadata,
    JobStatus,
    JobType,
    aws_service_manager,
)

logger = structlog.get_logger()


@dataclass
class JobResult:
    """Job execution result."""

    job_id: str
    status: JobStatus
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    s3_output_path: Optional[str] = None


class JobOrchestrator:
    """Main job orchestration service."""

    def __init__(self, aws_manager: AWSServiceManager = None):
        """Initialize the job orchestrator."""
        self.aws_manager = aws_manager or aws_service_manager
        self.lambda_functions = {
            JobType.ETL: "retail-inventory-etl-processor",
            JobType.FORECAST: "retail-inventory-forecast-processor",
            JobType.OPTIMIZATION: "retail-inventory-optimization-processor",
            JobType.BATCH_PROCESSING: "retail-inventory-batch-processor",
        }

        # Job processors for local execution (fallback)
        self.local_processors = {}

    def register_local_processor(self, job_type: JobType, processor: Callable):
        """Register a local processor for a job type."""
        self.local_processors[job_type] = processor
        logger.info(f"Registered local processor for {job_type.value}")

    async def create_job(
        self,
        job_type: JobType,
        input_data: Dict[str, Any],
        user_id: str,
        priority: int = 1,
    ) -> str:
        """Create a new job and enqueue it for processing."""
        job_id = str(uuid4())
        current_time = datetime.now()

        # Create job metadata
        job_metadata = JobMetadata(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            created_at=current_time,
            updated_at=current_time,
            user_id=user_id,
            input_data=input_data,
            priority=priority,
        )

        try:
            # Save job metadata to DynamoDB
            success = await self.aws_manager.create_job_metadata(job_metadata)
            if not success:
                raise Exception("Failed to save job metadata")

            # Enqueue job for processing
            success = await self.aws_manager.enqueue_job(job_metadata)
            if not success:
                # Update job status to failed
                await self.aws_manager.update_job_metadata(
                    job_id,
                    {
                        "status": JobStatus.FAILED.value,
                        "error_message": "Failed to enqueue job",
                    },
                )
                raise Exception("Failed to enqueue job")

            logger.info(f"Created and enqueued job: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise

    async def process_job(self, job_id: str) -> JobResult:
        """Process a job by invoking the appropriate Lambda function or local processor."""
        try:
            # Get job metadata
            job_metadata = await self.aws_manager.get_job_metadata(job_id)
            if not job_metadata:
                raise Exception(f"Job {job_id} not found")

            # Update job status to running
            await self.aws_manager.update_job_metadata(
                job_id, {"status": JobStatus.RUNNING.value, "progress": 0.1}
            )

            start_time = datetime.now()

            # Prepare Lambda payload
            lambda_payload = {
                "job_id": job_id,
                "job_type": job_metadata.job_type.value,
                "input_data": job_metadata.input_data,
                "user_id": job_metadata.user_id,
                "s3_bucket": self.aws_manager.bucket_name,
            }

            # Try to invoke Lambda function
            lambda_function_name = self.lambda_functions.get(job_metadata.job_type)

            if lambda_function_name:
                try:
                    # Invoke Lambda asynchronously
                    lambda_result = await self.aws_manager.invoke_lambda_function(
                        lambda_function_name, lambda_payload, asynchronous=True
                    )

                    if lambda_result:
                        logger.info(f"Lambda function invoked for job {job_id}")
                        # The Lambda function will update the job status
                        return JobResult(
                            job_id=job_id,
                            status=JobStatus.RUNNING,
                            output_data={
                                "lambda_request_id": lambda_result.get("request_id")
                            },
                        )
                    else:
                        raise Exception("Lambda invocation failed")

                except Exception as e:
                    logger.warning(f"Lambda invocation failed for job {job_id}: {e}")
                    # Fall back to local processing

            # Local processing fallback
            if job_metadata.job_type in self.local_processors:
                processor = self.local_processors[job_metadata.job_type]

                # Update progress
                await self.aws_manager.update_job_metadata(job_id, {"progress": 0.5})

                # Execute local processor
                result = await processor(job_metadata.input_data)

                execution_time = (datetime.now() - start_time).total_seconds()

                # Update job status to completed
                await self.aws_manager.update_job_metadata(
                    job_id,
                    {
                        "status": JobStatus.COMPLETED.value,
                        "progress": 1.0,
                        "output_data": result,
                    },
                )

                return JobResult(
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    output_data=result,
                    execution_time=execution_time,
                )
            else:
                raise Exception(
                    f"No processor available for job type: {job_metadata.job_type}"
                )

        except Exception as e:
            logger.error(f"Job processing failed for {job_id}: {e}")

            # Update job status to failed
            await self.aws_manager.update_job_metadata(
                job_id, {"status": JobStatus.FAILED.value, "error_message": str(e)}
            )

            return JobResult(
                job_id=job_id, status=JobStatus.FAILED, error_message=str(e)
            )

    async def get_job_status(self, job_id: str) -> Optional[JobMetadata]:
        """Get the current status of a job."""
        return await self.aws_manager.get_job_metadata(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        try:
            # Update job status to cancelled
            success = await self.aws_manager.update_job_metadata(
                job_id, {"status": JobStatus.CANCELLED.value}
            )

            if success:
                logger.info(f"Job {job_id} cancelled")

            return success

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    async def retry_job(self, job_id: str) -> bool:
        """Retry a failed job."""
        try:
            job_metadata = await self.aws_manager.get_job_metadata(job_id)
            if not job_metadata:
                return False

            # Check if job can be retried
            if job_metadata.status != JobStatus.FAILED:
                logger.warning(
                    f"Job {job_id} cannot be retried (status: {job_metadata.status})"
                )
                return False

            if job_metadata.retry_count >= job_metadata.max_retries:
                logger.warning(f"Job {job_id} has exceeded maximum retry attempts")
                return False

            # Update retry count and status
            updates = {
                "status": JobStatus.QUEUED.value,
                "retry_count": job_metadata.retry_count + 1,
                "error_message": None,
                "progress": 0.0,
            }

            success = await self.aws_manager.update_job_metadata(job_id, updates)
            if not success:
                return False

            # Re-enqueue the job
            updated_metadata = await self.aws_manager.get_job_metadata(job_id)
            success = await self.aws_manager.enqueue_job(updated_metadata)

            if success:
                logger.info(
                    f"Job {job_id} retried (attempt {job_metadata.retry_count + 1})"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to retry job {job_id}: {e}")
            return False

    async def list_jobs(
        self,
        user_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 50,
    ) -> List[JobMetadata]:
        """List jobs with optional filtering."""
        jobs = await self.aws_manager.list_jobs(user_id, status, limit)

        # Additional filtering by job type if specified
        if job_type:
            jobs = [job for job in jobs if job.job_type == job_type]

        # Sort by created_at descending
        jobs.sort(key=lambda x: x.created_at, reverse=True)

        return jobs

    async def get_job_logs(self, job_id: str) -> List[str]:
        """Get logs for a job (this would typically fetch from CloudWatch)."""
        # In a real implementation, this would fetch logs from CloudWatch
        # For now, return a placeholder
        return [
            f"Job {job_id} created",
            f"Job {job_id} queued for processing",
            f"Job {job_id} started processing",
            # Add more log entries as needed
        ]

    async def get_job_metrics(self, job_id: str) -> Dict[str, Any]:
        """Get metrics for a job."""
        job_metadata = await self.aws_manager.get_job_metadata(job_id)
        if not job_metadata:
            return {}

        metrics = {
            "job_id": job_id,
            "job_type": job_metadata.job_type.value,
            "status": job_metadata.status.value,
            "progress": job_metadata.progress,
            "created_at": job_metadata.created_at.isoformat(),
            "updated_at": job_metadata.updated_at.isoformat(),
            "retry_count": job_metadata.retry_count,
            "priority": job_metadata.priority,
        }

        if job_metadata.status == JobStatus.COMPLETED:
            execution_time = (
                job_metadata.updated_at - job_metadata.created_at
            ).total_seconds()
            metrics["execution_time_seconds"] = execution_time

        return metrics

    async def cleanup_old_jobs(self, days: int = 30) -> int:
        """Clean up old job records."""
        # In a real implementation, this would scan DynamoDB for old jobs
        # and delete them. For now, return a placeholder count.
        cutoff_date = datetime.now() - timedelta(days=days)

        # This is a placeholder implementation
        # In reality, you'd scan DynamoDB with a filter expression
        logger.info(f"Cleaning up jobs older than {cutoff_date}")

        return 0  # Placeholder return value

    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get statistics about the job queue."""
        try:
            # Get queue URL
            queue_url = self.aws_manager.sqs_client.get_queue_url(
                QueueName=self.aws_manager.queue_name
            )["QueueUrl"]

            # Get queue attributes
            response = self.aws_manager.sqs_client.get_queue_attributes(
                QueueUrl=queue_url, AttributeNames=["All"]
            )

            attributes = response["Attributes"]

            return {
                "queue_name": self.aws_manager.queue_name,
                "approximate_number_of_messages": int(
                    attributes.get("ApproximateNumberOfMessages", 0)
                ),
                "approximate_number_of_messages_not_visible": int(
                    attributes.get("ApproximateNumberOfMessagesNotVisible", 0)
                ),
                "approximate_number_of_messages_delayed": int(
                    attributes.get("ApproximateNumberOfMessagesDelayed", 0)
                ),
                "created_timestamp": attributes.get("CreatedTimestamp"),
                "last_modified_timestamp": attributes.get("LastModifiedTimestamp"),
                "visibility_timeout": int(attributes.get("VisibilityTimeout", 30)),
                "message_retention_period": int(
                    attributes.get("MessageRetentionPeriod", 345600)
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get queue statistics: {e}")
            return {}


# Global instance
job_orchestrator = JobOrchestrator()


# Local job processors (examples)
async def process_etl_job(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process an ETL job locally."""
    logger.info("Processing ETL job locally")

    # Simulate ETL processing
    await asyncio.sleep(2)

    return {
        "status": "completed",
        "records_processed": 1000,
        "output_s3_path": "s3://bucket/processed/data.parquet",
    }


async def process_forecast_job(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a forecast job locally."""
    logger.info("Processing forecast job locally")

    # Simulate forecast processing
    await asyncio.sleep(3)

    return {
        "status": "completed",
        "forecast_accuracy": 0.85,
        "forecast_s3_path": "s3://bucket/forecasts/forecast.json",
    }


async def process_optimization_job(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process an optimization job locally."""
    logger.info("Processing optimization job locally")

    # Simulate optimization processing
    await asyncio.sleep(4)

    return {
        "status": "completed",
        "optimization_score": 0.92,
        "recommendations": ["order_sku_001", "discount_sku_002"],
    }


# Register local processors
job_orchestrator.register_local_processor(JobType.ETL, process_etl_job)
job_orchestrator.register_local_processor(JobType.FORECAST, process_forecast_job)
job_orchestrator.register_local_processor(
    JobType.OPTIMIZATION, process_optimization_job
)

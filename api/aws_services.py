"""
AWS services integration module.

This module provides clients and utilities for interacting with AWS services:
- S3 for data storage
- DynamoDB for job metadata
- SQS for job queuing
- Lambda for async task processing
"""

import boto3
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import uuid4
from botocore.exceptions import ClientError, NoCredentialsError
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from api.config import settings

logger = structlog.get_logger()


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Job type enumeration."""
    ETL = "etl"
    FORECAST = "forecast"
    OPTIMIZATION = "optimization"
    BATCH_PROCESSING = "batch_processing"


@dataclass
class JobMetadata:
    """Job metadata structure."""
    job_id: str
    job_type: JobType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    user_id: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    estimated_completion: Optional[datetime] = None
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DynamoDB storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.estimated_completion:
            data['estimated_completion'] = self.estimated_completion.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobMetadata':
        """Create instance from dictionary."""
        # Convert ISO strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('estimated_completion'):
            data['estimated_completion'] = datetime.fromisoformat(data['estimated_completion'])
        return cls(**data)


class AWSServiceManager:
    """Manager for AWS services integration."""
    
    def __init__(self):
        """Initialize AWS clients."""
        self.region = settings.s3_region
        self.bucket_name = settings.s3_bucket
        
        # Check if we're in test mode
        self.is_test_mode = settings.app_env.lower() in ['test', 'development', 'local']
        
        # Initialize clients
        self._s3_client = None
        self._dynamodb_client = None
        self._sqs_client = None
        self._lambda_client = None
        
        # Table and queue names
        self.jobs_table_name = f"{settings.app_name.lower().replace(' ', '-')}-jobs"
        self.queue_name = f"{settings.app_name.lower().replace(' ', '-')}-job-queue"
        
    @property
    def s3_client(self):
        """Get S3 client (lazy initialization)."""
        if self._s3_client is None:
            # Use dummy credentials in test mode
            if self.is_test_mode:
                self._s3_client = boto3.client(
                    's3',
                    region_name=self.region,
                    aws_access_key_id='dummy',
                    aws_secret_access_key='dummy'
                )
            else:
                self._s3_client = boto3.client(
                    's3',
                    region_name=self.region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key
                )
        return self._s3_client
    
    @property
    def dynamodb_client(self):
        """Get DynamoDB client (lazy initialization)."""
        if self._dynamodb_client is None:
            # Use dummy credentials in test mode
            if self.is_test_mode:
                self._dynamodb_client = boto3.client(
                    'dynamodb',
                    region_name=self.region,
                    aws_access_key_id='dummy',
                    aws_secret_access_key='dummy'
                )
            else:
                self._dynamodb_client = boto3.client(
                    'dynamodb',
                    region_name=self.region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key
                )
        return self._dynamodb_client
    
    @property
    def sqs_client(self):
        """Get SQS client (lazy initialization)."""
        if self._sqs_client is None:
            # Use dummy credentials in test mode
            if self.is_test_mode:
                self._sqs_client = boto3.client(
                    'sqs',
                    region_name=self.region,
                    aws_access_key_id='dummy',
                    aws_secret_access_key='dummy'
                )
            else:
                self._sqs_client = boto3.client(
                    'sqs',
                    region_name=self.region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key
                )
        return self._sqs_client
    
    @property
    def lambda_client(self):
        """Get Lambda client (lazy initialization)."""
        if self._lambda_client is None:
            # Use dummy credentials in test mode
            if self.is_test_mode:
                self._lambda_client = boto3.client(
                    'lambda',
                    region_name=self.region,
                    aws_access_key_id='dummy',
                    aws_secret_access_key='dummy'
                )
            else:
                self._lambda_client = boto3.client(
                    'lambda',
                    region_name=self.region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key
                )
        return self._lambda_client
    
    async def upload_to_s3(self, data: bytes, key: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """Upload data to S3."""
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                **extra_args
            )
            
            logger.info(f"Successfully uploaded to S3: {key}")
            return f"s3://{self.bucket_name}/{key}"
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise
    
    async def download_from_s3(self, key: str) -> bytes:
        """Download data from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
            
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            raise
    
    async def list_s3_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List objects in S3 bucket."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            return response.get('Contents', [])
            
        except ClientError as e:
            logger.error(f"Failed to list S3 objects: {e}")
            raise
    
    async def create_job_metadata(self, job_metadata: JobMetadata) -> bool:
        """Create job metadata in DynamoDB."""
        try:
            # Convert to DynamoDB format
            item = {}
            for key, value in job_metadata.to_dict().items():
                if isinstance(value, str):
                    item[key] = {'S': value}
                elif isinstance(value, (int, float)):
                    item[key] = {'N': str(value)}
                elif isinstance(value, dict):
                    item[key] = {'S': json.dumps(value)}
                elif value is None:
                    continue  # Skip None values
                else:
                    item[key] = {'S': str(value)}
            
            self.dynamodb_client.put_item(
                TableName=self.jobs_table_name,
                Item=item
            )
            
            logger.info(f"Created job metadata: {job_metadata.job_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to create job metadata: {e}")
            return False
    
    async def get_job_metadata(self, job_id: str) -> Optional[JobMetadata]:
        """Get job metadata from DynamoDB."""
        try:
            response = self.dynamodb_client.get_item(
                TableName=self.jobs_table_name,
                Key={'job_id': {'S': job_id}}
            )
            
            if 'Item' not in response:
                return None
            
            # Convert from DynamoDB format
            item = response['Item']
            data = {}
            for key, value in item.items():
                if 'S' in value:
                    if key in ['input_data', 'output_data'] and value['S']:
                        try:
                            data[key] = json.loads(value['S'])
                        except json.JSONDecodeError:
                            data[key] = value['S']
                    else:
                        data[key] = value['S']
                elif 'N' in value:
                    if key in ['progress']:
                        data[key] = float(value['N'])
                    else:
                        data[key] = int(value['N'])
            
            return JobMetadata.from_dict(data)
            
        except ClientError as e:
            logger.error(f"Failed to get job metadata: {e}")
            return None
    
    async def update_job_metadata(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job metadata in DynamoDB."""
        try:
            # Build update expression
            update_expression_parts = []
            expression_attribute_names = {}
            expression_attribute_values = {}
            
            for key, value in updates.items():
                attr_name = f"#{key}"
                attr_value = f":{key}"
                
                update_expression_parts.append(f"{attr_name} = {attr_value}")
                expression_attribute_names[attr_name] = key
                
                if isinstance(value, str):
                    expression_attribute_values[attr_value] = {'S': value}
                elif isinstance(value, (int, float)):
                    expression_attribute_values[attr_value] = {'N': str(value)}
                elif isinstance(value, dict):
                    expression_attribute_values[attr_value] = {'S': json.dumps(value)}
                elif isinstance(value, datetime):
                    expression_attribute_values[attr_value] = {'S': value.isoformat()}
                else:
                    expression_attribute_values[attr_value] = {'S': str(value)}
            
            # Always update the updated_at timestamp
            update_expression_parts.append("#updated_at = :updated_at")
            expression_attribute_names["#updated_at"] = "updated_at"
            expression_attribute_values[":updated_at"] = {'S': datetime.now().isoformat()}
            
            update_expression = "SET " + ", ".join(update_expression_parts)
            
            self.dynamodb_client.update_item(
                TableName=self.jobs_table_name,
                Key={'job_id': {'S': job_id}},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values
            )
            
            logger.info(f"Updated job metadata: {job_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to update job metadata: {e}")
            return False
    
    async def list_jobs(self, user_id: Optional[str] = None, status: Optional[JobStatus] = None, 
                       limit: int = 50) -> List[JobMetadata]:
        """List jobs with optional filtering."""
        try:
            # Build scan parameters
            scan_params = {
                'TableName': self.jobs_table_name,
                'Limit': limit
            }
            
            filter_expressions = []
            expression_attribute_names = {}
            expression_attribute_values = {}
            
            if user_id:
                filter_expressions.append("#user_id = :user_id")
                expression_attribute_names["#user_id"] = "user_id"
                expression_attribute_values[":user_id"] = {'S': user_id}
            
            if status:
                filter_expressions.append("#status = :status")
                expression_attribute_names["#status"] = "status"
                expression_attribute_values[":status"] = {'S': status.value}
            
            if filter_expressions:
                scan_params['FilterExpression'] = " AND ".join(filter_expressions)
                scan_params['ExpressionAttributeNames'] = expression_attribute_names
                scan_params['ExpressionAttributeValues'] = expression_attribute_values
            
            response = self.dynamodb_client.scan(**scan_params)
            
            jobs = []
            for item in response.get('Items', []):
                # Convert from DynamoDB format
                data = {}
                for key, value in item.items():
                    if 'S' in value:
                        if key in ['input_data', 'output_data'] and value['S']:
                            try:
                                data[key] = json.loads(value['S'])
                            except json.JSONDecodeError:
                                data[key] = value['S']
                        else:
                            data[key] = value['S']
                    elif 'N' in value:
                        if key in ['progress']:
                            data[key] = float(value['N'])
                        else:
                            data[key] = int(value['N'])
                
                jobs.append(JobMetadata.from_dict(data))
            
            return jobs
            
        except ClientError as e:
            logger.error(f"Failed to list jobs: {e}")
            return []
    
    async def enqueue_job(self, job_metadata: JobMetadata, delay_seconds: int = 0) -> bool:
        """Enqueue a job for processing."""
        try:
            # Get queue URL
            queue_url = self.sqs_client.get_queue_url(QueueName=self.queue_name)['QueueUrl']
            
            # Create message
            message = {
                'job_id': job_metadata.job_id,
                'job_type': job_metadata.job_type.value,
                'user_id': job_metadata.user_id,
                'input_data': job_metadata.input_data,
                'priority': job_metadata.priority
            }
            
            # Send message to queue
            response = self.sqs_client.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message),
                DelaySeconds=delay_seconds,
                MessageAttributes={
                    'job_type': {
                        'StringValue': job_metadata.job_type.value,
                        'DataType': 'String'
                    },
                    'priority': {
                        'StringValue': str(job_metadata.priority),
                        'DataType': 'Number'
                    }
                }
            )
            
            logger.info(f"Enqueued job: {job_metadata.job_id}, Message ID: {response['MessageId']}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to enqueue job: {e}")
            return False
    
    async def invoke_lambda_function(self, function_name: str, payload: Dict[str, Any], 
                                   asynchronous: bool = True) -> Optional[Dict[str, Any]]:
        """Invoke a Lambda function."""
        try:
            invocation_type = 'Event' if asynchronous else 'RequestResponse'
            
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType=invocation_type,
                Payload=json.dumps(payload)
            )
            
            if not asynchronous:
                # For synchronous invocation, parse the response
                response_payload = json.loads(response['Payload'].read())
                return response_payload
            
            logger.info(f"Invoked Lambda function: {function_name}")
            return {'status': 'invoked', 'request_id': response['ResponseMetadata']['RequestId']}
            
        except ClientError as e:
            logger.error(f"Failed to invoke Lambda function: {e}")
            return None
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of all AWS services."""
        health_status = {}
        
        # In test mode, mock the services as healthy
        if self.is_test_mode:
            health_status = {
                's3': 'healthy',
                'dynamodb': 'healthy',
                'sqs': 'healthy',
                'lambda': 'healthy'
            }
        else:
            # Check S3
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                health_status['s3'] = 'healthy'
            except ClientError:
                health_status['s3'] = 'unhealthy'
            
            # Check DynamoDB
            try:
                self.dynamodb_client.describe_table(TableName=self.jobs_table_name)
                health_status['dynamodb'] = 'healthy'
            except ClientError:
                health_status['dynamodb'] = 'unhealthy'
            
            # Check SQS
            try:
                self.sqs_client.get_queue_url(QueueName=self.queue_name)
                health_status['sqs'] = 'healthy'
            except ClientError:
                health_status['sqs'] = 'unhealthy'
            
            # Check Lambda (this is optional, depends on your setup)
            try:
                self.lambda_client.list_functions(MaxItems=1)
                health_status['lambda'] = 'healthy'
            except ClientError:
                health_status['lambda'] = 'unhealthy'
        
        return health_status


# Global instance
aws_service_manager = AWSServiceManager()

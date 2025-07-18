"""
Middleware for FastAPI application.

This module provides middleware for rate limiting, request tracking,
security headers, and other cross-cutting concerns.
"""

import time
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import structlog

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


logger = structlog.get_logger()


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window algorithm."""
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.client_requests: Dict[str, deque] = defaultdict(deque)
        
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        now = time.time()
        minute_ago = now - 60
        
        # Get or create request queue for this client
        requests = self.client_requests[client_ip]
        
        # Remove old requests (older than 1 minute)
        while requests and requests[0] < minute_ago:
            requests.popleft()
        
        # Check if rate limit exceeded
        if len(requests) >= self.requests_per_minute:
            return True
        
        # Add current request
        requests.append(now)
        
        return False
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        client_ip = self._get_client_ip(request)
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/liveness", "/health/readiness"]:
            return await call_next(request)
        
        # Check rate limit
        if self._is_rate_limited(client_ip):
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=request.url.path,
                method=request.method
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute allowed"
                },
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics and logging."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with tracking."""
        start_time = time.time()
        
        # Generate request ID
        request_id = f"req_{int(start_time * 1000)}"
        
        # Log request start
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("User-Agent", "unknown")
        )
        
        # Add request ID to state
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error
            processing_time = time.time() - start_time
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                processing_time=processing_time,
                error=str(e)
            )
            raise
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log request completion
        logger.info(
            "Request completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            processing_time=processing_time
        )
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = str(processing_time)
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Middleware to add cache control headers."""
    
    def __init__(self, app: ASGIApp, cache_control_rules: Optional[Dict[str, str]] = None):
        super().__init__(app)
        self.cache_control_rules = cache_control_rules or {
            "/health": "no-cache",
            "/forecast": "public, max-age=300",  # 5 minutes
            "/optimize": "no-cache",
            "/": "public, max-age=3600",  # 1 hour
        }
    
    async def dispatch(self, request: Request, call_next):
        """Add cache control headers based on path."""
        response = await call_next(request)
        
        # Determine cache control based on path
        path = request.url.path
        cache_control = "no-cache"  # Default
        
        for pattern, control in self.cache_control_rules.items():
            if path.startswith(pattern):
                cache_control = control
                break
        
        response.headers["Cache-Control"] = cache_control
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""
    
    async def dispatch(self, request: Request, call_next):
        """Handle errors globally."""
        try:
            return await call_next(request)
        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(
                "Unexpected error",
                path=request.url.path,
                method=request.method,
                error=str(e),
                exc_info=True
            )
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": "An unexpected error occurred. Please try again later."
                }
            )


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect metrics."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_count = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_count = defaultdict(int)
    
    async def dispatch(self, request: Request, call_next):
        """Collect metrics for monitoring."""
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        try:
            response = await call_next(request)
            
            # Record metrics
            processing_time = time.time() - start_time
            key = f"{method} {path}"
            
            self.request_count[key] += 1
            self.response_times[key].append(processing_time)
            
            # Keep only last 1000 response times
            if len(self.response_times[key]) > 1000:
                self.response_times[key] = self.response_times[key][-1000:]
            
            # Record errors
            if response.status_code >= 400:
                self.error_count[key] += 1
            
            return response
            
        except Exception as e:
            # Record error
            key = f"{method} {path}"
            self.error_count[key] += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = {
            "request_counts": dict(self.request_count),
            "error_counts": dict(self.error_count),
            "response_times": {}
        }
        
        # Calculate response time statistics
        for key, times in self.response_times.items():
            if times:
                metrics["response_times"][key] = {
                    "count": len(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "p95": sorted(times)[int(len(times) * 0.95)] if len(times) > 0 else 0,
                    "p99": sorted(times)[int(len(times) * 0.99)] if len(times) > 0 else 0
                }
        
        return metrics


# Global metrics instance
metrics_middleware = None


def get_metrics() -> Dict[str, Any]:
    """Get current metrics from middleware."""
    if metrics_middleware:
        return metrics_middleware.get_metrics()
    return {}

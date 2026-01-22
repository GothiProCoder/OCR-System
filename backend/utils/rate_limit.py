"""
Rate Limiting Middleware
========================
Simple in-memory rate limiting for API endpoints.

For production, consider using Redis-based rate limiting for
distributed deployments.

Usage:
    from utils.rate_limit import RateLimitMiddleware, rate_limit
    
    # As middleware (global)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    
    # As decorator (per-endpoint)
    @rate_limit(requests_per_minute=10)
    async def my_endpoint():
        ...
"""

import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple
from functools import wraps

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10  # Allow short bursts
    cleanup_interval: int = 300  # Seconds between cleanup runs


@dataclass
class ClientBucket:
    """Token bucket for a single client."""
    tokens: float = 60.0
    last_update: float = field(default_factory=time.time)
    request_count: int = 0


class RateLimiter:
    """
    Token bucket rate limiter with sliding window.
    
    Thread-safe implementation using asyncio locks.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, ClientBucket] = defaultdict(ClientBucket)
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()
    
    def _get_client_key(self, request: Request) -> str:
        """
        Get unique identifier for the client.
        Uses X-Forwarded-For if behind proxy, otherwise client host.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP if there are multiple
            return forwarded.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"
    
    async def _cleanup_old_buckets(self) -> None:
        """Remove stale client buckets to prevent memory leaks."""
        now = time.time()
        if now - self._last_cleanup < self.config.cleanup_interval:
            return
        
        async with self._lock:
            threshold = now - 3600  # Remove clients inactive for 1 hour
            stale_keys = [
                key for key, bucket in self._buckets.items()
                if bucket.last_update < threshold
            ]
            for key in stale_keys:
                del self._buckets[key]
            
            if stale_keys:
                logger.debug(f"Cleaned up {len(stale_keys)} stale rate limit buckets")
            
            self._last_cleanup = now
    
    async def check_rate_limit(
        self, 
        request: Request
    ) -> Tuple[bool, int, int]:
        """
        Check if request is allowed under rate limit.
        
        Returns:
            Tuple of (allowed, remaining_requests, reset_time_seconds)
        """
        await self._cleanup_old_buckets()
        
        client_key = self._get_client_key(request)
        now = time.time()
        
        async with self._lock:
            bucket = self._buckets[client_key]
            
            # Refill tokens based on time passed
            time_passed = now - bucket.last_update
            tokens_to_add = time_passed * (self.config.requests_per_minute / 60.0)
            bucket.tokens = min(
                self.config.requests_per_minute,
                bucket.tokens + tokens_to_add
            )
            bucket.last_update = now
            
            # Check if request is allowed
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                bucket.request_count += 1
                remaining = int(bucket.tokens)
                reset_time = int(60 - (now % 60))
                return True, remaining, reset_time
            else:
                remaining = 0
                reset_time = int((1.0 - bucket.tokens) * 60 / self.config.requests_per_minute)
                return False, remaining, reset_time
    
    def get_rate_limit_headers(
        self, 
        remaining: int, 
        reset_time: int
    ) -> Dict[str, str]:
        """Generate rate limit headers for response."""
        return {
            "X-RateLimit-Limit": str(self.config.requests_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Usage:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=60,
            exclude_paths=["/health", "/docs"]
        )
    """
    
    def __init__(
        self, 
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour
        )
        self.limiter = RateLimiter(self.config)
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through rate limiter."""
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Check rate limit
        allowed, remaining, reset_time = await self.limiter.check_rate_limit(request)
        
        if not allowed:
            headers = self.limiter.get_rate_limit_headers(remaining, reset_time)
            headers["Retry-After"] = str(reset_time)
            
            return Response(
                content='{"detail": "Rate limit exceeded. Please try again later."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers=headers
            )
        
        # Process request and add rate limit headers to response
        response = await call_next(request)
        headers = self.limiter.get_rate_limit_headers(remaining, reset_time)
        for key, value in headers.items():
            response.headers[key] = value
        
        return response


def rate_limit(
    requests_per_minute: int = 10,
    error_message: str = "Rate limit exceeded for this endpoint"
) -> Callable:
    """
    Decorator for per-endpoint rate limiting.
    
    Usage:
        @router.get("/expensive-operation")
        @rate_limit(requests_per_minute=5)
        async def expensive_operation():
            ...
    """
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=requests_per_minute))
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs):
            allowed, remaining, reset_time = await limiter.check_rate_limit(request)
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=error_message,
                    headers={
                        "Retry-After": str(reset_time),
                        "X-RateLimit-Remaining": str(remaining),
                        "X-RateLimit-Reset": str(reset_time)
                    }
                )
            
            return await func(*args, request=request, **kwargs)
        
        return wrapper
    return decorator


# Default rate limiter instance for OCR/LLM endpoints
ocr_rate_limiter = RateLimiter(RateLimitConfig(
    requests_per_minute=20,  # Limit OCR requests
    requests_per_hour=200
))

llm_rate_limiter = RateLimiter(RateLimitConfig(
    requests_per_minute=30,  # Limit LLM requests
    requests_per_hour=500
))

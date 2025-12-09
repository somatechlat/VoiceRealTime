"""Prometheus metrics for AgentVoiceBox platform.

Implements Requirements 14.1, 14.2, 14.3:
- Latency histograms (p50/p95/p99) for all operations
- Gauges for active_connections, queue_depth, worker_utilization
- Standard naming convention with tenant labels

Metrics exposed at /metrics endpoint.
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, Tuple

from flask import Flask
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    make_wsgi_app,
    REGISTRY,
)

from ..config import AppConfig

# Namespace for all metrics
NAMESPACE = "agentvoicebox"

# Standard buckets for latency histograms (in seconds)
# Covers 1ms to 10s with good granularity for p50/p95/p99
LATENCY_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0, float("inf")
)

# Audio processing buckets (longer operations)
AUDIO_LATENCY_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, float("inf")
)

# ============================================================================
# Service Info
# ============================================================================

service_info = Info(
    "service",
    "Service information",
    namespace=NAMESPACE,
)

# ============================================================================
# WebSocket/Gateway Metrics (Requirement 14.1, 14.2)
# ============================================================================

websocket_message_latency = Histogram(
    "websocket_message_processing_seconds",
    "Latency of WebSocket message processing",
    labelnames=["message_type", "tenant_id"],
    namespace=NAMESPACE,
    buckets=LATENCY_BUCKETS,
)

websocket_connections_total = Counter(
    "websocket_connections_total",
    "Total WebSocket connections",
    labelnames=["tenant_id", "status"],
    namespace=NAMESPACE,
)

active_connections = Gauge(
    "active_connections",
    "Current active WebSocket connections",
    labelnames=["tenant_id"],
    namespace=NAMESPACE,
)

active_connections_total = Gauge(
    "active_connections_total",
    "Total active WebSocket connections across all tenants",
    namespace=NAMESPACE,
)

# ============================================================================
# Session Metrics
# ============================================================================

session_starts_total = Counter(
    "session_starts_total",
    "Total sessions started",
    labelnames=["tenant_id"],
    namespace=NAMESPACE,
)

session_duration_seconds = Histogram(
    "session_duration_seconds",
    "Duration of voice sessions",
    labelnames=["tenant_id"],
    namespace=NAMESPACE,
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, float("inf")),
)

# ============================================================================
# STT Metrics (Requirement 14.2)
# ============================================================================

stt_transcription_latency = Histogram(
    "stt_transcription_seconds",
    "Latency of STT transcription",
    labelnames=["tenant_id", "model"],
    namespace=NAMESPACE,
    buckets=AUDIO_LATENCY_BUCKETS,
)

stt_requests_total = Counter(
    "stt_requests_total",
    "Total STT transcription requests",
    labelnames=["tenant_id", "status"],
    namespace=NAMESPACE,
)

stt_audio_duration_seconds = Histogram(
    "stt_audio_duration_seconds",
    "Duration of audio sent for transcription",
    labelnames=["tenant_id"],
    namespace=NAMESPACE,
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, float("inf")),
)

# ============================================================================
# TTS Metrics (Requirement 14.2)
# ============================================================================

tts_synthesis_latency = Histogram(
    "tts_synthesis_seconds",
    "Latency of TTS synthesis (time to first byte)",
    labelnames=["tenant_id", "voice"],
    namespace=NAMESPACE,
    buckets=AUDIO_LATENCY_BUCKETS,
)

tts_requests_total = Counter(
    "tts_requests_total",
    "Total TTS synthesis requests",
    labelnames=["tenant_id", "status"],
    namespace=NAMESPACE,
)

tts_characters_total = Counter(
    "tts_characters_total",
    "Total characters synthesized",
    labelnames=["tenant_id"],
    namespace=NAMESPACE,
)

# ============================================================================
# LLM Metrics (Requirement 14.2)
# ============================================================================

llm_generation_latency = Histogram(
    "llm_generation_seconds",
    "Latency of LLM response generation (time to first token)",
    labelnames=["tenant_id", "provider", "model"],
    namespace=NAMESPACE,
    buckets=AUDIO_LATENCY_BUCKETS,
)

llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM generation requests",
    labelnames=["tenant_id", "provider", "status"],
    namespace=NAMESPACE,
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens processed",
    labelnames=["tenant_id", "provider", "direction"],
    namespace=NAMESPACE,
)

# ============================================================================
# Queue/Worker Metrics (Requirement 14.3)
# ============================================================================

queue_depth = Gauge(
    "queue_depth",
    "Current depth of work queues",
    labelnames=["queue_name"],
    namespace=NAMESPACE,
)

worker_utilization = Gauge(
    "worker_utilization_ratio",
    "Worker utilization (0-1)",
    labelnames=["worker_type"],
    namespace=NAMESPACE,
)

worker_active = Gauge(
    "worker_active",
    "Number of active workers",
    labelnames=["worker_type"],
    namespace=NAMESPACE,
)

# ============================================================================
# Rate Limiting Metrics
# ============================================================================

rate_limit_hits_total = Counter(
    "rate_limit_hits_total",
    "Total rate limit hits",
    labelnames=["tenant_id", "limit_type"],
    namespace=NAMESPACE,
)

rate_limit_remaining = Gauge(
    "rate_limit_remaining",
    "Remaining rate limit quota",
    labelnames=["tenant_id", "limit_type"],
    namespace=NAMESPACE,
)

# ============================================================================
# Authentication Metrics
# ============================================================================

auth_attempts_total = Counter(
    "auth_attempts_total",
    "Total authentication attempts",
    labelnames=["method", "status"],
    namespace=NAMESPACE,
)

auth_latency = Histogram(
    "auth_latency_seconds",
    "Authentication latency",
    labelnames=["method"],
    namespace=NAMESPACE,
    buckets=LATENCY_BUCKETS,
)

# ============================================================================
# Error Metrics
# ============================================================================

errors_total = Counter(
    "errors_total",
    "Total errors by type",
    labelnames=["error_type", "tenant_id"],
    namespace=NAMESPACE,
)

# ============================================================================
# Database Metrics
# ============================================================================

db_query_latency = Histogram(
    "db_query_seconds",
    "Database query latency",
    labelnames=["operation", "table"],
    namespace=NAMESPACE,
    buckets=LATENCY_BUCKETS,
)

db_connections_active = Gauge(
    "db_connections_active",
    "Active database connections",
    namespace=NAMESPACE,
)

# ============================================================================
# Redis Metrics
# ============================================================================

redis_operation_latency = Histogram(
    "redis_operation_seconds",
    "Redis operation latency",
    labelnames=["operation"],
    namespace=NAMESPACE,
    buckets=LATENCY_BUCKETS,
)

redis_connections_active = Gauge(
    "redis_connections_active",
    "Active Redis connections",
    namespace=NAMESPACE,
)

# ============================================================================
# Helper Functions and Decorators
# ============================================================================

@contextmanager
def track_latency(
    histogram: Histogram,
    labels: Optional[dict] = None,
) -> Generator[None, None, None]:
    """Context manager to track operation latency.
    
    Usage:
        with track_latency(websocket_message_latency, {"message_type": "audio", "tenant_id": "t1"}):
            process_message()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        if labels:
            histogram.labels(**labels).observe(duration)
        else:
            histogram.observe(duration)


def timed(
    histogram: Histogram,
    label_func: Optional[Callable[..., dict]] = None,
) -> Callable:
    """Decorator to track function execution time.
    
    Usage:
        @timed(stt_transcription_latency, lambda self, audio: {"tenant_id": self.tenant_id, "model": "whisper"})
        async def transcribe(self, audio):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                labels = label_func(*args, **kwargs) if label_func else {}
                if labels:
                    histogram.labels(**labels).observe(duration)
                else:
                    histogram.observe(duration)
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                labels = label_func(*args, **kwargs) if label_func else {}
                if labels:
                    histogram.labels(**labels).observe(duration)
                else:
                    histogram.observe(duration)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def increment_counter(
    counter: Counter,
    labels: Optional[dict] = None,
    value: float = 1,
) -> None:
    """Increment a counter with optional labels."""
    if labels:
        counter.labels(**labels).inc(value)
    else:
        counter.inc(value)


def set_gauge(
    gauge: Gauge,
    value: float,
    labels: Optional[dict] = None,
) -> None:
    """Set a gauge value with optional labels."""
    if labels:
        gauge.labels(**labels).set(value)
    else:
        gauge.set(value)


# ============================================================================
# Initialization
# ============================================================================

def init_metrics(config: AppConfig) -> Tuple[CollectorRegistry, Any]:
    """Initialize Prometheus metrics and return registry and WSGI app.
    
    Args:
        config: Application configuration
        
    Returns:
        Tuple of (registry, metrics_wsgi_app)
    """
    # Set service info
    service_info.info({
        "version": "1.0.0",
        "environment": config.flask_env,
        "service_name": config.observability.service_name,
    })
    
    # Create WSGI app for /metrics endpoint
    metrics_app = make_wsgi_app(registry=REGISTRY)
    
    return REGISTRY, metrics_app


# Legacy aliases for backward compatibility (without labels for existing code)
# These are separate metrics to avoid label conflicts
session_starts = Counter(
    "session_starts_legacy_total",
    "Total sessions started (legacy, no labels)",
    namespace=NAMESPACE,
)

policy_denials = Counter(
    "policy_denials_total",
    "Number of requests denied by OPA",
    namespace=NAMESPACE,
)

response_latency = Histogram(
    "response_latency_legacy_seconds",
    "Latency of AI response generation (legacy)",
    namespace=NAMESPACE,
    buckets=AUDIO_LATENCY_BUCKETS,
)

active_sessions = Gauge(
    "active_sessions_legacy",
    "Current active realtime sessions (legacy, no labels)",
    namespace=NAMESPACE,
)


__all__ = [
    # Initialization
    "init_metrics",
    
    # Service info
    "service_info",
    
    # WebSocket metrics
    "websocket_message_latency",
    "websocket_connections_total",
    "active_connections",
    "active_connections_total",
    
    # Session metrics
    "session_starts_total",
    "session_duration_seconds",
    
    # STT metrics
    "stt_transcription_latency",
    "stt_requests_total",
    "stt_audio_duration_seconds",
    
    # TTS metrics
    "tts_synthesis_latency",
    "tts_requests_total",
    "tts_characters_total",
    
    # LLM metrics
    "llm_generation_latency",
    "llm_requests_total",
    "llm_tokens_total",
    
    # Queue/Worker metrics
    "queue_depth",
    "worker_utilization",
    "worker_active",
    
    # Rate limiting metrics
    "rate_limit_hits_total",
    "rate_limit_remaining",
    
    # Auth metrics
    "auth_attempts_total",
    "auth_latency",
    
    # Error metrics
    "errors_total",
    
    # Database metrics
    "db_query_latency",
    "db_connections_active",
    
    # Redis metrics
    "redis_operation_latency",
    "redis_connections_active",
    
    # Helpers
    "track_latency",
    "timed",
    "increment_counter",
    "set_gauge",
    
    # Legacy aliases
    "session_starts",
    "policy_denials",
    "response_latency",
    "active_sessions",
]

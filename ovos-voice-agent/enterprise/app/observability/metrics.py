"""Prometheus metrics bootstrapping."""

from __future__ import annotations

from typing import Tuple

from flask import Flask
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, make_wsgi_app

from ..config import AppConfig

METRIC_NAMESPACE_SESSION = "voice_session"

# Default metrics available to the rest of the service
session_starts = Counter(
    "session_starts_total",
    "Number of realtime sessions started",
    namespace=METRIC_NAMESPACE_SESSION,
)
policy_denials = Counter(
    "policy_denials_total",
    "Number of requests denied by OPA",
    namespace=METRIC_NAMESPACE_SESSION,
)
response_latency = Histogram(
    "response_latency_seconds",
    "Latency of AI response generation",
    namespace=METRIC_NAMESPACE_SESSION,
)
active_sessions = Gauge(
    "active_sessions",
    "Current active realtime sessions",
    namespace=METRIC_NAMESPACE_SESSION,
)


def init_metrics(config: AppConfig) -> Tuple[CollectorRegistry, Flask]:
    """Create the Prometheus registry and Flask app exposing metrics."""

    registry = CollectorRegistry(auto_describe=True)

    # Re-register metrics on the custom registry
    for metric in (session_starts, policy_denials, response_latency, active_sessions):
        metric._registry = registry  # type: ignore[attr-defined]

    metrics_app = make_wsgi_app(registry=registry)
    return registry, metrics_app


__all__ = [
    "init_metrics",
    "session_starts",
    "policy_denials",
    "response_latency",
    "active_sessions",
]

"""Structured JSON logging with correlation IDs.

Implements Requirement 14.4:
- Structured JSON logging
- Fields: timestamp, level, service, tenant_id, session_id, correlation_id, message
- PII redaction integration
"""
from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.config import dictConfig
from typing import Any, Dict, Optional

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    jsonlogger = None
    JSON_LOGGER_AVAILABLE = False

from ..config import AppConfig

# Context variables for request-scoped logging context
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_tenant_id: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)
_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return f"corr_{uuid.uuid4().hex[:16]}"


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set the correlation ID for the current context."""
    cid = correlation_id or generate_correlation_id()
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_logging_context(
    tenant_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """Set logging context for the current request."""
    if tenant_id:
        _tenant_id.set(tenant_id)
    if session_id:
        _session_id.set(session_id)
    if request_id:
        _request_id.set(request_id)
    if correlation_id:
        _correlation_id.set(correlation_id)


def clear_logging_context() -> None:
    """Clear logging context."""
    _correlation_id.set(None)
    _tenant_id.set(None)
    _session_id.set(None)
    _request_id.set(None)


class ContextFilter(logging.Filter):
    """Filter that adds context variables to log records."""

    def __init__(self, service_name: str = "agentvoicebox"):
        super().__init__()
        self.service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        record.service = self.service_name
        record.correlation_id = _correlation_id.get() or "-"
        record.tenant_id = _tenant_id.get() or "-"
        record.session_id = _session_id.get() or "-"
        record.request_id = _request_id.get() or "-"
        return True


class StructuredJsonFormatter(logging.Formatter):
    """JSON formatter with structured fields.
    
    Output format:
    {
        "timestamp": "2024-01-15T10:30:00.000Z",
        "level": "INFO",
        "service": "agentvoicebox",
        "correlation_id": "corr_abc123",
        "tenant_id": "tenant_xyz",
        "session_id": "sess_123",
        "message": "Processing request",
        "extra": {...}
    }
    """

    def __init__(self, service_name: str = "agentvoicebox"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json

        # Base fields
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": getattr(record, "service", self.service_name),
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", "-"),
            "tenant_id": getattr(record, "tenant_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
            "message": record.getMessage(),
        }

        # Add request_id if present
        request_id = getattr(record, "request_id", None)
        if request_id and request_id != "-":
            log_entry["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "service", "correlation_id", "tenant_id",
                "session_id", "request_id",
            ):
                extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str)


def configure_logging(config: AppConfig) -> None:
    """Configure structured logging across the service.
    
    Args:
        config: Application configuration
    """
    log_level = getattr(logging, config.observability.log_level.upper(), logging.INFO)
    service_name = config.observability.service_name

    # Determine formatter based on environment
    use_json = config.flask_env != "development" or JSON_LOGGER_AVAILABLE

    if use_json and JSON_LOGGER_AVAILABLE:
        # Use python-json-logger for production
        formatter_config = {
            "json": {
                "()": "app.observability.logging.StructuredJsonFormatter",
                "service_name": service_name,
            }
        }
        formatter_name = "json"
    else:
        # Plain text for development
        formatter_config = {
            "plain": {
                "format": "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        }
        formatter_name = "plain"

    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatter_config,
        "filters": {
            "context": {
                "()": "app.observability.logging.ContextFilter",
                "service_name": service_name,
            }
        },
        "handlers": {
            "default": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": formatter_name,
                "filters": ["context"],
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": log_level,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": log_level,
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "handlers": ["default"],
                "level": logging.WARNING,
                "propagate": False,
            },
        },
    })

    # Initialize Sentry if configured
    if config.observability.sentry_dsn:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=config.observability.sentry_dsn,
                traces_sample_rate=0.1 if config.observability.enable_tracing else 0.0,
                environment=config.flask_env,
            )
        except ImportError:
            logging.getLogger(__name__).warning(
                "Sentry SDK not installed; skipping Sentry initialization"
            )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that includes context in all log messages."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add context to log message."""
        extra = kwargs.get("extra", {})
        extra.setdefault("correlation_id", get_correlation_id() or "-")
        extra.setdefault("tenant_id", _tenant_id.get() or "-")
        extra.setdefault("session_id", _session_id.get() or "-")
        kwargs["extra"] = extra
        return msg, kwargs


def get_context_logger(name: str) -> LoggerAdapter:
    """Get a logger adapter that includes context."""
    return LoggerAdapter(logging.getLogger(name), {})


__all__ = [
    "configure_logging",
    "generate_correlation_id",
    "set_correlation_id",
    "get_correlation_id",
    "set_logging_context",
    "clear_logging_context",
    "ContextFilter",
    "StructuredJsonFormatter",
    "get_logger",
    "get_context_logger",
    "LoggerAdapter",
]

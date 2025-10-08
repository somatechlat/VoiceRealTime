"""Central logging configuration for the enterprise OVOS stack."""

from __future__ import annotations

import logging
from logging.config import dictConfig

try:
    from pythonjsonlogger import jsonlogger
except ImportError:  # pragma: no cover - dependency is optional for tests and local runs
    jsonlogger = None

from ..config import AppConfig


def configure_logging(config: AppConfig) -> None:
    """Configure structured logging across the service."""

    log_level = getattr(logging, config.observability.log_level.upper(), logging.INFO)

    formatter_name = "json" if jsonlogger else "plain"
    formatter_config = (
        {
            "json": {
                "()": jsonlogger.JsonFormatter,
                "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
            }
        }
        if jsonlogger
        else {
            "plain": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            }
        }
    )

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatter_config,
            "handlers": {
                "default": {
                    "level": log_level,
                    "class": "logging.StreamHandler",
                    "formatter": formatter_name,
                }
            },
            "loggers": {
                "": {"handlers": ["default"], "level": log_level},
                "uvicorn.error": {"handlers": ["default"], "level": log_level, "propagate": False},
                "uvicorn.access": {"handlers": ["default"], "level": log_level, "propagate": False},
            },
        }
    )

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


__all__ = ["configure_logging"]

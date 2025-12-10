"""Minimal configuration for workers - no Flask dependencies.

This module provides configuration classes that workers can import
without triggering Flask imports from app/__init__.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class RedisSettings(BaseModel):
    """Redis connection settings."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field("redis://localhost:6379/0", description="Redis connection URL")
    max_connections: int = Field(200, description="Maximum connections in pool")
    socket_timeout: float = Field(5.0, description="Socket timeout in seconds")
    socket_connect_timeout: float = Field(5.0, description="Connection timeout in seconds")
    retry_on_timeout: bool = Field(True, description="Retry on timeout")
    health_check_interval: int = Field(30, description="Health check interval in seconds")


__all__ = ["RedisSettings"]

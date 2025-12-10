"""Minimal configuration for workers - no Flask dependencies.

This module provides configuration classes that workers can import
without triggering Flask imports from app/__init__.py.

Re-exports RedisSettings from app.config to avoid duplicate class definitions.
"""

from __future__ import annotations

# Re-export RedisSettings from the canonical location to avoid type conflicts
from ..config import RedisSettings

__all__ = ["RedisSettings"]

"""Application-level dependency helpers for the AgentVoiceBox Engine."""

from __future__ import annotations

from typing import Optional

from flask import current_app

from .config import AppConfig, configure_app_from_env
from .services.connection_manager import ConnectionManager
from .services.distributed_rate_limiter import DistributedRateLimiter, RateLimitConfig
from .services.distributed_session import DistributedSessionManager
from .services.opa_client import OPAClient
from .services.redis_client import RedisClient
from .services.redis_streams import RedisStreamsClient
from .services.session_service import SessionService
from .services.token_service import TokenService
from .utils.database import create_session_factory


def get_app_config() -> AppConfig:
    cfg = current_app.extensions.get("app_config")
    if cfg is None:
        cfg = configure_app_from_env()
        current_app.extensions["app_config"] = cfg
    return cfg


def get_session_factory(config: AppConfig):
    session_factory = current_app.extensions.get("session_factory")
    if session_factory is None:
        session_factory = create_session_factory(config)
        current_app.extensions["session_factory"] = session_factory
    return session_factory


def get_session_service(config: AppConfig) -> SessionService:
    service = current_app.extensions.get("session_service")
    if service is None:
        session_factory = get_session_factory(config)
        service = SessionService(session_factory)
        current_app.extensions["session_service"] = service
    return service


def get_opa_client(config: AppConfig) -> OPAClient:
    opa_client = current_app.extensions.get("opa_client")
    if opa_client is None:
        opa_client = OPAClient(config)
        current_app.extensions["opa_client"] = opa_client
    return opa_client


def get_token_service() -> TokenService:
    token_service = current_app.extensions.get("token_service")
    if token_service is None:
        token_service = TokenService()
        current_app.extensions["token_service"] = token_service
    return token_service


def get_redis_client() -> Optional[RedisClient]:
    """Get the Redis client instance.

    Returns None if Redis is not configured or not connected.
    """
    return current_app.extensions.get("redis_client")


def get_rate_limiter() -> Optional[DistributedRateLimiter]:
    """Get the distributed rate limiter instance.

    Returns None if Redis is not available (falls back to in-memory).
    """
    rate_limiter = current_app.extensions.get("rate_limiter")
    if rate_limiter is None:
        redis_client = get_redis_client()
        if redis_client is not None:
            config = get_app_config()
            rate_limit_config = RateLimitConfig(
                requests_per_minute=config.security.rate_limits.requests_per_minute,
                tokens_per_minute=config.security.rate_limits.tokens_per_minute,
            )
            rate_limiter = DistributedRateLimiter(redis_client, rate_limit_config)
            current_app.extensions["rate_limiter"] = rate_limiter
    return rate_limiter


def get_distributed_session_manager() -> Optional[DistributedSessionManager]:
    """Get the distributed session manager instance.

    Returns None if Redis is not available (falls back to PostgreSQL).
    """
    manager = current_app.extensions.get("distributed_session_manager")
    if manager is None:
        redis_client = get_redis_client()
        if redis_client is not None:
            import os

            gateway_id = os.getenv("GATEWAY_ID", f"gateway-{os.getpid()}")
            manager = DistributedSessionManager(redis_client, gateway_id)
            current_app.extensions["distributed_session_manager"] = manager
    return manager


def get_connection_manager() -> Optional[ConnectionManager]:
    """Get the connection manager instance for tracking active connections."""
    return current_app.extensions.get("connection_manager")


def get_redis_streams_client() -> Optional[RedisStreamsClient]:
    """Get the Redis Streams client for worker communication."""
    return current_app.extensions.get("redis_streams_client")


__all__ = [
    "get_app_config",
    "get_session_factory",
    "get_session_service",
    "get_opa_client",
    "get_token_service",
    "get_redis_client",
    "get_rate_limiter",
    "get_distributed_session_manager",
    "get_connection_manager",
    "get_redis_streams_client",
]

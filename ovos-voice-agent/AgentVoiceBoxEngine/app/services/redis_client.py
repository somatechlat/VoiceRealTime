"""Redis client wrapper with connection pooling and automatic reconnection.

This module provides a production-grade Redis client for AgentVoiceBox with:
- Connection pooling (configurable max connections)
- Automatic reconnection on failure
- Health checking
- Async support via redis-py async API
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from ..config import RedisSettings

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client with connection pooling and automatic reconnection.
    
    This client wraps redis-py's async client with:
    - Connection pooling (default 50 connections)
    - Automatic reconnection with exponential backoff
    - Health check support
    - Graceful shutdown
    """

    def __init__(self, settings: RedisSettings) -> None:
        self._settings = settings
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish connection to Redis with pooling."""
        if self._connected:
            return

        try:
            self._pool = ConnectionPool.from_url(
                self._settings.url,
                max_connections=self._settings.max_connections,
                socket_timeout=self._settings.socket_timeout,
                socket_connect_timeout=self._settings.socket_connect_timeout,
                retry_on_timeout=self._settings.retry_on_timeout,
                health_check_interval=self._settings.health_check_interval,
                decode_responses=True,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
            
            # Verify connection
            await self._client.ping()
            self._connected = True
            logger.info("Redis connection established", extra={"url": self._settings.url})
            
        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to connect to Redis", extra={"error": str(e)})
            raise

    async def disconnect(self) -> None:
        """Gracefully close Redis connection."""
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()
            self._client = None
        
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        
        self._connected = False
        logger.info("Redis connection closed")

    async def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        if not self._client:
            return False
        try:
            await self._client.ping()
            return True
        except RedisError:
            return False

    @property
    def client(self) -> aioredis.Redis:
        """Get the underlying Redis client.
        
        Raises:
            RuntimeError: If not connected.
        """
        if not self._client or not self._connected:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._client

    async def _reconnect_with_backoff(self) -> None:
        """Attempt reconnection with exponential backoff."""
        backoff = 1.0
        max_backoff = 60.0
        
        while not self._connected:
            try:
                logger.info("Attempting Redis reconnection", extra={"backoff": backoff})
                await self.connect()
                logger.info("Redis reconnection successful")
                return
            except (ConnectionError, TimeoutError) as e:
                logger.warning(
                    "Redis reconnection failed, retrying",
                    extra={"error": str(e), "next_attempt_seconds": backoff}
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    def start_reconnect(self) -> None:
        """Start background reconnection task."""
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._connected = False
        self._reconnect_task = asyncio.create_task(self._reconnect_with_backoff())

    # Convenience methods that wrap common Redis operations

    async def get(self, key: str) -> Optional[str]:
        """Get a value from Redis."""
        try:
            return await self.client.get(key)
        except (ConnectionError, TimeoutError) as e:
            logger.error("Redis GET failed", extra={"key": key, "error": str(e)})
            self.start_reconnect()
            raise

    async def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set a value in Redis with optional expiration."""
        try:
            result = await self.client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)
            return result is True
        except (ConnectionError, TimeoutError) as e:
            logger.error("Redis SET failed", extra={"key": key, "error": str(e)})
            self.start_reconnect()
            raise

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        try:
            return await self.client.delete(*keys)
        except (ConnectionError, TimeoutError) as e:
            logger.error("Redis DELETE failed", extra={"keys": keys, "error": str(e)})
            self.start_reconnect()
            raise

    async def hset(self, name: str, mapping: dict[str, Any]) -> int:
        """Set multiple hash fields."""
        try:
            return await self.client.hset(name, mapping=mapping)
        except (ConnectionError, TimeoutError) as e:
            logger.error("Redis HSET failed", extra={"name": name, "error": str(e)})
            self.start_reconnect()
            raise

    async def hgetall(self, name: str) -> dict[str, str]:
        """Get all fields and values in a hash."""
        try:
            return await self.client.hgetall(name)
        except (ConnectionError, TimeoutError) as e:
            logger.error("Redis HGETALL failed", extra={"name": name, "error": str(e)})
            self.start_reconnect()
            raise

    async def expire(self, name: str, time: int) -> bool:
        """Set a key's time to live in seconds."""
        try:
            return await self.client.expire(name, time)
        except (ConnectionError, TimeoutError) as e:
            logger.error("Redis EXPIRE failed", extra={"name": name, "error": str(e)})
            self.start_reconnect()
            raise

    async def publish(self, channel: str, message: str) -> int:
        """Publish a message to a channel."""
        try:
            return await self.client.publish(channel, message)
        except (ConnectionError, TimeoutError) as e:
            logger.error("Redis PUBLISH failed", extra={"channel": channel, "error": str(e)})
            self.start_reconnect()
            raise

    async def subscribe(self, *channels: str) -> aioredis.client.PubSub:
        """Subscribe to one or more channels."""
        try:
            pubsub = self.client.pubsub()
            await pubsub.subscribe(*channels)
            return pubsub
        except (ConnectionError, TimeoutError) as e:
            logger.error("Redis SUBSCRIBE failed", extra={"channels": channels, "error": str(e)})
            self.start_reconnect()
            raise

    async def eval_script(
        self,
        script: str,
        keys: list[str],
        args: list[Any],
    ) -> Any:
        """Execute a Lua script."""
        try:
            return await self.client.eval(script, len(keys), *keys, *args)
        except (ConnectionError, TimeoutError) as e:
            logger.error("Redis EVAL connection failed", extra={"error": str(e)})
            self.start_reconnect()
            raise
        except RedisError as e:
            logger.error("Redis EVAL script error", extra={"error": str(e)})
            raise


# Global client instance (initialized on app startup)
_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """Get the global Redis client instance.
    
    Raises:
        RuntimeError: If client not initialized.
    """
    if _redis_client is None:
        raise RuntimeError("Redis client not initialized. Call init_redis_client() first.")
    return _redis_client


async def init_redis_client(settings: RedisSettings) -> RedisClient:
    """Initialize the global Redis client."""
    global _redis_client
    _redis_client = RedisClient(settings)
    await _redis_client.connect()
    return _redis_client


async def close_redis_client() -> None:
    """Close the global Redis client."""
    global _redis_client
    if _redis_client:
        await _redis_client.disconnect()
        _redis_client = None


__all__ = [
    "RedisClient",
    "get_redis_client",
    "init_redis_client",
    "close_redis_client",
]

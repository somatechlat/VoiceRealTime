"""Redis client for workers - no Flask dependencies.

This is a standalone Redis client that workers can use without
importing from the app package (which would trigger Flask loading).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from .worker_config import RedisSettings

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client with connection pooling and automatic reconnection."""

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
        """Get the underlying Redis client."""
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

    async def publish(self, channel: str, message: str) -> int:
        """Publish a message to a channel."""
        try:
            return await self.client.publish(channel, message)
        except (ConnectionError, TimeoutError) as e:
            logger.error("Redis PUBLISH failed", extra={"channel": channel, "error": str(e)})
            self.start_reconnect()
            raise


__all__ = ["RedisClient", "RedisSettings"]

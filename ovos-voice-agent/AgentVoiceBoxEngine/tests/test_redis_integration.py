"""Integration tests for Redis-backed services against real Redis.

Run with: python -m pytest tests/test_redis_integration.py -v
Requires: Redis running on localhost:6379 (or via Docker)
"""

import asyncio
import os
import sys
import time

import pytest
import pytest_asyncio

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import RedisSettings
from app.services.redis_client import RedisClient
from app.services.distributed_session import (
    DistributedSessionManager,
    Session,
    SessionConfig,
)
from app.services.distributed_rate_limiter import (
    DistributedRateLimiter,
    RateLimitConfig,
)


# Use Docker Redis URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Configure pytest-asyncio mode
pytestmark = pytest.mark.asyncio(loop_scope="function")


@pytest_asyncio.fixture
async def redis_client():
    """Create a Redis client connected to real Redis."""
    settings = RedisSettings(url=REDIS_URL)
    client = RedisClient(settings)
    await client.connect()
    yield client
    await client.disconnect()


@pytest_asyncio.fixture
async def session_manager(redis_client):
    """Create a session manager with real Redis."""
    manager = DistributedSessionManager(redis_client, "test-gateway-1")
    yield manager


@pytest_asyncio.fixture
async def rate_limiter(redis_client):
    """Create a rate limiter with real Redis."""
    config = RateLimitConfig(requests_per_minute=10, tokens_per_minute=100)
    limiter = DistributedRateLimiter(redis_client, config)
    yield limiter


class TestRedisClient:
    """Test Redis client against real Redis."""

    @pytest.mark.asyncio
    async def test_connect_and_ping(self, redis_client):
        """Test basic connection and health check."""
        assert await redis_client.health_check() is True

    @pytest.mark.asyncio
    async def test_set_and_get(self, redis_client):
        """Test basic set/get operations."""
        key = f"test:key:{time.time()}"
        value = "test_value"
        
        await redis_client.set(key, value, ex=60)
        result = await redis_client.get(key)
        
        assert result == value
        
        # Cleanup
        await redis_client.delete(key)

    @pytest.mark.asyncio
    async def test_hash_operations(self, redis_client):
        """Test hash set/get operations."""
        key = f"test:hash:{time.time()}"
        mapping = {"field1": "value1", "field2": "value2"}
        
        await redis_client.hset(key, mapping)
        result = await redis_client.hgetall(key)
        
        assert result == mapping
        
        # Cleanup
        await redis_client.delete(key)


class TestDistributedSessionManager:
    """Test session manager against real Redis."""

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test creating a session in Redis."""
        session_id = f"sess_test_{int(time.time())}"
        tenant_id = "test_tenant"
        
        session = await session_manager.create_session(
            session_id=session_id,
            tenant_id=tenant_id,
            project_id="test_project",
        )
        
        assert session.id == session_id
        assert session.tenant_id == tenant_id
        assert session.status == "created"
        
        # Cleanup
        await session_manager.close_session(session_id, tenant_id)

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager):
        """Test retrieving a session from Redis."""
        session_id = f"sess_test_{int(time.time())}"
        tenant_id = "test_tenant"
        
        # Create session
        await session_manager.create_session(
            session_id=session_id,
            tenant_id=tenant_id,
        )
        
        # Get session
        session = await session_manager.get_session(session_id, tenant_id)
        
        assert session is not None
        assert session.id == session_id
        
        # Cleanup
        await session_manager.close_session(session_id, tenant_id)

    @pytest.mark.asyncio
    async def test_update_session(self, session_manager):
        """Test updating a session in Redis."""
        session_id = f"sess_test_{int(time.time())}"
        tenant_id = "test_tenant"
        
        # Create session
        await session_manager.create_session(
            session_id=session_id,
            tenant_id=tenant_id,
        )
        
        # Update session
        updated = await session_manager.update_session(
            session_id=session_id,
            tenant_id=tenant_id,
            updates={"status": "connected", "config": {"voice": "af_bella"}},
        )
        
        assert updated is not None
        assert updated.status == "connected"
        assert updated.config.voice == "af_bella"
        
        # Cleanup
        await session_manager.close_session(session_id, tenant_id)

    @pytest.mark.asyncio
    async def test_heartbeat(self, session_manager):
        """Test session heartbeat refresh."""
        session_id = f"sess_test_{int(time.time())}"
        tenant_id = "test_tenant"
        
        # Create session
        await session_manager.create_session(
            session_id=session_id,
            tenant_id=tenant_id,
        )
        
        # Heartbeat
        result = await session_manager.heartbeat(session_id, tenant_id)
        assert result is True
        
        # Cleanup
        await session_manager.close_session(session_id, tenant_id)

    @pytest.mark.asyncio
    async def test_conversation_items(self, session_manager):
        """Test appending and retrieving conversation items."""
        session_id = f"sess_test_{int(time.time())}"
        tenant_id = "test_tenant"
        
        # Create session
        await session_manager.create_session(
            session_id=session_id,
            tenant_id=tenant_id,
        )
        
        # Append items
        item1 = {"role": "user", "content": "Hello"}
        item2 = {"role": "assistant", "content": "Hi there!"}
        
        await session_manager.append_conversation_item(session_id, tenant_id, item1)
        await session_manager.append_conversation_item(session_id, tenant_id, item2)
        
        # Get items
        items = await session_manager.get_conversation_items(session_id, tenant_id)
        
        assert len(items) == 2
        assert items[0]["role"] == "user"
        assert items[1]["role"] == "assistant"
        
        # Cleanup
        await session_manager.close_session(session_id, tenant_id)


class TestDistributedRateLimiter:
    """Test rate limiter against real Redis."""

    @pytest.mark.asyncio
    async def test_check_and_consume_allowed(self, rate_limiter):
        """Test rate limit check when under limit."""
        tenant_id = f"tenant_{int(time.time())}"
        identifier = "test_session"
        
        result = await rate_limiter.check_and_consume(
            tenant_id=tenant_id,
            identifier=identifier,
            requests=1,
            tokens=10,
        )
        
        assert result.allowed is True
        assert result.requests_remaining == 9  # 10 - 1
        assert result.tokens_remaining == 90  # 100 - 10

    @pytest.mark.asyncio
    async def test_check_and_consume_exceeded(self, rate_limiter):
        """Test rate limit check when limit exceeded."""
        tenant_id = f"tenant_{int(time.time())}"
        identifier = "test_session"
        
        # Consume all requests
        for _ in range(10):
            await rate_limiter.check_and_consume(
                tenant_id=tenant_id,
                identifier=identifier,
                requests=1,
                tokens=0,
            )
        
        # Next request should be denied
        result = await rate_limiter.check_and_consume(
            tenant_id=tenant_id,
            identifier=identifier,
            requests=1,
            tokens=0,
        )
        
        assert result.allowed is False
        assert result.requests_remaining == 0

    @pytest.mark.asyncio
    async def test_get_limits(self, rate_limiter):
        """Test getting current limits without consuming."""
        import uuid
        tenant_id = f"tenant_getlimits_{uuid.uuid4().hex[:8]}"
        identifier = "test_session"
        
        # Consume some quota
        await rate_limiter.check_and_consume(
            tenant_id=tenant_id,
            identifier=identifier,
            requests=3,
            tokens=30,
        )
        
        # Get limits
        result = await rate_limiter.get_limits(tenant_id, identifier)
        
        assert result.requests_remaining == 7  # 10 - 3
        assert result.tokens_remaining == 70  # 100 - 30

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, rate_limiter):
        """Test that rate limits are isolated per tenant."""
        tenant1 = f"tenant1_{int(time.time())}"
        tenant2 = f"tenant2_{int(time.time())}"
        identifier = "test_session"
        
        # Consume all quota for tenant1
        for _ in range(10):
            await rate_limiter.check_and_consume(
                tenant_id=tenant1,
                identifier=identifier,
                requests=1,
                tokens=0,
            )
        
        # Tenant1 should be rate limited
        result1 = await rate_limiter.check_and_consume(
            tenant_id=tenant1,
            identifier=identifier,
            requests=1,
            tokens=0,
        )
        assert result1.allowed is False
        
        # Tenant2 should still have quota
        result2 = await rate_limiter.check_and_consume(
            tenant_id=tenant2,
            identifier=identifier,
            requests=1,
            tokens=0,
        )
        assert result2.allowed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Distributed rate limiter using Redis with Lua scripts.

This module provides a Redis-backed sliding window rate limiter for AgentVoiceBox:
- Atomic check-and-consume using Lua scripts
- Support for both request and token limits
- Per-tenant limit overrides
- Tenant isolation via key namespacing

The sliding window algorithm uses Redis sorted sets to track requests/tokens
within a time window, providing accurate rate limiting across distributed gateways.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from .redis_client import RedisClient

logger = logging.getLogger(__name__)


# Lua script for atomic rate limiting
# Uses sliding window algorithm with sorted sets
RATE_LIMIT_LUA_SCRIPT = """
-- KEYS[1] = rate limit key prefix
-- ARGV[1] = current timestamp (ms)
-- ARGV[2] = window size (ms)
-- ARGV[3] = max requests
-- ARGV[4] = max tokens
-- ARGV[5] = requests to consume
-- ARGV[6] = tokens to consume

local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local max_tokens = tonumber(ARGV[4])
local req_consume = tonumber(ARGV[5])
local tok_consume = tonumber(ARGV[6])

-- Remove expired entries
local window_start = now - window
redis.call('ZREMRANGEBYSCORE', key .. ':req', '-inf', window_start)
redis.call('ZREMRANGEBYSCORE', key .. ':tok', '-inf', window_start)

-- Count current usage
local req_count = redis.call('ZCARD', key .. ':req')
local tok_count = redis.call('ZCARD', key .. ':tok')

-- Check limits
if req_count + req_consume > max_requests then
    return {0, max_requests - req_count, max_tokens - tok_count, window}
end
if tok_count + tok_consume > max_tokens then
    return {0, max_requests - req_count, max_tokens - tok_count, window}
end

-- Consume quota
for i = 1, req_consume do
    redis.call('ZADD', key .. ':req', now, now .. ':' .. i .. ':' .. math.random(1000000))
end
for i = 1, tok_consume do
    redis.call('ZADD', key .. ':tok', now, now .. ':' .. i .. ':' .. math.random(1000000))
end

-- Set expiry
redis.call('EXPIRE', key .. ':req', math.ceil(window / 1000) + 1)
redis.call('EXPIRE', key .. ':tok', math.ceil(window / 1000) + 1)

return {1, max_requests - req_count - req_consume, max_tokens - tok_count - tok_consume, window}
"""

# Lua script for getting current limits without consuming
GET_LIMITS_LUA_SCRIPT = """
-- KEYS[1] = rate limit key prefix
-- ARGV[1] = current timestamp (ms)
-- ARGV[2] = window size (ms)
-- ARGV[3] = max requests
-- ARGV[4] = max tokens

local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local max_tokens = tonumber(ARGV[4])

-- Remove expired entries
local window_start = now - window
redis.call('ZREMRANGEBYSCORE', key .. ':req', '-inf', window_start)
redis.call('ZREMRANGEBYSCORE', key .. ':tok', '-inf', window_start)

-- Count current usage
local req_count = redis.call('ZCARD', key .. ':req')
local tok_count = redis.call('ZCARD', key .. ':tok')

return {max_requests - req_count, max_tokens - tok_count, window}
"""


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_minute: int = 100
    tokens_per_minute: int = 100000
    window_ms: int = 60000  # 1 minute in milliseconds


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    requests_remaining: int
    tokens_remaining: int
    reset_ms: int

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "requests_remaining": self.requests_remaining,
            "tokens_remaining": self.tokens_remaining,
            "reset_ms": self.reset_ms,
        }


class DistributedRateLimiter:
    """Redis-based sliding window rate limiter.

    Uses Lua scripts for atomic check-and-consume operations across
    distributed gateway instances.

    Features:
    - Sliding window algorithm for accurate rate limiting
    - Atomic operations via Lua scripts (<5ms latency)
    - Per-tenant limit overrides
    - Tenant isolation via key namespacing
    """

    KEY_PREFIX = "ratelimit"

    def __init__(
        self,
        redis_client: RedisClient,
        default_config: Optional[RateLimitConfig] = None,
    ) -> None:
        self._redis = redis_client
        self._default_config = default_config or RateLimitConfig()
        self._tenant_configs: dict[str, RateLimitConfig] = {}
        self._script_sha: Optional[str] = None
        self._get_limits_sha: Optional[str] = None

    def _rate_limit_key(self, tenant_id: str, identifier: str) -> str:
        """Generate namespaced rate limit key.

        Args:
            tenant_id: Tenant ID for isolation
            identifier: Additional identifier (e.g., session_id, user_id)
        """
        return f"{self.KEY_PREFIX}:{tenant_id}:{identifier}"

    def set_tenant_limits(self, tenant_id: str, config: RateLimitConfig) -> None:
        """Set custom rate limits for a tenant.

        Args:
            tenant_id: Tenant ID
            config: Custom rate limit configuration
        """
        self._tenant_configs[tenant_id] = config
        logger.info(
            "Tenant rate limits updated",
            extra={
                "tenant_id": tenant_id,
                "requests_per_minute": config.requests_per_minute,
                "tokens_per_minute": config.tokens_per_minute,
            },
        )

    def get_tenant_config(self, tenant_id: str) -> RateLimitConfig:
        """Get rate limit config for a tenant."""
        return self._tenant_configs.get(tenant_id, self._default_config)

    async def check_and_consume(
        self,
        tenant_id: str,
        identifier: str,
        requests: int = 1,
        tokens: int = 0,
    ) -> RateLimitResult:
        """Atomically check limits and consume quota.

        Args:
            tenant_id: Tenant ID for isolation
            identifier: Additional identifier (e.g., session_id)
            requests: Number of requests to consume
            tokens: Number of tokens to consume

        Returns:
            RateLimitResult with allowed status and remaining quota
        """
        config = self.get_tenant_config(tenant_id)
        key = self._rate_limit_key(tenant_id, identifier)
        now_ms = int(time.time() * 1000)

        try:
            result = await self._redis.eval_script(
                RATE_LIMIT_LUA_SCRIPT,
                keys=[key],
                args=[
                    now_ms,
                    config.window_ms,
                    config.requests_per_minute,
                    config.tokens_per_minute,
                    requests,
                    tokens,
                ],
            )

            allowed = bool(result[0])
            requests_remaining = int(result[1])
            tokens_remaining = int(result[2])
            reset_ms = int(result[3])

            if not allowed:
                logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "tenant_id": tenant_id,
                        "identifier": identifier,
                        "requests_remaining": requests_remaining,
                        "tokens_remaining": tokens_remaining,
                    },
                )

            return RateLimitResult(
                allowed=allowed,
                requests_remaining=max(0, requests_remaining),
                tokens_remaining=max(0, tokens_remaining),
                reset_ms=reset_ms,
            )

        except Exception as e:
            logger.error(
                "Rate limit check failed, failing open",
                extra={"tenant_id": tenant_id, "error": str(e)},
            )
            # Fail open - allow request but log error
            return RateLimitResult(
                allowed=True,
                requests_remaining=config.requests_per_minute,
                tokens_remaining=config.tokens_per_minute,
                reset_ms=config.window_ms,
            )

    async def get_limits(
        self,
        tenant_id: str,
        identifier: str,
    ) -> RateLimitResult:
        """Get current limit status without consuming.

        Args:
            tenant_id: Tenant ID for isolation
            identifier: Additional identifier

        Returns:
            RateLimitResult with current remaining quota
        """
        config = self.get_tenant_config(tenant_id)
        key = self._rate_limit_key(tenant_id, identifier)
        now_ms = int(time.time() * 1000)

        try:
            result = await self._redis.eval_script(
                GET_LIMITS_LUA_SCRIPT,
                keys=[key],
                args=[
                    now_ms,
                    config.window_ms,
                    config.requests_per_minute,
                    config.tokens_per_minute,
                ],
            )

            return RateLimitResult(
                allowed=True,
                requests_remaining=max(0, int(result[0])),
                tokens_remaining=max(0, int(result[1])),
                reset_ms=int(result[2]),
            )

        except Exception as e:
            logger.error("Get limits failed", extra={"tenant_id": tenant_id, "error": str(e)})
            return RateLimitResult(
                allowed=True,
                requests_remaining=config.requests_per_minute,
                tokens_remaining=config.tokens_per_minute,
                reset_ms=config.window_ms,
            )

    async def reset_limits(self, tenant_id: str, identifier: str) -> bool:
        """Reset rate limits for a specific key.

        Args:
            tenant_id: Tenant ID
            identifier: Additional identifier

        Returns:
            True if limits were reset
        """
        key = self._rate_limit_key(tenant_id, identifier)
        try:
            await self._redis.delete(f"{key}:req", f"{key}:tok")
            logger.info(
                "Rate limits reset", extra={"tenant_id": tenant_id, "identifier": identifier}
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to reset rate limits", extra={"tenant_id": tenant_id, "error": str(e)}
            )
            return False


def count_tokens(text: str) -> int:
    """Approximate token count for rate limiting.

    Uses simple heuristic: ~4 characters per token.
    For production, consider using tiktoken for accurate counts.
    """
    return max(1, len(text) // 4)


__all__ = [
    "DistributedRateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    "count_tokens",
]

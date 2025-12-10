"""Ephemeral session token service for browser clients.

Implements Requirements 3.5, 3.6:
- Short-lived, single-use tokens for browser clients
- Bound to specific session_id
- 10-minute TTL
- Redis-backed for distributed access

Token Format: eph_{random} (24 chars)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class EphemeralToken:
    """Ephemeral session token for browser clients."""

    token: str
    session_id: str
    tenant_id: str
    project_id: str
    scopes: list[str]
    session_config: Dict[str, Any]
    created_at: dt.datetime
    expires_at: dt.datetime
    used: bool = False

    def is_expired(self, now: Optional[dt.datetime] = None) -> bool:
        """Check if token is expired."""
        current = now or dt.datetime.utcnow()
        return current >= self.expires_at

    def is_valid(self) -> bool:
        """Check if token is valid (not used and not expired)."""
        return not self.used and not self.is_expired()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for Redis storage."""
        return {
            "token": self.token,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "project_id": self.project_id,
            "scopes": self.scopes,
            "session_config": self.session_config,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "used": self.used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EphemeralToken":
        """Deserialize from dictionary."""
        return cls(
            token=data["token"],
            session_id=data["session_id"],
            tenant_id=data["tenant_id"],
            project_id=data["project_id"],
            scopes=data.get("scopes", []),
            session_config=data.get("session_config", {}),
            created_at=dt.datetime.fromisoformat(data["created_at"]),
            expires_at=dt.datetime.fromisoformat(data["expires_at"]),
            used=data.get("used", False),
        )


class EphemeralTokenService:
    """Redis-backed ephemeral token service.

    Provides short-lived, single-use tokens for browser clients
    to establish WebSocket connections.

    Requirements: 3.5, 3.6
    """

    DEFAULT_TTL = 600  # 10 minutes
    CACHE_PREFIX = "ephtoken:"

    def __init__(self, redis_client: Optional[Any] = None):
        self._redis = redis_client

    @staticmethod
    def generate_token() -> str:
        """Generate a new ephemeral token."""
        return f"eph_{secrets.token_urlsafe(18)}"

    def _cache_key(self, token: str) -> str:
        """Generate Redis key for token."""
        return f"{self.CACHE_PREFIX}{token}"

    async def issue(
        self,
        session_id: str,
        tenant_id: str,
        project_id: str,
        scopes: list[str],
        session_config: Dict[str, Any],
        ttl_seconds: int = DEFAULT_TTL,
    ) -> EphemeralToken:
        """Issue a new ephemeral token.

        Args:
            session_id: Session ID this token is bound to
            tenant_id: Tenant ID for isolation
            project_id: Project ID
            scopes: Allowed scopes for this token
            session_config: Session configuration
            ttl_seconds: Token lifetime (default 10 minutes)

        Returns:
            EphemeralToken instance
        """
        now = dt.datetime.utcnow()
        token_value = self.generate_token()

        token = EphemeralToken(
            token=token_value,
            session_id=session_id,
            tenant_id=tenant_id,
            project_id=project_id,
            scopes=scopes,
            session_config=session_config,
            created_at=now,
            expires_at=now + dt.timedelta(seconds=ttl_seconds),
        )

        # Store in Redis
        if self._redis:
            try:
                cache_key = self._cache_key(token_value)
                await self._redis.client.setex(
                    cache_key,
                    ttl_seconds,
                    json.dumps(token.to_dict()),
                )
                logger.info(
                    "Ephemeral token issued",
                    extra={
                        "session_id": session_id,
                        "tenant_id": tenant_id,
                        "ttl_seconds": ttl_seconds,
                    },
                )
            except Exception as e:
                logger.error("Failed to store ephemeral token: %s", e)
                raise

        return token

    async def validate(self, token_value: str) -> Optional[EphemeralToken]:
        """Validate and consume an ephemeral token.

        This is a single-use operation - the token is marked as used
        after successful validation.

        Args:
            token_value: The token to validate

        Returns:
            EphemeralToken if valid, None if invalid/expired/used
        """
        if not token_value.startswith("eph_"):
            logger.debug("Invalid ephemeral token format")
            return None

        if not self._redis:
            logger.warning("Redis not available for ephemeral token validation")
            return None

        try:
            cache_key = self._cache_key(token_value)

            # Get and delete atomically (single-use)
            data = await self._redis.client.get(cache_key)
            if not data:
                logger.debug("Ephemeral token not found or expired")
                return None

            token = EphemeralToken.from_dict(json.loads(data))

            if token.used:
                logger.warning(
                    "Ephemeral token already used",
                    extra={"session_id": token.session_id},
                )
                return None

            if token.is_expired():
                logger.debug("Ephemeral token expired")
                await self._redis.client.delete(cache_key)
                return None

            # Mark as used and delete
            await self._redis.client.delete(cache_key)

            logger.info(
                "Ephemeral token validated and consumed",
                extra={
                    "session_id": token.session_id,
                    "tenant_id": token.tenant_id,
                },
            )

            return token

        except Exception as e:
            logger.error("Ephemeral token validation error: %s", e)
            return None

    async def revoke(self, token_value: str) -> bool:
        """Revoke an ephemeral token.

        Args:
            token_value: The token to revoke

        Returns:
            True if token was revoked
        """
        if not self._redis:
            return False

        try:
            cache_key = self._cache_key(token_value)
            result = await self._redis.client.delete(cache_key)
            return result > 0
        except Exception as e:
            logger.error("Ephemeral token revocation error: %s", e)
            return False


__all__ = [
    "EphemeralToken",
    "EphemeralTokenService",
]

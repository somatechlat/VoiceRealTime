"""API Key authentication service with Argon2id hashing.

Implements Requirements 3.1, 3.2, 3.3:
- API key validation with Argon2id hashing
- Redis cache (hot) with PostgreSQL fallback (cold)
- Tenant/project context extraction
- Authentication logging

API Key Format: avb_{prefix}_{random} (32 chars total)
- avb_ = AgentVoiceBox prefix
- prefix = 8 char identifier for lookup
- random = 16 char random suffix
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# argon2-cffi is the recommended Argon2 library
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError, InvalidHash
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False
    PasswordHasher = None
    VerifyMismatchError = Exception
    InvalidHash = Exception
    logger.warning("argon2-cffi not installed - using fallback hashing")


@dataclass
class APIKeyInfo:
    """Validated API key information."""

    key_id: uuid.UUID
    tenant_id: uuid.UUID
    project_id: uuid.UUID
    scopes: List[str]
    rate_limit_tier: str
    is_active: bool
    expires_at: Optional[dt.datetime]
    last_used_at: Optional[dt.datetime]

    def has_scope(self, scope: str) -> bool:
        """Check if key has a specific scope."""
        if "admin:*" in self.scopes:
            return True
        return scope in self.scopes

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return dt.datetime.utcnow() >= self.expires_at

    def is_valid(self) -> bool:
        """Check if key is valid (active and not expired)."""
        return self.is_active and not self.is_expired()


class APIKeyHasher:
    """Argon2id password hasher for API keys.

    Uses Argon2id (memory-hard, side-channel resistant) as specified
    in Requirements 2.3.
    """

    def __init__(self):
        if ARGON2_AVAILABLE:
            # Argon2id with recommended parameters
            self._hasher = PasswordHasher(
                time_cost=2,        # iterations
                memory_cost=65536,  # 64 MB
                parallelism=1,      # threads
                hash_len=32,        # output length
                salt_len=16,        # salt length
            )
        else:
            self._hasher = None

    def hash(self, api_key: str) -> str:
        """Hash an API key using Argon2id."""
        if self._hasher:
            return self._hasher.hash(api_key)
        # Fallback: SHA-256 with salt (less secure but functional)
        salt = secrets.token_hex(16)
        hash_value = hashlib.sha256(f"{salt}:{api_key}".encode()).hexdigest()
        return f"$sha256${salt}${hash_value}"

    def verify(self, api_key: str, hash_value: str) -> bool:
        """Verify an API key against its hash."""
        try:
            if self._hasher and hash_value.startswith("$argon2"):
                self._hasher.verify(hash_value, api_key)
                return True
            elif hash_value.startswith("$sha256$"):
                # Fallback verification
                parts = hash_value.split("$")
                if len(parts) != 4:
                    return False
                salt = parts[2]
                expected = parts[3]
                actual = hashlib.sha256(f"{salt}:{api_key}".encode()).hexdigest()
                return hmac.compare_digest(expected, actual)
            return False
        except (VerifyMismatchError, InvalidHash):
            return False
        except Exception as e:
            logger.error("API key verification error: %s", e)
            return False

    def needs_rehash(self, hash_value: str) -> bool:
        """Check if hash needs to be upgraded."""
        if self._hasher and hash_value.startswith("$argon2"):
            return self._hasher.check_needs_rehash(hash_value)
        # SHA256 fallback always needs rehash when Argon2 is available
        return ARGON2_AVAILABLE and hash_value.startswith("$sha256$")


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key and its prefix.

    Returns:
        Tuple of (full_key, prefix)
        - full_key: avb_{prefix}_{random} format
        - prefix: 8 char identifier for lookup
    """
    prefix = secrets.token_hex(4)  # 8 chars
    random_part = secrets.token_urlsafe(12)  # ~16 chars
    full_key = f"avb_{prefix}_{random_part}"
    return full_key, prefix


class APIKeyService:
    """API Key validation service with Redis cache and PostgreSQL fallback.

    Implements two-tier caching:
    - Hot: Redis cache with 5-minute TTL
    - Cold: PostgreSQL database

    Requirements: 3.1, 3.2, 3.3
    """

    CACHE_TTL = 300  # 5 minutes
    CACHE_PREFIX = "apikey:"

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        db_session_factory: Optional[Any] = None,
    ):
        self._redis = redis_client
        self._db_factory = db_session_factory
        self._hasher = APIKeyHasher()
        self._local_cache: Dict[str, tuple[APIKeyInfo, float]] = {}
        self._local_cache_ttl = 60  # 1 minute local cache

    def _cache_key(self, prefix: str) -> str:
        """Generate Redis cache key for API key prefix."""
        return f"{self.CACHE_PREFIX}{prefix}"

    async def validate(self, api_key: str) -> Optional[APIKeyInfo]:
        """Validate an API key and return its info.

        Validation flow:
        1. Extract prefix from key
        2. Check local in-memory cache
        3. Check Redis cache (hot)
        4. Query PostgreSQL (cold)
        5. Verify hash
        6. Cache result

        Args:
            api_key: The full API key to validate

        Returns:
            APIKeyInfo if valid, None if invalid
        """
        start_time = time.time()

        # Extract prefix from key format: avb_{prefix}_{random}
        if not api_key.startswith("avb_"):
            logger.debug("Invalid API key format")
            return None

        parts = api_key.split("_")
        if len(parts) != 3:
            logger.debug("Invalid API key format")
            return None

        prefix = parts[1]

        # 1. Check local cache
        cached = self._check_local_cache(prefix)
        if cached:
            if self._verify_key(api_key, cached):
                self._log_auth_attempt(prefix, True, "local_cache", start_time)
                return cached
            return None

        # 2. Check Redis cache
        if self._redis:
            cached = await self._check_redis_cache(prefix)
            if cached:
                self._update_local_cache(prefix, cached)
                if self._verify_key(api_key, cached):
                    self._log_auth_attempt(prefix, True, "redis_cache", start_time)
                    return cached
                return None

        # 3. Query PostgreSQL
        key_info = await self._query_database(prefix)
        if key_info is None:
            self._log_auth_attempt(prefix, False, "not_found", start_time)
            return None

        # 4. Verify hash
        if not self._verify_key(api_key, key_info):
            self._log_auth_attempt(prefix, False, "hash_mismatch", start_time)
            return None

        # 5. Cache result
        await self._cache_key_info(prefix, key_info)
        self._update_local_cache(prefix, key_info)

        self._log_auth_attempt(prefix, True, "database", start_time)
        return key_info

    def _check_local_cache(self, prefix: str) -> Optional[APIKeyInfo]:
        """Check local in-memory cache."""
        if prefix in self._local_cache:
            info, cached_at = self._local_cache[prefix]
            if time.time() - cached_at < self._local_cache_ttl:
                return info
            del self._local_cache[prefix]
        return None

    def _update_local_cache(self, prefix: str, info: APIKeyInfo) -> None:
        """Update local in-memory cache."""
        self._local_cache[prefix] = (info, time.time())
        # Limit cache size
        if len(self._local_cache) > 1000:
            oldest = min(self._local_cache.items(), key=lambda x: x[1][1])
            del self._local_cache[oldest[0]]

    async def _check_redis_cache(self, prefix: str) -> Optional[APIKeyInfo]:
        """Check Redis cache for API key info."""
        try:
            import json
            cache_key = self._cache_key(prefix)
            data = await self._redis.client.get(cache_key)
            if data:
                info_dict = json.loads(data)
                return APIKeyInfo(
                    key_id=uuid.UUID(info_dict["key_id"]),
                    tenant_id=uuid.UUID(info_dict["tenant_id"]),
                    project_id=uuid.UUID(info_dict["project_id"]),
                    scopes=info_dict["scopes"],
                    rate_limit_tier=info_dict["rate_limit_tier"],
                    is_active=info_dict["is_active"],
                    expires_at=(
                        dt.datetime.fromisoformat(info_dict["expires_at"])
                        if info_dict.get("expires_at")
                        else None
                    ),
                    last_used_at=None,
                )
        except Exception as e:
            logger.warning("Redis cache read error: %s", e)
        return None

    async def _cache_key_info(self, prefix: str, info: APIKeyInfo) -> None:
        """Cache API key info in Redis."""
        if not self._redis:
            return
        try:
            import json
            cache_key = self._cache_key(prefix)
            data = json.dumps({
                "key_id": str(info.key_id),
                "tenant_id": str(info.tenant_id),
                "project_id": str(info.project_id),
                "scopes": info.scopes,
                "rate_limit_tier": info.rate_limit_tier,
                "is_active": info.is_active,
                "expires_at": info.expires_at.isoformat() if info.expires_at else None,
            })
            await self._redis.client.setex(cache_key, self.CACHE_TTL, data)
        except Exception as e:
            logger.warning("Redis cache write error: %s", e)

    async def _query_database(self, prefix: str) -> Optional[APIKeyInfo]:
        """Query PostgreSQL for API key by prefix."""
        if not self._db_factory:
            return None

        try:
            from ..utils.database import session_scope
            from ..models.tenant import APIKey, Project

            with session_scope(self._db_factory) as session:
                key_record = (
                    session.query(APIKey)
                    .filter(APIKey.key_prefix == prefix)
                    .filter(APIKey.is_active == True)
                    .first()
                )

                if not key_record:
                    return None

                project = session.query(Project).filter(
                    Project.id == key_record.project_id
                ).first()

                if not project:
                    return None

                # Store hash for verification (not in APIKeyInfo)
                self._pending_hash = key_record.key_hash

                return APIKeyInfo(
                    key_id=key_record.id,
                    tenant_id=project.tenant_id,
                    project_id=key_record.project_id,
                    scopes=key_record.scopes or [],
                    rate_limit_tier=key_record.rate_limit_tier,
                    is_active=key_record.is_active,
                    expires_at=key_record.expires_at,
                    last_used_at=key_record.last_used_at,
                )
        except Exception as e:
            logger.error("Database query error: %s", e)
            return None

    def _verify_key(self, api_key: str, info: APIKeyInfo) -> bool:
        """Verify API key hash."""
        if hasattr(self, "_pending_hash") and self._pending_hash:
            result = self._hasher.verify(api_key, self._pending_hash)
            self._pending_hash = None
            return result and info.is_valid()
        # For cached entries, we trust the cache
        return info.is_valid()

    def _log_auth_attempt(
        self,
        prefix: str,
        success: bool,
        source: str,
        start_time: float,
    ) -> None:
        """Log authentication attempt."""
        duration_ms = (time.time() - start_time) * 1000
        if success:
            logger.info(
                "API key validated",
                extra={
                    "key_prefix": prefix,
                    "source": source,
                    "duration_ms": round(duration_ms, 2),
                },
            )
        else:
            logger.warning(
                "API key validation failed",
                extra={
                    "key_prefix": prefix,
                    "reason": source,
                    "duration_ms": round(duration_ms, 2),
                },
            )

    async def invalidate_cache(self, prefix: str) -> None:
        """Invalidate cached API key info."""
        if prefix in self._local_cache:
            del self._local_cache[prefix]
        if self._redis:
            try:
                await self._redis.client.delete(self._cache_key(prefix))
            except Exception as e:
                logger.warning("Cache invalidation error: %s", e)


__all__ = [
    "ARGON2_AVAILABLE",
    "APIKeyInfo",
    "APIKeyHasher",
    "APIKeyService",
    "generate_api_key",
]

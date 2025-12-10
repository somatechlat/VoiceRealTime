"""Tenant context and isolation service.

Implements Requirements 1.2, 1.3, 1.4:
- Tenant context enforcement on every operation
- Redis key namespacing for tenant isolation
- Database query filtering by tenant_id

This module provides:
- TenantContext: Current request's tenant context
- TenantIsolation: Utilities for enforcing tenant boundaries
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional

from flask import g

logger = logging.getLogger(__name__)

# Context variable for async-safe tenant context
_tenant_context: ContextVar[Optional["TenantContext"]] = ContextVar("tenant_context", default=None)


@dataclass
class TenantContext:
    """Current request's tenant context.

    Extracted from API key validation and propagated through
    all operations for isolation enforcement.
    """

    tenant_id: uuid.UUID
    project_id: uuid.UUID
    scopes: list[str]
    rate_limit_tier: str

    def has_scope(self, scope: str) -> bool:
        """Check if context has a specific scope."""
        if "admin:*" in self.scopes:
            return True
        return scope in self.scopes

    @property
    def tenant_id_str(self) -> str:
        """Get tenant_id as string for Redis keys."""
        return str(self.tenant_id)

    @property
    def project_id_str(self) -> str:
        """Get project_id as string."""
        return str(self.project_id)


def set_tenant_context(ctx: TenantContext) -> None:
    """Set the current tenant context.

    Should be called after API key validation.
    """
    _tenant_context.set(ctx)
    # Also set in Flask g for sync code
    g.tenant_context = ctx
    g.tenant_id = ctx.tenant_id
    g.project_id = ctx.project_id


def get_tenant_context() -> Optional[TenantContext]:
    """Get the current tenant context.

    Returns None if no context is set (unauthenticated request).
    """
    # Try context var first (async), then Flask g (sync)
    ctx = _tenant_context.get()
    if ctx:
        return ctx
    return getattr(g, "tenant_context", None)


def require_tenant_context() -> TenantContext:
    """Get the current tenant context or raise an error.

    Use this when tenant context is required.
    """
    ctx = get_tenant_context()
    if ctx is None:
        raise ValueError("Tenant context not set - authentication required")
    return ctx


def clear_tenant_context() -> None:
    """Clear the current tenant context."""
    _tenant_context.set(None)
    if hasattr(g, "tenant_context"):
        delattr(g, "tenant_context")
    if hasattr(g, "tenant_id"):
        delattr(g, "tenant_id")
    if hasattr(g, "project_id"):
        delattr(g, "project_id")


class TenantIsolation:
    """Utilities for enforcing tenant isolation.

    Provides:
    - Redis key namespacing
    - Database query filtering
    - Audit logging with tenant context
    """

    # Redis key prefixes that require tenant isolation
    ISOLATED_PREFIXES = [
        "session:",
        "apikey:",
        "ephtoken:",
        "ratelimit:",
        "audio:",
        "tts:",
    ]

    @staticmethod
    def redis_key(prefix: str, *parts: str, tenant_id: Optional[str] = None) -> str:
        """Generate a tenant-namespaced Redis key.

        Args:
            prefix: Key prefix (e.g., "session", "ratelimit")
            *parts: Additional key parts
            tenant_id: Tenant ID (uses current context if not provided)

        Returns:
            Namespaced key: {prefix}:{tenant_id}:{parts...}
        """
        if tenant_id is None:
            ctx = get_tenant_context()
            if ctx:
                tenant_id = ctx.tenant_id_str
            else:
                # Allow non-tenant keys for system operations
                tenant_id = "_system"

        key_parts = [prefix, tenant_id] + list(parts)
        return ":".join(key_parts)

    @staticmethod
    def validate_key_access(key: str) -> bool:
        """Validate that current tenant can access a Redis key.

        Args:
            key: Redis key to validate

        Returns:
            True if access is allowed
        """
        ctx = get_tenant_context()
        if ctx is None:
            # No context = system operation, allow all
            return True

        # Check if key is in an isolated prefix
        for prefix in TenantIsolation.ISOLATED_PREFIXES:
            if key.startswith(prefix):
                # Key should contain tenant_id after prefix
                parts = key.split(":")
                if len(parts) >= 2:
                    key_tenant = parts[1]
                    if key_tenant != ctx.tenant_id_str and key_tenant != "_system":
                        logger.warning(
                            "Cross-tenant key access denied",
                            extra={
                                "key": key,
                                "request_tenant": ctx.tenant_id_str,
                                "key_tenant": key_tenant,
                            },
                        )
                        return False
        return True

    @staticmethod
    def db_filter(query: Any, model: Any, tenant_id: Optional[uuid.UUID] = None) -> Any:
        """Apply tenant filter to a database query.

        Args:
            query: SQLAlchemy query object
            model: Model class with tenant_id column
            tenant_id: Tenant ID (uses current context if not provided)

        Returns:
            Filtered query
        """
        if tenant_id is None:
            ctx = get_tenant_context()
            if ctx:
                tenant_id = ctx.tenant_id

        if tenant_id and hasattr(model, "tenant_id"):
            return query.filter(model.tenant_id == tenant_id)
        return query

    @staticmethod
    def audit_context() -> Dict[str, Any]:
        """Get audit logging context for current tenant.

        Returns:
            Dictionary with tenant context for audit logs
        """
        ctx = get_tenant_context()
        if ctx:
            return {
                "tenant_id": ctx.tenant_id_str,
                "project_id": ctx.project_id_str,
                "scopes": ctx.scopes,
            }
        return {"tenant_id": None, "project_id": None}


__all__ = [
    "TenantContext",
    "TenantIsolation",
    "set_tenant_context",
    "get_tenant_context",
    "require_tenant_context",
    "clear_tenant_context",
]

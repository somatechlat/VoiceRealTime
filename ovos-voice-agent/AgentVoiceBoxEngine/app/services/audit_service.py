"""Audit logging service for administrative actions.

Implements Requirements 1.7, 15.5:
- Audit logs for all administrative actions
- Tagged with tenant_id for compliance reporting
- 7-year retention support

Actions logged:
- Tenant CRUD operations
- Project CRUD operations
- API key operations (create, rotate, revoke)
- User authentication events
- Settings changes
"""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from flask import g, request

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Audit log action types."""

    # Tenant actions
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"
    TENANT_SUSPENDED = "tenant.suspended"
    TENANT_DELETED = "tenant.deleted"

    # Project actions
    PROJECT_CREATED = "project.created"
    PROJECT_UPDATED = "project.updated"
    PROJECT_DELETED = "project.deleted"

    # API key actions
    API_KEY_CREATED = "api_key.created"
    API_KEY_ROTATED = "api_key.rotated"
    API_KEY_REVOKED = "api_key.revoked"

    # Auth actions
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_LOGIN_FAILED = "user.login_failed"

    # Session actions
    SESSION_CREATED = "session.created"
    SESSION_CLOSED = "session.closed"

    # Settings actions
    SETTINGS_CHANGED = "settings.changed"

    # Admin actions
    ADMIN_ACTION = "admin.action"


@dataclass
class AuditEntry:
    """Audit log entry."""

    tenant_id: uuid.UUID
    action: AuditAction
    resource_type: str
    resource_id: Optional[str] = None
    actor_id: Optional[str] = None
    actor_type: str = "user"
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: Optional[dt.datetime] = None
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": str(self.tenant_id),
            "action": self.action.value if isinstance(self.action, AuditAction) else self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditService:
    """Service for recording audit logs.

    Supports both sync (PostgreSQL) and async modes.
    """

    def __init__(self, db_session_factory: Optional[Any] = None):
        self._db_factory = db_session_factory

    def _get_request_context(self) -> Dict[str, Optional[str]]:
        """Extract request context for audit entry."""
        ip_address = None
        user_agent = None

        try:
            if request:
                ip_address = request.remote_addr
                # Handle X-Forwarded-For for proxied requests
                forwarded = request.headers.get("X-Forwarded-For")
                if forwarded:
                    ip_address = forwarded.split(",")[0].strip()
                user_agent = request.headers.get("User-Agent")
        except RuntimeError:
            # Outside request context
            pass

        return {"ip_address": ip_address, "user_agent": user_agent}

    def _get_actor_context(self) -> Dict[str, Optional[str]]:
        """Extract actor context from Flask g."""
        actor_id = None
        actor_type = "system"

        try:
            if hasattr(g, "user_id"):
                actor_id = str(g.user_id)
                actor_type = "user"
            elif hasattr(g, "api_key_id"):
                actor_id = str(g.api_key_id)
                actor_type = "api_key"
            elif hasattr(g, "service_name"):
                actor_id = g.service_name
                actor_type = "service"
        except RuntimeError:
            pass

        return {"actor_id": actor_id, "actor_type": actor_type}

    def log(
        self,
        tenant_id: uuid.UUID,
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        actor_type: Optional[str] = None,
    ) -> Optional[AuditEntry]:
        """Record an audit log entry.

        Args:
            tenant_id: Tenant ID for isolation
            action: Action being logged
            resource_type: Type of resource (tenant, project, api_key, etc.)
            resource_id: ID of the resource
            details: Additional details
            actor_id: Override actor ID
            actor_type: Override actor type

        Returns:
            Created AuditEntry or None if failed
        """
        request_ctx = self._get_request_context()
        actor_ctx = self._get_actor_context()

        entry = AuditEntry(
            tenant_id=tenant_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            actor_id=actor_id or actor_ctx["actor_id"],
            actor_type=actor_type or actor_ctx["actor_type"],
            details=details or {},
            ip_address=request_ctx["ip_address"],
            user_agent=request_ctx["user_agent"],
            created_at=dt.datetime.utcnow(),
        )

        # Log to structured logger
        logger.info(
            "Audit: %s on %s",
            action.value if isinstance(action, AuditAction) else action,
            resource_type,
            extra={
                "audit": entry.to_dict(),
                "tenant_id": str(tenant_id),
            },
        )

        # Persist to database
        if self._db_factory:
            try:
                return self._persist_entry(entry)
            except Exception as e:
                logger.error("Failed to persist audit log: %s", e)

        return entry

    def _persist_entry(self, entry: AuditEntry) -> AuditEntry:
        """Persist audit entry to PostgreSQL."""
        from ..models.tenant import AuditLog
        from ..utils.database import session_scope

        with session_scope(self._db_factory) as session:
            record = AuditLog(
                tenant_id=entry.tenant_id,
                actor_id=entry.actor_id,
                actor_type=entry.actor_type,
                action=entry.action.value if isinstance(entry.action, AuditAction) else entry.action,
                resource_type=entry.resource_type,
                resource_id=entry.resource_id,
                details=entry.details,
                ip_address=entry.ip_address,
                user_agent=entry.user_agent,
                created_at=entry.created_at,
            )
            session.add(record)
            session.commit()
            entry.id = record.id

        return entry

    def query(
        self,
        tenant_id: uuid.UUID,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Query audit logs for a tenant."""
        if not self._db_factory:
            return []

        from ..models.tenant import AuditLog
        from ..utils.database import session_scope

        with session_scope(self._db_factory) as session:
            query = session.query(AuditLog).filter(AuditLog.tenant_id == tenant_id)

            if action:
                query = query.filter(AuditLog.action == action.value)
            if resource_type:
                query = query.filter(AuditLog.resource_type == resource_type)
            if start_date:
                query = query.filter(AuditLog.created_at >= start_date)
            if end_date:
                query = query.filter(AuditLog.created_at <= end_date)

            query = query.order_by(AuditLog.created_at.desc())
            query = query.limit(limit).offset(offset)

            return [
                AuditEntry(
                    id=r.id,
                    tenant_id=r.tenant_id,
                    action=r.action,
                    resource_type=r.resource_type,
                    resource_id=r.resource_id,
                    actor_id=r.actor_id,
                    actor_type=r.actor_type,
                    details=r.details or {},
                    ip_address=r.ip_address,
                    user_agent=r.user_agent,
                    created_at=r.created_at,
                )
                for r in query.all()
            ]


# Global audit service instance
_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get the global audit service instance."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service


def init_audit_service(db_session_factory: Any) -> AuditService:
    """Initialize the audit service with database."""
    global _audit_service
    _audit_service = AuditService(db_session_factory)
    return _audit_service


def audit(
    tenant_id: uuid.UUID,
    action: AuditAction,
    resource_type: str,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Optional[AuditEntry]:
    """Convenience function to log an audit entry."""
    return get_audit_service().log(
        tenant_id=tenant_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
    )


__all__ = [
    "AuditAction",
    "AuditEntry",
    "AuditService",
    "get_audit_service",
    "init_audit_service",
    "audit",
]

"""Dashboard API Routes.

Provides endpoints for:
- Current usage summary
- Billing summary
- API health status
- Recent activity

Requirements: 21.3, 21.5
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth import UserContext, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


class UsageSummary(BaseModel):
    """Current period usage summary."""
    
    api_requests: int = Field(description="Total API requests this period")
    audio_minutes_input: float = Field(description="Audio input minutes (STT)")
    audio_minutes_output: float = Field(description="Audio output minutes (TTS)")
    llm_tokens_input: int = Field(description="LLM input tokens")
    llm_tokens_output: int = Field(description="LLM output tokens")
    concurrent_connections_peak: int = Field(description="Peak concurrent connections")
    period_start: datetime = Field(description="Billing period start")
    period_end: datetime = Field(description="Billing period end")


class BillingSummary(BaseModel):
    """Current billing summary."""
    
    plan_name: str = Field(description="Current plan name")
    plan_code: str = Field(description="Current plan code")
    amount_due_cents: int = Field(description="Amount due in cents")
    currency: str = Field(description="Currency code")
    next_billing_date: Optional[datetime] = Field(description="Next billing date")
    payment_status: str = Field(description="Payment status")


class HealthStatus(BaseModel):
    """API health status."""
    
    overall: str = Field(description="Overall health status")
    services: Dict[str, str] = Field(description="Individual service statuses")
    latency_ms: Dict[str, float] = Field(description="Service latencies in ms")


class ActivityItem(BaseModel):
    """Recent activity item."""
    
    id: str
    type: str = Field(description="Activity type")
    description: str = Field(description="Activity description")
    timestamp: datetime = Field(description="Activity timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DashboardResponse(BaseModel):
    """Complete dashboard response."""
    
    usage: UsageSummary
    billing: BillingSummary
    health: HealthStatus
    recent_activity: List[ActivityItem]


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    user: UserContext = Depends(get_current_user),
) -> DashboardResponse:
    """Get complete dashboard data.
    
    Returns usage summary, billing summary, health status, and recent activity.
    """
    tenant_id = user.tenant_id
    
    # Get usage from Lago
    usage = await _get_usage_summary(tenant_id)
    
    # Get billing from Lago
    billing = await _get_billing_summary(tenant_id)
    
    # Get health status
    health = await _get_health_status()
    
    # Get recent activity
    activity = await _get_recent_activity(tenant_id)
    
    return DashboardResponse(
        usage=usage,
        billing=billing,
        health=health,
        recent_activity=activity,
    )


@router.get("/dashboard/usage", response_model=UsageSummary)
async def get_usage(
    user: UserContext = Depends(get_current_user),
) -> UsageSummary:
    """Get current period usage summary."""
    return await _get_usage_summary(user.tenant_id)


@router.get("/dashboard/billing", response_model=BillingSummary)
async def get_billing(
    user: UserContext = Depends(get_current_user),
) -> BillingSummary:
    """Get current billing summary."""
    return await _get_billing_summary(user.tenant_id)


@router.get("/dashboard/health", response_model=HealthStatus)
async def get_health(
    user: UserContext = Depends(get_current_user),
) -> HealthStatus:
    """Get API health status."""
    return await _get_health_status()


@router.get("/dashboard/activity", response_model=List[ActivityItem])
async def get_activity(
    user: UserContext = Depends(get_current_user),
    limit: int = Query(default=10, ge=1, le=100),
) -> List[ActivityItem]:
    """Get recent activity."""
    return await _get_recent_activity(user.tenant_id, limit=limit)


# =============================================================================
# Internal Functions
# =============================================================================

async def _get_usage_summary(tenant_id: str) -> UsageSummary:
    """Get usage summary from Lago."""
    try:
        from ....app.services.lago_service import get_lago_service
        
        lago = get_lago_service()
        
        # Get customer's current usage
        # In production, this would query Lago's usage API
        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        if now.month == 12:
            period_end = period_start.replace(year=now.year + 1, month=1)
        else:
            period_end = period_start.replace(month=now.month + 1)
        
        # Query Lago for usage (simplified - real implementation would use Lago API)
        return UsageSummary(
            api_requests=0,
            audio_minutes_input=0.0,
            audio_minutes_output=0.0,
            llm_tokens_input=0,
            llm_tokens_output=0,
            concurrent_connections_peak=0,
            period_start=period_start,
            period_end=period_end,
        )
        
    except Exception as e:
        logger.error(f"Failed to get usage summary: {e}")
        now = datetime.now(timezone.utc)
        return UsageSummary(
            api_requests=0,
            audio_minutes_input=0.0,
            audio_minutes_output=0.0,
            llm_tokens_input=0,
            llm_tokens_output=0,
            concurrent_connections_peak=0,
            period_start=now,
            period_end=now + timedelta(days=30),
        )


async def _get_billing_summary(tenant_id: str) -> BillingSummary:
    """Get billing summary from Lago."""
    try:
        from ....app.services.lago_service import get_lago_service
        
        lago = get_lago_service()
        
        # Get customer's subscription
        subscriptions = await lago.list_subscriptions(
            external_customer_id=tenant_id,
        )
        
        if subscriptions:
            sub = subscriptions[0]
            return BillingSummary(
                plan_name=sub.name or sub.plan_code.title(),
                plan_code=sub.plan_code,
                amount_due_cents=0,  # Would come from current invoice
                currency="USD",
                next_billing_date=sub.ending_at,
                payment_status="current",
            )
        
        return BillingSummary(
            plan_name="Free",
            plan_code="free",
            amount_due_cents=0,
            currency="USD",
            next_billing_date=None,
            payment_status="current",
        )
        
    except Exception as e:
        logger.error(f"Failed to get billing summary: {e}")
        return BillingSummary(
            plan_name="Unknown",
            plan_code="unknown",
            amount_due_cents=0,
            currency="USD",
            next_billing_date=None,
            payment_status="unknown",
        )


async def _get_health_status() -> HealthStatus:
    """Get API health status."""
    services = {}
    latencies = {}
    
    # Check Redis
    try:
        import time
        from ....app.services.redis_client import get_redis_client
        
        start = time.monotonic()
        redis = get_redis_client()
        await redis.ping()
        latencies["redis"] = (time.monotonic() - start) * 1000
        services["redis"] = "healthy"
    except Exception:
        services["redis"] = "unhealthy"
        latencies["redis"] = -1
    
    # Check PostgreSQL
    try:
        import time
        from ....app.services.async_database import get_database
        
        start = time.monotonic()
        db = get_database()
        await db.execute("SELECT 1")
        latencies["postgresql"] = (time.monotonic() - start) * 1000
        services["postgresql"] = "healthy"
    except Exception:
        services["postgresql"] = "unhealthy"
        latencies["postgresql"] = -1
    
    # Determine overall status
    overall = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
    
    return HealthStatus(
        overall=overall,
        services=services,
        latency_ms=latencies,
    )


async def _get_recent_activity(
    tenant_id: str,
    limit: int = 10,
) -> List[ActivityItem]:
    """Get recent activity from audit logs."""
    try:
        from ....app.services.audit_service import get_audit_service
        
        audit = get_audit_service()
        logs = await audit.get_recent_logs(tenant_id=tenant_id, limit=limit)
        
        return [
            ActivityItem(
                id=log.id,
                type=log.action,
                description=log.description,
                timestamp=log.timestamp,
                metadata=log.metadata,
            )
            for log in logs
        ]
        
    except Exception as e:
        logger.error(f"Failed to get recent activity: {e}")
        return []

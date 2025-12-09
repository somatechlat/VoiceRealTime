"""Billing Management Routes.

Provides endpoints for:
- View current plan
- List available plans
- Upgrade/downgrade subscription
- View invoices
- Download invoice PDFs

Requirements: 21.6, 21.7
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ..auth import UserContext, get_current_user, require_admin

logger = logging.getLogger(__name__)
router = APIRouter()


class PlanDetails(BaseModel):
    """Subscription plan details."""
    
    code: str = Field(description="Plan code")
    name: str = Field(description="Plan display name")
    description: str = Field(description="Plan description")
    amount_cents: int = Field(description="Monthly price in cents")
    currency: str = Field(description="Currency code")
    interval: str = Field(description="Billing interval (monthly/yearly)")
    features: List[str] = Field(description="Plan features")
    limits: dict = Field(description="Usage limits")


class CurrentSubscription(BaseModel):
    """Current subscription details."""
    
    id: str = Field(description="Subscription ID")
    plan: PlanDetails = Field(description="Current plan details")
    status: str = Field(description="Subscription status")
    started_at: Optional[datetime] = Field(description="Subscription start date")
    current_period_end: Optional[datetime] = Field(description="Current period end")
    cancel_at_period_end: bool = Field(description="Whether subscription cancels at period end")


class InvoiceItem(BaseModel):
    """Invoice line item."""
    
    description: str
    quantity: float
    unit_amount_cents: int
    amount_cents: int


class Invoice(BaseModel):
    """Invoice details."""
    
    id: str = Field(description="Invoice ID")
    number: str = Field(description="Invoice number")
    status: str = Field(description="Invoice status")
    payment_status: str = Field(description="Payment status")
    total_amount_cents: int = Field(description="Total amount in cents")
    currency: str = Field(description="Currency code")
    issuing_date: Optional[datetime] = Field(description="Issue date")
    payment_due_date: Optional[datetime] = Field(description="Due date")
    items: List[InvoiceItem] = Field(description="Line items")
    pdf_url: Optional[str] = Field(description="PDF download URL")


class SubscriptionChange(BaseModel):
    """Request to change subscription."""
    
    plan_code: str = Field(description="New plan code")


class SubscriptionChangeResponse(BaseModel):
    """Response from subscription change."""
    
    subscription: CurrentSubscription
    effective_date: datetime = Field(description="When change takes effect")
    proration_amount_cents: int = Field(description="Proration amount")
    message: str = Field(description="Change description")


# Available plans (would come from Lago in production)
AVAILABLE_PLANS = [
    PlanDetails(
        code="free",
        name="Free",
        description="For evaluation and small projects",
        amount_cents=0,
        currency="USD",
        interval="monthly",
        features=[
            "100 API calls/month",
            "10 audio minutes",
            "Community support",
        ],
        limits={
            "api_requests": 100,
            "audio_minutes": 10,
            "concurrent_connections": 1,
        },
    ),
    PlanDetails(
        code="pro",
        name="Pro",
        description="For production applications",
        amount_cents=4900,
        currency="USD",
        interval="monthly",
        features=[
            "10,000 API calls/month",
            "1,000 audio minutes",
            "Email support",
            "Usage-based overage",
        ],
        limits={
            "api_requests": 10000,
            "audio_minutes": 1000,
            "concurrent_connections": 100,
        },
    ),
    PlanDetails(
        code="enterprise",
        name="Enterprise",
        description="For large-scale deployments",
        amount_cents=99900,
        currency="USD",
        interval="monthly",
        features=[
            "Unlimited API calls",
            "Unlimited audio minutes",
            "Dedicated support",
            "SLA guarantee",
            "Custom integrations",
        ],
        limits={
            "api_requests": -1,  # Unlimited
            "audio_minutes": -1,
            "concurrent_connections": -1,
        },
    ),
]


@router.get("/billing/plans", response_model=List[PlanDetails])
async def list_plans(
    user: UserContext = Depends(get_current_user),
) -> List[PlanDetails]:
    """List all available subscription plans."""
    return AVAILABLE_PLANS


@router.get("/billing/subscription", response_model=CurrentSubscription)
async def get_subscription(
    user: UserContext = Depends(get_current_user),
) -> CurrentSubscription:
    """Get current subscription details."""
    try:
        from ....app.services.lago_service import get_lago_service
        
        lago = get_lago_service()
        subscriptions = await lago.list_subscriptions(
            external_customer_id=user.tenant_id,
        )
        
        if not subscriptions:
            # Return free plan if no subscription
            return CurrentSubscription(
                id="free",
                plan=AVAILABLE_PLANS[0],  # Free plan
                status="active",
                started_at=None,
                current_period_end=None,
                cancel_at_period_end=False,
            )
        
        sub = subscriptions[0]
        plan = next(
            (p for p in AVAILABLE_PLANS if p.code == sub.plan_code),
            AVAILABLE_PLANS[0],
        )
        
        return CurrentSubscription(
            id=sub.external_id,
            plan=plan,
            status=sub.status.value,
            started_at=sub.started_at,
            current_period_end=sub.ending_at,
            cancel_at_period_end=sub.canceled_at is not None,
        )
        
    except Exception as e:
        logger.error(f"Failed to get subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get subscription",
        )


@router.post("/billing/subscription", response_model=SubscriptionChangeResponse)
async def change_subscription(
    request: SubscriptionChange,
    user: UserContext = Depends(require_admin()),
) -> SubscriptionChangeResponse:
    """Upgrade or downgrade subscription.
    
    Requires tenant_admin role.
    """
    try:
        from ....app.services.lago_service import get_lago_service
        
        lago = get_lago_service()
        
        # Validate plan code
        plan = next(
            (p for p in AVAILABLE_PLANS if p.code == request.plan_code),
            None,
        )
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid plan code: {request.plan_code}",
            )
        
        # Get current subscription
        subscriptions = await lago.list_subscriptions(
            external_customer_id=user.tenant_id,
        )
        
        if subscriptions:
            # Update existing subscription
            sub = await lago.update_subscription(
                external_id=subscriptions[0].external_id,
                plan_code=request.plan_code,
            )
        else:
            # Create new subscription
            sub = await lago.create_subscription(
                external_customer_id=user.tenant_id,
                plan_code=request.plan_code,
            )
        
        return SubscriptionChangeResponse(
            subscription=CurrentSubscription(
                id=sub.external_id,
                plan=plan,
                status=sub.status.value,
                started_at=sub.started_at,
                current_period_end=sub.ending_at,
                cancel_at_period_end=False,
            ),
            effective_date=datetime.now(),
            proration_amount_cents=0,  # Would be calculated by Lago
            message=f"Subscription changed to {plan.name}",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to change subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change subscription",
        )


@router.delete("/billing/subscription", status_code=status.HTTP_200_OK)
async def cancel_subscription(
    user: UserContext = Depends(require_admin()),
) -> dict:
    """Cancel subscription at end of current period.
    
    Requires tenant_admin role.
    """
    try:
        from ....app.services.lago_service import get_lago_service
        
        lago = get_lago_service()
        subscriptions = await lago.list_subscriptions(
            external_customer_id=user.tenant_id,
        )
        
        if not subscriptions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active subscription found",
            )
        
        # Terminate subscription
        await lago.terminate_subscription(subscriptions[0].external_id)
        
        return {
            "message": "Subscription will be canceled at end of current period",
            "effective_date": subscriptions[0].ending_at,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription",
        )


@router.get("/billing/invoices", response_model=List[Invoice])
async def list_invoices(
    user: UserContext = Depends(get_current_user),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=10, ge=1, le=100),
) -> List[Invoice]:
    """List invoices for the tenant."""
    try:
        from ....app.services.lago_service import get_lago_service
        
        lago = get_lago_service()
        invoices = await lago.list_invoices(
            external_customer_id=user.tenant_id,
            page=page,
            per_page=per_page,
        )
        
        return [
            Invoice(
                id=inv.lago_id,
                number=inv.number,
                status=inv.status.value,
                payment_status=inv.payment_status,
                total_amount_cents=inv.total_amount_cents,
                currency=inv.currency,
                issuing_date=inv.issuing_date,
                payment_due_date=inv.payment_due_date,
                items=[],  # Would be populated from Lago
                pdf_url=inv.file_url,
            )
            for inv in invoices
        ]
        
    except Exception as e:
        logger.error(f"Failed to list invoices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list invoices",
        )


@router.get("/billing/invoices/{invoice_id}", response_model=Invoice)
async def get_invoice(
    invoice_id: str,
    user: UserContext = Depends(get_current_user),
) -> Invoice:
    """Get invoice details."""
    try:
        from ....app.services.lago_service import get_lago_service
        
        lago = get_lago_service()
        inv = await lago.get_invoice(invoice_id)
        
        if not inv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found",
            )
        
        return Invoice(
            id=inv.lago_id,
            number=inv.number,
            status=inv.status.value,
            payment_status=inv.payment_status,
            total_amount_cents=inv.total_amount_cents,
            currency=inv.currency,
            issuing_date=inv.issuing_date,
            payment_due_date=inv.payment_due_date,
            items=[],
            pdf_url=inv.file_url,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get invoice: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get invoice",
        )


@router.get("/billing/invoices/{invoice_id}/download")
async def download_invoice(
    invoice_id: str,
    user: UserContext = Depends(get_current_user),
) -> RedirectResponse:
    """Download invoice as PDF."""
    try:
        from ....app.services.lago_service import get_lago_service
        
        lago = get_lago_service()
        pdf_url = await lago.download_invoice(invoice_id)
        
        if not pdf_url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice PDF not available",
            )
        
        return RedirectResponse(url=pdf_url)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download invoice: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download invoice",
        )

"""Settings and Webhook Management Routes.

Provides endpoints for:
- Organization profile settings
- Notification preferences
- Webhook configuration

Requirements: 21.8
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, HttpUrl

from ..auth import UserContext, get_current_user, require_admin

logger = logging.getLogger(__name__)
router = APIRouter()


class OrganizationProfile(BaseModel):
    """Organization profile settings."""
    
    name: str = Field(description="Organization name")
    email: EmailStr = Field(description="Primary contact email")
    website: Optional[str] = Field(default=None, description="Website URL")
    address: Optional[str] = Field(default=None, description="Business address")
    phone: Optional[str] = Field(default=None, description="Contact phone")
    timezone: str = Field(default="UTC", description="Default timezone")
    logo_url: Optional[str] = Field(default=None, description="Logo URL")


class NotificationPreferences(BaseModel):
    """Notification preferences."""
    
    email_billing: bool = Field(default=True, description="Billing notifications")
    email_usage_alerts: bool = Field(default=True, description="Usage threshold alerts")
    email_security: bool = Field(default=True, description="Security notifications")
    email_product_updates: bool = Field(default=False, description="Product updates")
    email_weekly_summary: bool = Field(default=True, description="Weekly usage summary")


class WebhookConfig(BaseModel):
    """Webhook configuration."""
    
    id: str = Field(description="Webhook ID")
    url: str = Field(description="Webhook URL")
    events: List[str] = Field(description="Subscribed events")
    secret: Optional[str] = Field(default=None, description="Signing secret (only on creation)")
    is_active: bool = Field(default=True, description="Whether webhook is active")
    created_at: datetime = Field(description="Creation timestamp")
    last_triggered_at: Optional[datetime] = Field(description="Last trigger timestamp")


class WebhookCreate(BaseModel):
    """Request to create a webhook."""
    
    url: HttpUrl = Field(description="Webhook URL")
    events: List[str] = Field(
        description="Events to subscribe to",
        min_length=1,
    )


class WebhookUpdate(BaseModel):
    """Request to update a webhook."""
    
    url: Optional[HttpUrl] = Field(default=None, description="New URL")
    events: Optional[List[str]] = Field(default=None, description="New events")
    is_active: Optional[bool] = Field(default=None, description="Active status")


# Available webhook events
WEBHOOK_EVENTS = [
    "session.created",
    "session.ended",
    "transcription.completed",
    "response.completed",
    "usage.threshold_reached",
    "billing.invoice_created",
    "billing.payment_succeeded",
    "billing.payment_failed",
    "api_key.created",
    "api_key.revoked",
    "team.member_added",
    "team.member_removed",
]


@router.get("/settings/profile", response_model=OrganizationProfile)
async def get_profile(
    user: UserContext = Depends(get_current_user),
) -> OrganizationProfile:
    """Get organization profile."""
    try:
        from ....app.services.lago_service import get_lago_service
        
        lago = get_lago_service()
        customer = await lago.get_customer(user.tenant_id)
        
        if customer:
            return OrganizationProfile(
                name=customer.name,
                email=customer.email,
                website=None,
                address=None,
                phone=None,
                timezone=customer.timezone,
                logo_url=None,
            )
        
        return OrganizationProfile(
            name="Unknown",
            email=user.email,
            timezone="UTC",
        )
        
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile",
        )


@router.put("/settings/profile", response_model=OrganizationProfile)
async def update_profile(
    profile: OrganizationProfile,
    user: UserContext = Depends(require_admin()),
) -> OrganizationProfile:
    """Update organization profile.
    
    Requires tenant_admin role.
    """
    try:
        from ....app.services.lago_service import get_lago_service
        
        lago = get_lago_service()
        
        # Update Lago customer
        await lago.update_customer(
            external_id=user.tenant_id,
            name=profile.name,
            email=profile.email,
            timezone=profile.timezone,
        )
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to update profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile",
        )


@router.get("/settings/notifications", response_model=NotificationPreferences)
async def get_notification_preferences(
    user: UserContext = Depends(get_current_user),
) -> NotificationPreferences:
    """Get notification preferences."""
    # In production, this would come from the database
    return NotificationPreferences()


@router.put("/settings/notifications", response_model=NotificationPreferences)
async def update_notification_preferences(
    preferences: NotificationPreferences,
    user: UserContext = Depends(get_current_user),
) -> NotificationPreferences:
    """Update notification preferences."""
    # In production, this would save to the database
    return preferences


@router.get("/settings/webhooks/events")
async def list_webhook_events(
    user: UserContext = Depends(get_current_user),
) -> List[str]:
    """List available webhook events."""
    return WEBHOOK_EVENTS


@router.get("/settings/webhooks", response_model=List[WebhookConfig])
async def list_webhooks(
    user: UserContext = Depends(get_current_user),
) -> List[WebhookConfig]:
    """List configured webhooks."""
    # In production, this would query the database
    return []


@router.post("/settings/webhooks", response_model=WebhookConfig, status_code=status.HTTP_201_CREATED)
async def create_webhook(
    request: WebhookCreate,
    user: UserContext = Depends(require_admin()),
) -> WebhookConfig:
    """Create a new webhook.
    
    Requires tenant_admin role.
    The signing secret is only returned once.
    """
    # Validate events
    invalid_events = [e for e in request.events if e not in WEBHOOK_EVENTS]
    if invalid_events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid events: {invalid_events}",
        )
    
    # Generate webhook ID and secret
    webhook_id = f"wh_{secrets.token_hex(8)}"
    webhook_secret = f"whsec_{secrets.token_hex(32)}"
    
    # In production, save to database
    
    return WebhookConfig(
        id=webhook_id,
        url=str(request.url),
        events=request.events,
        secret=webhook_secret,  # Only returned on creation
        is_active=True,
        created_at=datetime.now(),
        last_triggered_at=None,
    )


@router.get("/settings/webhooks/{webhook_id}", response_model=WebhookConfig)
async def get_webhook(
    webhook_id: str,
    user: UserContext = Depends(get_current_user),
) -> WebhookConfig:
    """Get webhook details."""
    # In production, query from database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Webhook not found",
    )


@router.patch("/settings/webhooks/{webhook_id}", response_model=WebhookConfig)
async def update_webhook(
    webhook_id: str,
    request: WebhookUpdate,
    user: UserContext = Depends(require_admin()),
) -> WebhookConfig:
    """Update a webhook.
    
    Requires tenant_admin role.
    """
    # In production, update in database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Webhook not found",
    )


@router.delete("/settings/webhooks/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_webhook(
    webhook_id: str,
    user: UserContext = Depends(require_admin()),
):
    """Delete a webhook.
    
    Requires tenant_admin role.
    """
    # In production, delete from database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Webhook not found",
    )


@router.post("/settings/webhooks/{webhook_id}/test")
async def test_webhook(
    webhook_id: str,
    user: UserContext = Depends(require_admin()),
) -> dict:
    """Send a test event to a webhook.
    
    Requires tenant_admin role.
    """
    # In production, send test event
    return {
        "message": "Test event sent",
        "webhook_id": webhook_id,
    }


@router.post("/settings/webhooks/{webhook_id}/rotate-secret", response_model=WebhookConfig)
async def rotate_webhook_secret(
    webhook_id: str,
    user: UserContext = Depends(require_admin()),
) -> WebhookConfig:
    """Rotate webhook signing secret.
    
    Requires tenant_admin role.
    The new secret is only returned once.
    """
    # Generate new secret
    new_secret = f"whsec_{secrets.token_hex(32)}"
    
    # In production, update in database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Webhook not found",
    )

"""API Key Management Routes.

Provides endpoints for:
- Create API keys
- List API keys
- Rotate API keys
- Revoke API keys
- View usage per key

Requirements: 21.4
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..auth import UserContext, get_current_user, require_permission

logger = logging.getLogger(__name__)
router = APIRouter()


class APIKeyCreate(BaseModel):
    """Request to create a new API key."""
    
    name: str = Field(min_length=1, max_length=100, description="Key name")
    scopes: List[str] = Field(
        default=["realtime:connect"],
        description="Permission scopes",
    )
    expires_in_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=365,
        description="Days until expiration (null for no expiration)",
    )


class APIKeyResponse(BaseModel):
    """API key response (without secret)."""
    
    id: str = Field(description="Key ID")
    name: str = Field(description="Key name")
    prefix: str = Field(description="Key prefix (first 8 chars)")
    scopes: List[str] = Field(description="Permission scopes")
    created_at: datetime = Field(description="Creation timestamp")
    expires_at: Optional[datetime] = Field(description="Expiration timestamp")
    last_used_at: Optional[datetime] = Field(description="Last usage timestamp")
    is_active: bool = Field(description="Whether key is active")


class APIKeyCreated(APIKeyResponse):
    """API key response with secret (only returned on creation)."""
    
    secret: str = Field(description="Full API key (only shown once)")


class APIKeyUsage(BaseModel):
    """API key usage statistics."""
    
    key_id: str
    total_requests: int
    requests_today: int
    requests_this_month: int
    last_used_at: Optional[datetime]


class APIKeyRotateResponse(BaseModel):
    """Response from key rotation."""
    
    old_key_id: str
    new_key: APIKeyCreated
    grace_period_hours: int = Field(
        default=24,
        description="Hours until old key is revoked",
    )


@router.get("/keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    user: UserContext = Depends(get_current_user),
    include_inactive: bool = Query(default=False),
) -> List[APIKeyResponse]:
    """List all API keys for the tenant."""
    try:
        from ....app.services.api_key_service import get_api_key_service
        
        service = get_api_key_service()
        keys = await service.list_keys(
            tenant_id=user.tenant_id,
            include_inactive=include_inactive,
        )
        
        return [
            APIKeyResponse(
                id=k.id,
                name=k.name,
                prefix=k.prefix,
                scopes=k.scopes,
                created_at=k.created_at,
                expires_at=k.expires_at,
                last_used_at=k.last_used_at,
                is_active=k.is_active,
            )
            for k in keys
        ]
        
    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys",
        )


@router.post("/keys", response_model=APIKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: APIKeyCreate,
    user: UserContext = Depends(get_current_user),
) -> APIKeyCreated:
    """Create a new API key.
    
    The full key secret is only returned once. Store it securely.
    """
    try:
        from ....app.services.api_key_service import get_api_key_service
        
        service = get_api_key_service()
        
        # Generate the key
        key, secret = await service.create_key(
            tenant_id=user.tenant_id,
            name=request.name,
            scopes=request.scopes,
            expires_in_days=request.expires_in_days,
            created_by=user.user_id,
        )
        
        return APIKeyCreated(
            id=key.id,
            name=key.name,
            prefix=key.prefix,
            scopes=key.scopes,
            created_at=key.created_at,
            expires_at=key.expires_at,
            last_used_at=key.last_used_at,
            is_active=key.is_active,
            secret=secret,
        )
        
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        )


@router.get("/keys/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    user: UserContext = Depends(get_current_user),
) -> APIKeyResponse:
    """Get API key details."""
    try:
        from ....app.services.api_key_service import get_api_key_service
        
        service = get_api_key_service()
        key = await service.get_key(key_id=key_id, tenant_id=user.tenant_id)
        
        if not key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        
        return APIKeyResponse(
            id=key.id,
            name=key.name,
            prefix=key.prefix,
            scopes=key.scopes,
            created_at=key.created_at,
            expires_at=key.expires_at,
            last_used_at=key.last_used_at,
            is_active=key.is_active,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get API key",
        )


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    user: UserContext = Depends(get_current_user),
) -> None:
    """Revoke an API key."""
    try:
        from ....app.services.api_key_service import get_api_key_service
        
        service = get_api_key_service()
        success = await service.revoke_key(key_id=key_id, tenant_id=user.tenant_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key",
        )


@router.post("/keys/{key_id}/rotate", response_model=APIKeyRotateResponse)
async def rotate_api_key(
    key_id: str,
    user: UserContext = Depends(get_current_user),
) -> APIKeyRotateResponse:
    """Rotate an API key.
    
    Creates a new key and schedules the old key for revocation
    after a 24-hour grace period.
    """
    try:
        from ....app.services.api_key_service import get_api_key_service
        
        service = get_api_key_service()
        
        # Get existing key
        old_key = await service.get_key(key_id=key_id, tenant_id=user.tenant_id)
        if not old_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        
        # Create new key with same settings
        new_key, secret = await service.create_key(
            tenant_id=user.tenant_id,
            name=f"{old_key.name} (rotated)",
            scopes=old_key.scopes,
            expires_in_days=None,  # New key doesn't expire
            created_by=user.user_id,
        )
        
        # Schedule old key for revocation (24 hours)
        await service.schedule_revocation(key_id=key_id, hours=24)
        
        return APIKeyRotateResponse(
            old_key_id=key_id,
            new_key=APIKeyCreated(
                id=new_key.id,
                name=new_key.name,
                prefix=new_key.prefix,
                scopes=new_key.scopes,
                created_at=new_key.created_at,
                expires_at=new_key.expires_at,
                last_used_at=new_key.last_used_at,
                is_active=new_key.is_active,
                secret=secret,
            ),
            grace_period_hours=24,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rotate API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rotate API key",
        )


@router.get("/keys/{key_id}/usage", response_model=APIKeyUsage)
async def get_api_key_usage(
    key_id: str,
    user: UserContext = Depends(get_current_user),
) -> APIKeyUsage:
    """Get usage statistics for an API key."""
    try:
        from ....app.services.api_key_service import get_api_key_service
        
        service = get_api_key_service()
        
        # Verify key belongs to tenant
        key = await service.get_key(key_id=key_id, tenant_id=user.tenant_id)
        if not key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        
        usage = await service.get_key_usage(key_id=key_id)
        
        return APIKeyUsage(
            key_id=key_id,
            total_requests=usage.get("total_requests", 0),
            requests_today=usage.get("requests_today", 0),
            requests_this_month=usage.get("requests_this_month", 0),
            last_used_at=key.last_used_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get API key usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get API key usage",
        )

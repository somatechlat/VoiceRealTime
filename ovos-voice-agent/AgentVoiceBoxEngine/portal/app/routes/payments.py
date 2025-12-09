"""Payment Method Management Routes.

Provides endpoints for:
- List payment methods
- Add payment method
- Remove payment method
- Set default payment method

Requirements: 22.3, 22.4, 22.6
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth import UserContext, get_current_user, require_admin

logger = logging.getLogger(__name__)
router = APIRouter()


class PaymentMethodResponse(BaseModel):
    """Payment method details."""
    
    id: str = Field(description="Payment method ID")
    type: str = Field(description="Payment method type (card, bank_account, paypal)")
    provider: str = Field(description="Payment provider (stripe, paypal)")
    last_four: Optional[str] = Field(description="Last 4 digits")
    brand: Optional[str] = Field(description="Card brand")
    exp_month: Optional[int] = Field(description="Expiration month")
    exp_year: Optional[int] = Field(description="Expiration year")
    is_default: bool = Field(description="Whether this is the default method")
    created_at: Optional[datetime] = Field(description="Creation timestamp")


class AddPaymentMethodRequest(BaseModel):
    """Request to add a payment method."""
    
    provider: str = Field(description="Payment provider (stripe, paypal)")
    payment_method_id: str = Field(description="Provider-specific payment method ID")
    set_default: bool = Field(default=True, description="Set as default method")


class SetDefaultRequest(BaseModel):
    """Request to set default payment method."""
    
    payment_method_id: str = Field(description="Payment method ID to set as default")


@router.get("/payments/methods", response_model=List[PaymentMethodResponse])
async def list_payment_methods(
    user: UserContext = Depends(get_current_user),
) -> List[PaymentMethodResponse]:
    """List all payment methods for the tenant."""
    try:
        from ....app.services.payment_service import get_payment_service, PaymentProvider
        
        service = get_payment_service()
        methods = []
        
        # Get methods from Stripe
        try:
            stripe_methods = await service.list_payment_methods(
                customer_id=user.tenant_id,
                provider=PaymentProvider.STRIPE,
            )
            methods.extend([
                PaymentMethodResponse(
                    id=m.id,
                    type=m.type,
                    provider="stripe",
                    last_four=m.last_four,
                    brand=m.brand,
                    exp_month=m.exp_month,
                    exp_year=m.exp_year,
                    is_default=m.is_default,
                    created_at=m.created_at,
                )
                for m in stripe_methods
            ])
        except Exception as e:
            logger.warning(f"Failed to get Stripe methods: {e}")
        
        # Get methods from PayPal
        try:
            paypal_methods = await service.list_payment_methods(
                customer_id=user.tenant_id,
                provider=PaymentProvider.PAYPAL,
            )
            methods.extend([
                PaymentMethodResponse(
                    id=m.id,
                    type=m.type,
                    provider="paypal",
                    last_four=m.last_four,
                    brand=m.brand,
                    exp_month=m.exp_month,
                    exp_year=m.exp_year,
                    is_default=m.is_default,
                    created_at=m.created_at,
                )
                for m in paypal_methods
            ])
        except Exception as e:
            logger.warning(f"Failed to get PayPal methods: {e}")
        
        return methods
        
    except Exception as e:
        logger.error(f"Failed to list payment methods: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list payment methods",
        )


@router.post("/payments/methods", response_model=PaymentMethodResponse, status_code=status.HTTP_201_CREATED)
async def add_payment_method(
    request: AddPaymentMethodRequest,
    user: UserContext = Depends(require_admin()),
) -> PaymentMethodResponse:
    """Add a new payment method.
    
    Requires tenant_admin role.
    The payment_method_id should be obtained from Stripe Elements or PayPal SDK.
    """
    try:
        from ....app.services.payment_service import get_payment_service, PaymentProvider
        
        service = get_payment_service()
        
        # Determine provider
        if request.provider.lower() == "stripe":
            provider = PaymentProvider.STRIPE
        elif request.provider.lower() == "paypal":
            provider = PaymentProvider.PAYPAL
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider: {request.provider}",
            )
        
        # Attach payment method
        method = await service.get_provider(provider).attach_payment_method(
            customer_id=user.tenant_id,
            payment_method_id=request.payment_method_id,
            set_default=request.set_default,
        )
        
        return PaymentMethodResponse(
            id=method.id,
            type=method.type,
            provider=request.provider.lower(),
            last_four=method.last_four,
            brand=method.brand,
            exp_month=method.exp_month,
            exp_year=method.exp_year,
            is_default=method.is_default,
            created_at=method.created_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add payment method: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add payment method",
        )


@router.delete("/payments/methods/{method_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def remove_payment_method(
    method_id: str,
    provider: str,
    user: UserContext = Depends(require_admin()),
):
    """Remove a payment method.
    
    Requires tenant_admin role.
    """
    try:
        from ....app.services.payment_service import get_payment_service, PaymentProvider
        
        service = get_payment_service()
        
        # Determine provider
        if provider.lower() == "stripe":
            prov = PaymentProvider.STRIPE
        elif provider.lower() == "paypal":
            prov = PaymentProvider.PAYPAL
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider: {provider}",
            )
        
        await service.get_provider(prov).detach_payment_method(method_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove payment method: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove payment method",
        )


@router.post("/payments/methods/default", response_model=PaymentMethodResponse)
async def set_default_payment_method(
    request: SetDefaultRequest,
    user: UserContext = Depends(require_admin()),
) -> PaymentMethodResponse:
    """Set the default payment method.
    
    Requires tenant_admin role.
    """
    try:
        from ....app.services.payment_service import get_payment_service, PaymentProvider
        
        service = get_payment_service()
        
        # Try to find and update the payment method
        # In production, we'd need to track which provider owns which method
        method = await service.get_provider(PaymentProvider.STRIPE).attach_payment_method(
            customer_id=user.tenant_id,
            payment_method_id=request.payment_method_id,
            set_default=True,
        )
        
        return PaymentMethodResponse(
            id=method.id,
            type=method.type,
            provider="stripe",
            last_four=method.last_four,
            brand=method.brand,
            exp_month=method.exp_month,
            exp_year=method.exp_year,
            is_default=True,
            created_at=method.created_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set default payment method: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set default payment method",
        )


@router.get("/payments/history")
async def get_payment_history(
    user: UserContext = Depends(get_current_user),
) -> List[dict]:
    """Get payment history."""
    # This would query payment records from the database
    return []

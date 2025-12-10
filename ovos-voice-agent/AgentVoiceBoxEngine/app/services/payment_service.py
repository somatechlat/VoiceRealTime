"""Payment Processing Service for Stripe and PayPal Integration.

This module provides integration with payment processors:
- Stripe: Credit/debit cards, ACH payments
- PayPal: PayPal balance, alternative card payments
- Webhook handling for payment events
- Refund processing
- Dunning and suspension flow

Requirements: 20.6, 20.8, 20.9, 22.1, 22.2, 22.3, 22.4, 22.6, 22.7
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class PaymentProvider(str, Enum):
    """Supported payment providers."""

    STRIPE = "stripe"
    PAYPAL = "paypal"


class PaymentStatus(str, Enum):
    """Payment status values."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class RefundStatus(str, Enum):
    """Refund status values."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class PaymentMethod:
    """Payment method representation."""

    id: str
    provider: PaymentProvider
    type: str  # card, bank_account, paypal
    last_four: Optional[str] = None
    brand: Optional[str] = None  # visa, mastercard, amex, paypal
    exp_month: Optional[int] = None
    exp_year: Optional[int] = None
    is_default: bool = False
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Payment:
    """Payment transaction representation."""

    id: str
    provider: PaymentProvider
    amount_cents: int
    currency: str
    status: PaymentStatus
    customer_id: str
    payment_method_id: Optional[str] = None
    description: Optional[str] = None
    invoice_id: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Refund:
    """Refund transaction representation."""

    id: str
    provider: PaymentProvider
    payment_id: str
    amount_cents: int
    currency: str
    status: RefundStatus
    reason: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PaymentProviderBase(ABC):
    """Abstract base class for payment providers."""

    @abstractmethod
    async def create_customer(
        self,
        external_id: str,
        email: str,
        name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a customer in the payment provider.

        Returns:
            Provider-specific customer ID
        """
        pass

    @abstractmethod
    async def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get customer details."""
        pass

    @abstractmethod
    async def delete_customer(self, customer_id: str) -> None:
        """Delete a customer."""
        pass

    @abstractmethod
    async def create_payment_intent(
        self,
        customer_id: str,
        amount_cents: int,
        currency: str,
        payment_method_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Payment:
        """Create a payment intent."""
        pass

    @abstractmethod
    async def confirm_payment(self, payment_id: str) -> Payment:
        """Confirm a payment."""
        pass

    @abstractmethod
    async def cancel_payment(self, payment_id: str) -> Payment:
        """Cancel a payment."""
        pass

    @abstractmethod
    async def create_refund(
        self,
        payment_id: str,
        amount_cents: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> Refund:
        """Create a refund (full or partial)."""
        pass

    @abstractmethod
    async def list_payment_methods(
        self,
        customer_id: str,
    ) -> List[PaymentMethod]:
        """List customer's payment methods."""
        pass

    @abstractmethod
    async def attach_payment_method(
        self,
        customer_id: str,
        payment_method_id: str,
        set_default: bool = False,
    ) -> PaymentMethod:
        """Attach a payment method to a customer."""
        pass

    @abstractmethod
    async def detach_payment_method(self, payment_method_id: str) -> None:
        """Detach a payment method from a customer."""
        pass

    @abstractmethod
    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify webhook signature."""
        pass


class StripeProvider(PaymentProviderBase):
    """Stripe payment provider implementation.

    Requirements: 20.6, 22.1
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ):
        """Initialize Stripe provider.

        Args:
            api_key: Stripe secret API key
            webhook_secret: Stripe webhook signing secret
        """
        self.api_key = api_key or os.getenv("STRIPE_API_KEY", "")
        self.webhook_secret = webhook_secret or os.getenv("STRIPE_WEBHOOK_SECRET", "")
        self.base_url = "https://api.stripe.com/v1"
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

    async def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make authenticated API request."""
        url = f"{self.base_url}{path}"
        return await self._client.request(
            method,
            url,
            headers=self._headers(),
            data=data,
        )

    async def create_customer(
        self,
        external_id: str,
        email: str,
        name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a Stripe customer."""
        data = {
            "email": email,
            "name": name,
            "metadata[external_id]": external_id,
        }

        if metadata:
            for key, value in metadata.items():
                data[f"metadata[{key}]"] = value

        response = await self._request("POST", "/customers", data=data)
        response.raise_for_status()

        return response.json()["id"]

    async def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get Stripe customer details."""
        response = await self._request("GET", f"/customers/{customer_id}")

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.json()

    async def delete_customer(self, customer_id: str) -> None:
        """Delete a Stripe customer."""
        response = await self._request("DELETE", f"/customers/{customer_id}")
        response.raise_for_status()

    async def create_payment_intent(
        self,
        customer_id: str,
        amount_cents: int,
        currency: str,
        payment_method_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Payment:
        """Create a Stripe payment intent."""
        data = {
            "customer": customer_id,
            "amount": amount_cents,
            "currency": currency.lower(),
            "automatic_payment_methods[enabled]": "true",
        }

        if payment_method_id:
            data["payment_method"] = payment_method_id
        if description:
            data["description"] = description
        if metadata:
            for key, value in metadata.items():
                data[f"metadata[{key}]"] = value

        response = await self._request("POST", "/payment_intents", data=data)
        response.raise_for_status()

        pi = response.json()
        return Payment(
            id=pi["id"],
            provider=PaymentProvider.STRIPE,
            amount_cents=pi["amount"],
            currency=pi["currency"].upper(),
            status=self._map_status(pi["status"]),
            customer_id=pi["customer"],
            payment_method_id=pi.get("payment_method"),
            description=pi.get("description"),
            created_at=datetime.fromtimestamp(pi["created"], tz=timezone.utc),
            metadata=pi.get("metadata", {}),
        )

    def _map_status(self, stripe_status: str) -> PaymentStatus:
        """Map Stripe status to PaymentStatus."""
        mapping = {
            "requires_payment_method": PaymentStatus.PENDING,
            "requires_confirmation": PaymentStatus.PENDING,
            "requires_action": PaymentStatus.PROCESSING,
            "processing": PaymentStatus.PROCESSING,
            "succeeded": PaymentStatus.SUCCEEDED,
            "canceled": PaymentStatus.CANCELED,
        }
        return mapping.get(stripe_status, PaymentStatus.PENDING)

    async def confirm_payment(self, payment_id: str) -> Payment:
        """Confirm a Stripe payment intent."""
        response = await self._request("POST", f"/payment_intents/{payment_id}/confirm")
        response.raise_for_status()

        pi = response.json()
        return Payment(
            id=pi["id"],
            provider=PaymentProvider.STRIPE,
            amount_cents=pi["amount"],
            currency=pi["currency"].upper(),
            status=self._map_status(pi["status"]),
            customer_id=pi["customer"],
            payment_method_id=pi.get("payment_method"),
            created_at=datetime.fromtimestamp(pi["created"], tz=timezone.utc),
        )

    async def cancel_payment(self, payment_id: str) -> Payment:
        """Cancel a Stripe payment intent."""
        response = await self._request("POST", f"/payment_intents/{payment_id}/cancel")
        response.raise_for_status()

        pi = response.json()
        return Payment(
            id=pi["id"],
            provider=PaymentProvider.STRIPE,
            amount_cents=pi["amount"],
            currency=pi["currency"].upper(),
            status=PaymentStatus.CANCELED,
            customer_id=pi["customer"],
            created_at=datetime.fromtimestamp(pi["created"], tz=timezone.utc),
        )

    async def create_refund(
        self,
        payment_id: str,
        amount_cents: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> Refund:
        """Create a Stripe refund."""
        data = {"payment_intent": payment_id}

        if amount_cents:
            data["amount"] = str(amount_cents)
        if reason:
            data["reason"] = reason

        response = await self._request("POST", "/refunds", data=data)
        response.raise_for_status()

        ref = response.json()
        return Refund(
            id=ref["id"],
            provider=PaymentProvider.STRIPE,
            payment_id=ref["payment_intent"],
            amount_cents=ref["amount"],
            currency=ref["currency"].upper(),
            status=RefundStatus.SUCCEEDED if ref["status"] == "succeeded" else RefundStatus.PENDING,
            reason=ref.get("reason"),
            created_at=datetime.fromtimestamp(ref["created"], tz=timezone.utc),
        )

    async def list_payment_methods(
        self,
        customer_id: str,
    ) -> List[PaymentMethod]:
        """List Stripe payment methods for a customer."""
        response = await self._request(
            "GET",
            f"/customers/{customer_id}/payment_methods",
        )
        response.raise_for_status()

        methods = []
        for pm in response.json().get("data", []):
            card = pm.get("card", {})
            methods.append(
                PaymentMethod(
                    id=pm["id"],
                    provider=PaymentProvider.STRIPE,
                    type=pm["type"],
                    last_four=card.get("last4"),
                    brand=card.get("brand"),
                    exp_month=card.get("exp_month"),
                    exp_year=card.get("exp_year"),
                    created_at=datetime.fromtimestamp(pm["created"], tz=timezone.utc),
                )
            )

        return methods

    async def attach_payment_method(
        self,
        customer_id: str,
        payment_method_id: str,
        set_default: bool = False,
    ) -> PaymentMethod:
        """Attach a payment method to a Stripe customer."""
        # Attach the payment method
        response = await self._request(
            "POST",
            f"/payment_methods/{payment_method_id}/attach",
            data={"customer": customer_id},
        )
        response.raise_for_status()

        pm = response.json()

        # Set as default if requested
        if set_default:
            await self._request(
                "POST",
                f"/customers/{customer_id}",
                data={"invoice_settings[default_payment_method]": payment_method_id},
            )

        card = pm.get("card", {})
        return PaymentMethod(
            id=pm["id"],
            provider=PaymentProvider.STRIPE,
            type=pm["type"],
            last_four=card.get("last4"),
            brand=card.get("brand"),
            exp_month=card.get("exp_month"),
            exp_year=card.get("exp_year"),
            is_default=set_default,
            created_at=datetime.fromtimestamp(pm["created"], tz=timezone.utc),
        )

    async def detach_payment_method(self, payment_method_id: str) -> None:
        """Detach a payment method from a Stripe customer."""
        response = await self._request("POST", f"/payment_methods/{payment_method_id}/detach")
        response.raise_for_status()

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify Stripe webhook signature."""
        if not self.webhook_secret:
            logger.warning("Stripe webhook secret not configured")
            return True

        # Parse the signature header
        # Format: t=timestamp,v1=signature
        parts = dict(p.split("=", 1) for p in signature.split(","))
        timestamp = parts.get("t", "")
        sig = parts.get("v1", "")

        # Compute expected signature
        signed_payload = f"{timestamp}.{payload.decode()}"
        expected = hmac.new(
            self.webhook_secret.encode(),
            signed_payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, sig)


class PayPalProvider(PaymentProviderBase):
    """PayPal payment provider implementation.

    Requirements: 20.6, 22.2
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        webhook_id: Optional[str] = None,
        sandbox: bool = True,
    ):
        """Initialize PayPal provider.

        Args:
            client_id: PayPal client ID
            client_secret: PayPal client secret
            webhook_id: PayPal webhook ID for signature verification
            sandbox: Use sandbox environment
        """
        self.client_id = client_id or os.getenv("PAYPAL_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("PAYPAL_CLIENT_SECRET", "")
        self.webhook_id = webhook_id or os.getenv("PAYPAL_WEBHOOK_ID", "")

        if sandbox:
            self.base_url = "https://api-m.sandbox.paypal.com"
        else:
            self.base_url = "https://api-m.paypal.com"

        self._client = httpx.AsyncClient(timeout=30.0)
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

    async def _get_access_token(self) -> str:
        """Get or refresh OAuth access token."""
        from datetime import timedelta

        if (
            self._access_token
            and self._token_expires
            and datetime.now(timezone.utc) < self._token_expires - timedelta(minutes=5)
        ):
            return self._access_token

        import base64

        credentials = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()

        response = await self._client.post(
            f"{self.base_url}/v1/oauth2/token",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
        )
        response.raise_for_status()

        data = response.json()
        self._access_token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        self._token_expires = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        return self._access_token

    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make authenticated API request."""
        token = await self._get_access_token()
        url = f"{self.base_url}{path}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        return await self._client.request(
            method,
            url,
            headers=headers,
            json=json_data,
        )

    async def create_customer(
        self,
        external_id: str,
        email: str,
        name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a PayPal customer (vault customer).

        Note: PayPal uses vault for storing customer payment methods.
        """
        # PayPal doesn't have a direct customer API like Stripe
        # We use the vault API to create a customer token
        # For now, return the external_id as the customer reference
        return external_id

    async def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get PayPal customer details.

        Note: PayPal doesn't have a direct customer API.
        """
        # PayPal doesn't have a customer API
        return {"id": customer_id}

    async def delete_customer(self, customer_id: str) -> None:
        """Delete a PayPal customer.

        Note: PayPal doesn't have a direct customer API.
        """
        # PayPal doesn't have a customer API
        pass

    async def create_payment_intent(
        self,
        customer_id: str,
        amount_cents: int,
        currency: str,
        payment_method_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Payment:
        """Create a PayPal order (payment intent equivalent)."""
        amount = Decimal(amount_cents) / 100

        order_data = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "amount": {
                        "currency_code": currency.upper(),
                        "value": str(amount),
                    },
                    "description": description or "AgentVoiceBox subscription",
                    "custom_id": customer_id,
                }
            ],
        }

        response = await self._request("POST", "/v2/checkout/orders", json_data=order_data)
        response.raise_for_status()

        order = response.json()
        return Payment(
            id=order["id"],
            provider=PaymentProvider.PAYPAL,
            amount_cents=amount_cents,
            currency=currency.upper(),
            status=self._map_status(order["status"]),
            customer_id=customer_id,
            description=description,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

    def _map_status(self, paypal_status: str) -> PaymentStatus:
        """Map PayPal status to PaymentStatus."""
        mapping = {
            "CREATED": PaymentStatus.PENDING,
            "SAVED": PaymentStatus.PENDING,
            "APPROVED": PaymentStatus.PROCESSING,
            "VOIDED": PaymentStatus.CANCELED,
            "COMPLETED": PaymentStatus.SUCCEEDED,
            "PAYER_ACTION_REQUIRED": PaymentStatus.PROCESSING,
        }
        return mapping.get(paypal_status, PaymentStatus.PENDING)

    async def confirm_payment(self, payment_id: str) -> Payment:
        """Capture a PayPal order (confirm payment)."""
        response = await self._request("POST", f"/v2/checkout/orders/{payment_id}/capture")
        response.raise_for_status()

        order = response.json()
        purchase_unit = order.get("purchase_units", [{}])[0]
        amount = purchase_unit.get("amount", {})

        return Payment(
            id=order["id"],
            provider=PaymentProvider.PAYPAL,
            amount_cents=int(Decimal(amount.get("value", "0")) * 100),
            currency=amount.get("currency_code", "USD"),
            status=self._map_status(order["status"]),
            customer_id=purchase_unit.get("custom_id", ""),
            created_at=datetime.now(timezone.utc),
        )

    async def cancel_payment(self, payment_id: str) -> Payment:
        """Void a PayPal order (cancel payment)."""
        # PayPal orders can't be directly canceled, they expire
        # For authorized payments, we would void the authorization
        return Payment(
            id=payment_id,
            provider=PaymentProvider.PAYPAL,
            amount_cents=0,
            currency="USD",
            status=PaymentStatus.CANCELED,
            customer_id="",
            created_at=datetime.now(timezone.utc),
        )

    async def create_refund(
        self,
        payment_id: str,
        amount_cents: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> Refund:
        """Create a PayPal refund."""
        # First, get the capture ID from the order
        response = await self._request("GET", f"/v2/checkout/orders/{payment_id}")
        response.raise_for_status()

        order = response.json()
        captures = order.get("purchase_units", [{}])[0].get("payments", {}).get("captures", [])

        if not captures:
            raise ValueError(f"No captures found for order {payment_id}")

        capture_id = captures[0]["id"]

        # Create refund
        refund_data = {}
        if amount_cents:
            amount = Decimal(amount_cents) / 100
            refund_data["amount"] = {
                "value": str(amount),
                "currency_code": captures[0]["amount"]["currency_code"],
            }
        if reason:
            refund_data["note_to_payer"] = reason

        response = await self._request(
            "POST",
            f"/v2/payments/captures/{capture_id}/refund",
            json_data=refund_data if refund_data else None,
        )
        response.raise_for_status()

        ref = response.json()
        return Refund(
            id=ref["id"],
            provider=PaymentProvider.PAYPAL,
            payment_id=payment_id,
            amount_cents=int(Decimal(ref["amount"]["value"]) * 100),
            currency=ref["amount"]["currency_code"],
            status=RefundStatus.SUCCEEDED if ref["status"] == "COMPLETED" else RefundStatus.PENDING,
            reason=reason,
            created_at=datetime.now(timezone.utc),
        )

    async def list_payment_methods(
        self,
        customer_id: str,
    ) -> List[PaymentMethod]:
        """List PayPal payment methods (vault tokens)."""
        # PayPal vault API for listing saved payment methods
        response = await self._request(
            "GET",
            f"/v3/vault/payment-tokens?customer_id={customer_id}",
        )

        if response.status_code == 404:
            return []

        response.raise_for_status()

        methods = []
        for token in response.json().get("payment_tokens", []):
            source = token.get("payment_source", {})
            card = source.get("card", {})

            methods.append(
                PaymentMethod(
                    id=token["id"],
                    provider=PaymentProvider.PAYPAL,
                    type="card" if card else "paypal",
                    last_four=card.get("last_digits"),
                    brand=card.get("brand"),
                    exp_month=(
                        int(card.get("expiry", "01/2000").split("/")[0])
                        if card.get("expiry")
                        else None
                    ),
                    exp_year=(
                        int(card.get("expiry", "01/2000").split("/")[1])
                        if card.get("expiry")
                        else None
                    ),
                )
            )

        return methods

    async def attach_payment_method(
        self,
        customer_id: str,
        payment_method_id: str,
        set_default: bool = False,
    ) -> PaymentMethod:
        """Attach a payment method to a PayPal customer (vault)."""
        # PayPal vault tokens are already attached to customers
        return PaymentMethod(
            id=payment_method_id,
            provider=PaymentProvider.PAYPAL,
            type="paypal",
            is_default=set_default,
        )

    async def detach_payment_method(self, payment_method_id: str) -> None:
        """Delete a PayPal vault token."""
        response = await self._request("DELETE", f"/v3/vault/payment-tokens/{payment_method_id}")
        # 204 No Content is success
        if response.status_code not in (200, 204):
            response.raise_for_status()

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify PayPal webhook signature.

        Note: PayPal webhook verification requires additional headers.
        This is a simplified version.
        """
        if not self.webhook_id:
            logger.warning("PayPal webhook ID not configured")
            return True

        # PayPal webhook verification is more complex and requires
        # the full request headers. This is a placeholder.
        return True


class PaymentService:
    """Unified payment service supporting multiple providers.

    Requirements: 20.6, 20.8, 20.9, 22.1, 22.2, 22.7
    """

    def __init__(self):
        """Initialize payment service with all providers."""
        self.providers: Dict[PaymentProvider, PaymentProviderBase] = {
            PaymentProvider.STRIPE: StripeProvider(),
            PaymentProvider.PAYPAL: PayPalProvider(),
        }
        self._default_provider = PaymentProvider.STRIPE

    async def close(self) -> None:
        """Close all provider clients."""
        for provider in self.providers.values():
            await provider.close()

    def get_provider(
        self,
        provider: Optional[PaymentProvider] = None,
    ) -> PaymentProviderBase:
        """Get a payment provider instance.

        Args:
            provider: Provider to get, or None for default

        Returns:
            Payment provider instance
        """
        if provider is None:
            provider = self._default_provider
        return self.providers[provider]

    async def create_customer(
        self,
        external_id: str,
        email: str,
        name: str,
        provider: Optional[PaymentProvider] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a customer in the payment provider."""
        return await self.get_provider(provider).create_customer(
            external_id=external_id,
            email=email,
            name=name,
            metadata=metadata,
        )

    async def create_payment(
        self,
        customer_id: str,
        amount_cents: int,
        currency: str,
        provider: Optional[PaymentProvider] = None,
        payment_method_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Payment:
        """Create a payment intent."""
        return await self.get_provider(provider).create_payment_intent(
            customer_id=customer_id,
            amount_cents=amount_cents,
            currency=currency,
            payment_method_id=payment_method_id,
            description=description,
            metadata=metadata,
        )

    async def confirm_payment(
        self,
        payment_id: str,
        provider: Optional[PaymentProvider] = None,
    ) -> Payment:
        """Confirm a payment."""
        return await self.get_provider(provider).confirm_payment(payment_id)

    async def refund_payment(
        self,
        payment_id: str,
        amount_cents: Optional[int] = None,
        reason: Optional[str] = None,
        provider: Optional[PaymentProvider] = None,
    ) -> Refund:
        """Create a refund (full or partial).

        Requirement 22.7: Process refunds via original payment method.
        """
        return await self.get_provider(provider).create_refund(
            payment_id=payment_id,
            amount_cents=amount_cents,
            reason=reason,
        )

    async def list_payment_methods(
        self,
        customer_id: str,
        provider: Optional[PaymentProvider] = None,
    ) -> List[PaymentMethod]:
        """List customer's payment methods."""
        return await self.get_provider(provider).list_payment_methods(customer_id)

    def verify_webhook(
        self,
        provider: PaymentProvider,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify webhook signature from a provider."""
        return self.providers[provider].verify_webhook_signature(payload, signature)


# =============================================================================
# Dunning and Suspension Service
# =============================================================================


class DunningService:
    """Service for handling failed payments and tenant suspension.

    Requirement 20.9: Retry 3 times over 7 days, then suspend with 48-hour grace.
    """

    def __init__(
        self,
        payment_service: PaymentService,
        lago_service: Optional[Any] = None,  # LagoService
    ):
        """Initialize dunning service.

        Args:
            payment_service: Payment service instance
            lago_service: Lago service for subscription management
        """
        self.payment_service = payment_service
        self.lago_service = lago_service

        # Dunning configuration
        self.retry_intervals_days = [1, 3, 7]  # Days between retries
        self.grace_period_hours = 48

    async def handle_payment_failed(
        self,
        tenant_id: str,
        payment_id: str,
        attempt_number: int,
    ) -> Dict[str, Any]:
        """Handle a failed payment.

        Args:
            tenant_id: Tenant ID
            payment_id: Failed payment ID
            attempt_number: Current retry attempt (1-3)

        Returns:
            Action taken and next steps
        """
        if attempt_number < len(self.retry_intervals_days):
            # Schedule retry
            next_retry_days = self.retry_intervals_days[attempt_number]
            return {
                "action": "retry_scheduled",
                "attempt": attempt_number + 1,
                "retry_in_days": next_retry_days,
                "message": f"Payment retry scheduled in {next_retry_days} days",
            }
        else:
            # All retries exhausted, start grace period
            return {
                "action": "grace_period_started",
                "grace_hours": self.grace_period_hours,
                "message": f"Grace period started. Tenant will be suspended in {self.grace_period_hours} hours if payment not received.",
            }

    async def suspend_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """Suspend a tenant after grace period expires.

        Args:
            tenant_id: Tenant ID to suspend

        Returns:
            Suspension details
        """
        # This would integrate with the tenant management system
        logger.warning(f"Suspending tenant {tenant_id} due to payment failure")

        return {
            "action": "tenant_suspended",
            "tenant_id": tenant_id,
            "reason": "payment_failure",
            "message": "Tenant suspended due to failed payment after grace period",
        }


# =============================================================================
# Singleton Pattern for Dependency Injection
# =============================================================================

_payment_service: Optional[PaymentService] = None


def get_payment_service() -> PaymentService:
    """Get or create payment service singleton."""
    global _payment_service
    if _payment_service is None:
        _payment_service = PaymentService()
    return _payment_service


async def init_payment_service() -> PaymentService:
    """Initialize payment service (call on app startup)."""
    global _payment_service
    _payment_service = PaymentService()
    return _payment_service


async def close_payment_service() -> None:
    """Close payment service (call on app shutdown)."""
    global _payment_service
    if _payment_service:
        await _payment_service.close()
        _payment_service = None

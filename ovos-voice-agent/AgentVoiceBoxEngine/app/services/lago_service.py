"""Lago Billing Service for Usage-Based Billing.

This module provides integration with Lago for:
- Customer management (create, update, delete)
- Subscription management (create, upgrade, downgrade, cancel)
- Usage event metering (async, non-blocking)
- Invoice retrieval and management
- Webhook event handling

Requirements: 20.1, 20.2, 20.3, 20.4, 20.5
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


class PlanCode(str, Enum):
    """Available billing plan codes."""

    FREE = "free"
    PRO = "pro"
    PRO_ANNUAL = "pro_annual"
    ENTERPRISE = "enterprise"


class MetricCode(str, Enum):
    """Billable metric codes."""

    API_REQUESTS = "api_requests"
    AUDIO_MINUTES_INPUT = "audio_minutes_input"
    AUDIO_MINUTES_OUTPUT = "audio_minutes_output"
    LLM_TOKENS_INPUT = "llm_tokens_input"
    LLM_TOKENS_OUTPUT = "llm_tokens_output"
    CONCURRENT_CONNECTIONS = "concurrent_connections"
    CONNECTION_MINUTES = "connection_minutes"


class SubscriptionStatus(str, Enum):
    """Subscription status values."""

    ACTIVE = "active"
    PENDING = "pending"
    TERMINATED = "terminated"
    CANCELED = "canceled"


class InvoiceStatus(str, Enum):
    """Invoice status values."""

    DRAFT = "draft"
    FINALIZED = "finalized"
    VOIDED = "voided"


@dataclass
class LagoConfig:
    """Lago API configuration."""

    api_url: str = "http://localhost:3000"
    api_key: str = ""
    webhook_secret: str = ""
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "LagoConfig":
        """Create config from environment variables."""
        return cls(
            api_url=os.getenv("LAGO_API_URL", "http://localhost:3000"),
            api_key=os.getenv("LAGO_API_KEY", ""),
            webhook_secret=os.getenv("LAGO_WEBHOOK_SECRET", ""),
            timeout=float(os.getenv("LAGO_TIMEOUT", "30")),
            max_retries=int(os.getenv("LAGO_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("LAGO_RETRY_DELAY", "1.0")),
        )


@dataclass
class LagoCustomer:
    """Lago customer representation."""

    lago_id: str
    external_id: str
    name: str
    email: str
    currency: str = "USD"
    timezone: str = "UTC"
    billing_configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: List[Dict[str, str]] = field(default_factory=list)
    created_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LagoCustomer":
        """Create customer from Lago API response."""
        customer = data.get("customer", data)
        created_at = None
        if customer.get("created_at"):
            created_at = datetime.fromisoformat(customer["created_at"].replace("Z", "+00:00"))
        return cls(
            lago_id=customer.get("lago_id", ""),
            external_id=customer.get("external_id", ""),
            name=customer.get("name", ""),
            email=customer.get("email", ""),
            currency=customer.get("currency", "USD"),
            timezone=customer.get("timezone", "UTC"),
            billing_configuration=customer.get("billing_configuration", {}),
            metadata=customer.get("metadata", []),
            created_at=created_at,
        )


@dataclass
class LagoSubscription:
    """Lago subscription representation."""

    lago_id: str
    external_id: str
    external_customer_id: str
    plan_code: str
    status: SubscriptionStatus
    name: Optional[str] = None
    started_at: Optional[datetime] = None
    ending_at: Optional[datetime] = None
    terminated_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LagoSubscription":
        """Create subscription from Lago API response."""
        sub = data.get("subscription", data)

        def parse_dt(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            return datetime.fromisoformat(val.replace("Z", "+00:00"))

        return cls(
            lago_id=sub.get("lago_id", ""),
            external_id=sub.get("external_id", ""),
            external_customer_id=sub.get("external_customer_id", ""),
            plan_code=sub.get("plan_code", ""),
            status=SubscriptionStatus(sub.get("status", "pending")),
            name=sub.get("name"),
            started_at=parse_dt(sub.get("started_at")),
            ending_at=parse_dt(sub.get("ending_at")),
            terminated_at=parse_dt(sub.get("terminated_at")),
            canceled_at=parse_dt(sub.get("canceled_at")),
            created_at=parse_dt(sub.get("created_at")),
        )


@dataclass
class LagoInvoice:
    """Lago invoice representation."""

    lago_id: str
    sequential_id: int
    number: str
    status: InvoiceStatus
    payment_status: str
    currency: str
    total_amount_cents: int
    taxes_amount_cents: int
    sub_total_excluding_taxes_amount_cents: int
    issuing_date: Optional[datetime] = None
    payment_due_date: Optional[datetime] = None
    file_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LagoInvoice":
        """Create invoice from Lago API response."""
        inv = data.get("invoice", data)

        def parse_dt(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            return datetime.fromisoformat(val.replace("Z", "+00:00"))

        return cls(
            lago_id=inv.get("lago_id", ""),
            sequential_id=inv.get("sequential_id", 0),
            number=inv.get("number", ""),
            status=InvoiceStatus(inv.get("status", "draft")),
            payment_status=inv.get("payment_status", "pending"),
            currency=inv.get("currency", "USD"),
            total_amount_cents=inv.get("total_amount_cents", 0),
            taxes_amount_cents=inv.get("taxes_amount_cents", 0),
            sub_total_excluding_taxes_amount_cents=inv.get(
                "sub_total_excluding_taxes_amount_cents", 0
            ),
            issuing_date=parse_dt(inv.get("issuing_date")),
            payment_due_date=parse_dt(inv.get("payment_due_date")),
            file_url=inv.get("file_url"),
        )


@dataclass
class UsageEvent:
    """Usage event for metering."""

    transaction_id: str
    external_customer_id: str
    code: str
    timestamp: datetime
    properties: Dict[str, Any] = field(default_factory=dict)
    external_subscription_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Lago API format."""
        data = {
            "transaction_id": self.transaction_id,
            "external_customer_id": self.external_customer_id,
            "code": self.code,
            "timestamp": int(self.timestamp.timestamp()),
            "properties": self.properties,
        }
        if self.external_subscription_id:
            data["external_subscription_id"] = self.external_subscription_id
        return data


class LagoService:
    """Service for Lago billing operations.

    Provides methods for:
    - Customer CRUD operations
    - Subscription management
    - Usage event metering (async pipeline)
    - Invoice retrieval
    - Webhook handling

    Requirements: 20.1, 20.2, 20.3, 20.4, 20.5
    """

    def __init__(self, config: Optional[LagoConfig] = None):
        """Initialize Lago service.

        Args:
            config: Lago configuration. If None, loads from environment.
        """
        self.config = config or LagoConfig.from_env()
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            base_url=self.config.api_url,
        )
        # Async metering queue
        self._event_queue: asyncio.Queue[UsageEvent] = asyncio.Queue(maxsize=10000)
        self._metering_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self) -> None:
        """Start the async metering pipeline."""
        self._shutdown = False
        self._metering_task = asyncio.create_task(self._metering_worker())
        logger.info("Lago metering pipeline started")

    async def stop(self) -> None:
        """Stop the async metering pipeline gracefully."""
        self._shutdown = True
        if self._metering_task:
            # Wait for queue to drain (max 10 seconds)
            try:
                await asyncio.wait_for(self._event_queue.join(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Metering queue not fully drained, {self._event_queue.qsize()} events remaining"
                )
            self._metering_task.cancel()
            try:
                await self._metering_task
            except asyncio.CancelledError:
                pass
        logger.info("Lago metering pipeline stopped")

    async def close(self) -> None:
        """Close HTTP client and stop metering."""
        await self.stop()
        await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make authenticated API request with retry logic.

        Args:
            method: HTTP method
            path: API path
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        headers = kwargs.pop("headers", {})
        headers.update(self._headers())

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.request(method, path, headers=headers, **kwargs)
                return response
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    logger.warning(f"Lago request failed (attempt {attempt + 1}), retrying: {e}")

        raise last_error or Exception("Request failed after retries")

    # =========================================================================
    # Customer Management
    # =========================================================================

    async def create_customer(
        self,
        external_id: str,
        name: str,
        email: str,
        currency: str = "USD",
        timezone: str = "UTC",
        billing_configuration: Optional[Dict[str, Any]] = None,
        metadata: Optional[List[Dict[str, str]]] = None,
    ) -> LagoCustomer:
        """Create a new customer in Lago.

        Args:
            external_id: External customer ID (tenant_id)
            name: Customer/organization name
            email: Billing email address
            currency: Billing currency (default: USD)
            timezone: Customer timezone
            billing_configuration: Payment provider config
            metadata: Additional metadata key-value pairs

        Returns:
            Created customer object
        """
        customer_data = {
            "customer": {
                "external_id": external_id,
                "name": name,
                "email": email,
                "currency": currency,
                "timezone": timezone,
            }
        }

        if billing_configuration:
            customer_data["customer"]["billing_configuration"] = billing_configuration

        if metadata:
            customer_data["customer"]["metadata"] = metadata

        response = await self._request("POST", "/api/v1/customers", json=customer_data)
        response.raise_for_status()

        return LagoCustomer.from_dict(response.json())

    async def get_customer(self, external_id: str) -> Optional[LagoCustomer]:
        """Get customer by external ID.

        Args:
            external_id: External customer ID (tenant_id)

        Returns:
            Customer object or None if not found
        """
        response = await self._request("GET", f"/api/v1/customers/{external_id}")

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return LagoCustomer.from_dict(response.json())

    async def update_customer(
        self,
        external_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        currency: Optional[str] = None,
        timezone: Optional[str] = None,
        billing_configuration: Optional[Dict[str, Any]] = None,
        metadata: Optional[List[Dict[str, str]]] = None,
    ) -> LagoCustomer:
        """Update customer attributes.

        Args:
            external_id: External customer ID
            name: New name (optional)
            email: New email (optional)
            currency: New currency (optional)
            timezone: New timezone (optional)
            billing_configuration: New billing config (optional)
            metadata: New metadata (optional)

        Returns:
            Updated customer object
        """
        update_data: Dict[str, Any] = {"customer": {"external_id": external_id}}

        if name is not None:
            update_data["customer"]["name"] = name
        if email is not None:
            update_data["customer"]["email"] = email
        if currency is not None:
            update_data["customer"]["currency"] = currency
        if timezone is not None:
            update_data["customer"]["timezone"] = timezone
        if billing_configuration is not None:
            update_data["customer"]["billing_configuration"] = billing_configuration
        if metadata is not None:
            update_data["customer"]["metadata"] = metadata

        response = await self._request("PUT", f"/api/v1/customers/{external_id}", json=update_data)
        response.raise_for_status()

        return LagoCustomer.from_dict(response.json())

    async def delete_customer(self, external_id: str) -> None:
        """Delete a customer.

        Args:
            external_id: External customer ID
        """
        response = await self._request("DELETE", f"/api/v1/customers/{external_id}")
        response.raise_for_status()

    async def list_customers(
        self,
        page: int = 1,
        per_page: int = 20,
    ) -> List[LagoCustomer]:
        """List customers with pagination.

        Args:
            page: Page number (1-indexed)
            per_page: Results per page

        Returns:
            List of customers
        """
        response = await self._request(
            "GET",
            "/api/v1/customers",
            params={"page": page, "per_page": per_page},
        )
        response.raise_for_status()

        data = response.json()
        return [LagoCustomer.from_dict(c) for c in data.get("customers", [])]

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def create_subscription(
        self,
        external_customer_id: str,
        plan_code: Union[str, PlanCode],
        external_id: Optional[str] = None,
        name: Optional[str] = None,
        billing_time: str = "calendar",
    ) -> LagoSubscription:
        """Create a new subscription.

        Args:
            external_customer_id: Customer's external ID (tenant_id)
            plan_code: Plan code to subscribe to
            external_id: External subscription ID (auto-generated if not provided)
            name: Subscription display name
            billing_time: "calendar" or "anniversary"

        Returns:
            Created subscription object
        """
        if isinstance(plan_code, PlanCode):
            plan_code = plan_code.value

        if external_id is None:
            external_id = f"sub_{uuid.uuid4().hex[:16]}"

        sub_data = {
            "subscription": {
                "external_customer_id": external_customer_id,
                "plan_code": plan_code,
                "external_id": external_id,
                "billing_time": billing_time,
            }
        }

        if name:
            sub_data["subscription"]["name"] = name

        response = await self._request("POST", "/api/v1/subscriptions", json=sub_data)
        response.raise_for_status()

        return LagoSubscription.from_dict(response.json())

    async def get_subscription(self, external_id: str) -> Optional[LagoSubscription]:
        """Get subscription by external ID.

        Args:
            external_id: External subscription ID

        Returns:
            Subscription object or None if not found
        """
        response = await self._request("GET", f"/api/v1/subscriptions/{external_id}")

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return LagoSubscription.from_dict(response.json())

    async def update_subscription(
        self,
        external_id: str,
        plan_code: Optional[Union[str, PlanCode]] = None,
        name: Optional[str] = None,
    ) -> LagoSubscription:
        """Update subscription (upgrade/downgrade plan).

        Args:
            external_id: External subscription ID
            plan_code: New plan code (for upgrade/downgrade)
            name: New display name

        Returns:
            Updated subscription object
        """
        update_data: Dict[str, Any] = {"subscription": {}}

        if plan_code is not None:
            if isinstance(plan_code, PlanCode):
                plan_code = plan_code.value
            update_data["subscription"]["plan_code"] = plan_code

        if name is not None:
            update_data["subscription"]["name"] = name

        response = await self._request(
            "PUT", f"/api/v1/subscriptions/{external_id}", json=update_data
        )
        response.raise_for_status()

        return LagoSubscription.from_dict(response.json())

    async def terminate_subscription(self, external_id: str) -> LagoSubscription:
        """Terminate a subscription immediately.

        Args:
            external_id: External subscription ID

        Returns:
            Terminated subscription object
        """
        response = await self._request("DELETE", f"/api/v1/subscriptions/{external_id}")
        response.raise_for_status()

        return LagoSubscription.from_dict(response.json())

    async def list_subscriptions(
        self,
        external_customer_id: Optional[str] = None,
        status: Optional[List[SubscriptionStatus]] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> List[LagoSubscription]:
        """List subscriptions with optional filtering.

        Args:
            external_customer_id: Filter by customer
            status: Filter by status(es)
            page: Page number
            per_page: Results per page

        Returns:
            List of subscriptions
        """
        params: Dict[str, Any] = {"page": page, "per_page": per_page}

        if external_customer_id:
            params["external_customer_id"] = external_customer_id

        if status:
            params["status[]"] = [s.value for s in status]

        response = await self._request("GET", "/api/v1/subscriptions", params=params)
        response.raise_for_status()

        data = response.json()
        return [LagoSubscription.from_dict(s) for s in data.get("subscriptions", [])]

    # =========================================================================
    # Invoice Management
    # =========================================================================

    async def get_invoice(self, lago_id: str) -> Optional[LagoInvoice]:
        """Get invoice by Lago ID.

        Args:
            lago_id: Lago invoice ID

        Returns:
            Invoice object or None if not found
        """
        response = await self._request("GET", f"/api/v1/invoices/{lago_id}")

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return LagoInvoice.from_dict(response.json())

    async def list_invoices(
        self,
        external_customer_id: Optional[str] = None,
        status: Optional[InvoiceStatus] = None,
        payment_status: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> List[LagoInvoice]:
        """List invoices with optional filtering.

        Args:
            external_customer_id: Filter by customer
            status: Filter by invoice status
            payment_status: Filter by payment status
            page: Page number
            per_page: Results per page

        Returns:
            List of invoices
        """
        params: Dict[str, Any] = {"page": page, "per_page": per_page}

        if external_customer_id:
            params["external_customer_id"] = external_customer_id
        if status:
            params["status"] = status.value
        if payment_status:
            params["payment_status"] = payment_status

        response = await self._request("GET", "/api/v1/invoices", params=params)
        response.raise_for_status()

        data = response.json()
        return [LagoInvoice.from_dict(i) for i in data.get("invoices", [])]

    async def download_invoice(self, lago_id: str) -> Optional[str]:
        """Get invoice PDF download URL.

        Args:
            lago_id: Lago invoice ID

        Returns:
            PDF download URL or None
        """
        response = await self._request("POST", f"/api/v1/invoices/{lago_id}/download")

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()
        return data.get("invoice", {}).get("file_url")

    async def refresh_invoice(self, lago_id: str) -> LagoInvoice:
        """Refresh a draft invoice to recalculate charges.

        Args:
            lago_id: Lago invoice ID

        Returns:
            Refreshed invoice object
        """
        response = await self._request("PUT", f"/api/v1/invoices/{lago_id}/refresh")
        response.raise_for_status()
        return LagoInvoice.from_dict(response.json())

    async def finalize_invoice(self, lago_id: str) -> LagoInvoice:
        """Finalize a draft invoice.

        Args:
            lago_id: Lago invoice ID

        Returns:
            Finalized invoice object
        """
        response = await self._request("PUT", f"/api/v1/invoices/{lago_id}/finalize")
        response.raise_for_status()
        return LagoInvoice.from_dict(response.json())

    # =========================================================================
    # Usage Metering (Async Pipeline)
    # =========================================================================

    async def _metering_worker(self) -> None:
        """Background worker that batches and sends usage events.

        Requirement 20.5: Send metering data within 60 seconds (async, non-blocking)
        """
        batch: List[UsageEvent] = []
        batch_start = time.monotonic()
        max_batch_size = 100
        max_batch_age = 5.0  # seconds

        while not self._shutdown:
            try:
                # Wait for event with timeout
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                    batch.append(event)
                    self._event_queue.task_done()
                except asyncio.TimeoutError:
                    pass

                # Send batch if full or old enough
                batch_age = time.monotonic() - batch_start
                if batch and (len(batch) >= max_batch_size or batch_age >= max_batch_age):
                    await self._send_batch(batch)
                    batch = []
                    batch_start = time.monotonic()

            except Exception as e:
                logger.error(f"Metering worker error: {e}")
                await asyncio.sleep(1.0)

        # Send remaining events on shutdown
        if batch:
            await self._send_batch(batch)

    async def _send_batch(self, events: List[UsageEvent]) -> None:
        """Send a batch of usage events to Lago.

        Args:
            events: List of usage events to send
        """
        if not events:
            return

        # Lago accepts batch events
        batch_data = {"events": [e.to_dict() for e in events]}

        try:
            response = await self._request("POST", "/api/v1/events/batch", json=batch_data)
            if response.status_code >= 400:
                logger.error(
                    f"Failed to send usage batch: {response.status_code} - {response.text}"
                )
            else:
                logger.debug(f"Sent {len(events)} usage events to Lago")
        except Exception as e:
            logger.error(f"Failed to send usage batch: {e}")

    def queue_usage_event(
        self,
        external_customer_id: str,
        code: Union[str, MetricCode],
        properties: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        external_subscription_id: Optional[str] = None,
    ) -> bool:
        """Queue a usage event for async sending.

        This method is non-blocking and returns immediately.
        Events are batched and sent by the background worker.

        Args:
            external_customer_id: Customer's external ID (tenant_id)
            code: Billable metric code
            properties: Event properties (e.g., {"duration_minutes": 5.5})
            transaction_id: Unique transaction ID (auto-generated if not provided)
            timestamp: Event timestamp (defaults to now)
            external_subscription_id: Specific subscription ID (optional)

        Returns:
            True if queued successfully, False if queue is full
        """
        if isinstance(code, MetricCode):
            code = code.value

        if transaction_id is None:
            transaction_id = f"evt_{uuid.uuid4().hex}"

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        event = UsageEvent(
            transaction_id=transaction_id,
            external_customer_id=external_customer_id,
            code=code,
            timestamp=timestamp,
            properties=properties or {},
            external_subscription_id=external_subscription_id,
        )

        try:
            self._event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            logger.warning("Usage event queue full, dropping event")
            return False

    async def send_usage_event(
        self,
        external_customer_id: str,
        code: Union[str, MetricCode],
        properties: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        external_subscription_id: Optional[str] = None,
    ) -> bool:
        """Send a usage event immediately (synchronous).

        Use queue_usage_event() for non-blocking operation.

        Args:
            external_customer_id: Customer's external ID (tenant_id)
            code: Billable metric code
            properties: Event properties
            transaction_id: Unique transaction ID
            timestamp: Event timestamp
            external_subscription_id: Specific subscription ID

        Returns:
            True if sent successfully
        """
        if isinstance(code, MetricCode):
            code = code.value

        if transaction_id is None:
            transaction_id = f"evt_{uuid.uuid4().hex}"

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        event_data = {
            "event": {
                "transaction_id": transaction_id,
                "external_customer_id": external_customer_id,
                "code": code,
                "timestamp": int(timestamp.timestamp()),
                "properties": properties or {},
            }
        }

        if external_subscription_id:
            event_data["event"]["external_subscription_id"] = external_subscription_id

        try:
            response = await self._request("POST", "/api/v1/events", json=event_data)
            return response.status_code < 400
        except Exception as e:
            logger.error(f"Failed to send usage event: {e}")
            return False

    # =========================================================================
    # Convenience Methods for Common Metrics
    # =========================================================================

    def track_api_request(self, tenant_id: str) -> bool:
        """Track an API request for billing.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if queued successfully
        """
        return self.queue_usage_event(
            external_customer_id=tenant_id,
            code=MetricCode.API_REQUESTS,
        )

    def track_audio_input(self, tenant_id: str, duration_minutes: float) -> bool:
        """Track audio input (STT) for billing.

        Args:
            tenant_id: Tenant ID
            duration_minutes: Audio duration in minutes

        Returns:
            True if queued successfully
        """
        return self.queue_usage_event(
            external_customer_id=tenant_id,
            code=MetricCode.AUDIO_MINUTES_INPUT,
            properties={"duration_minutes": duration_minutes},
        )

    def track_audio_output(self, tenant_id: str, duration_minutes: float) -> bool:
        """Track audio output (TTS) for billing.

        Args:
            tenant_id: Tenant ID
            duration_minutes: Audio duration in minutes

        Returns:
            True if queued successfully
        """
        return self.queue_usage_event(
            external_customer_id=tenant_id,
            code=MetricCode.AUDIO_MINUTES_OUTPUT,
            properties={"duration_minutes": duration_minutes},
        )

    def track_llm_tokens(
        self,
        tenant_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        """Track LLM token usage for billing.

        Args:
            tenant_id: Tenant ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            True if both events queued successfully
        """
        success = True
        if input_tokens > 0:
            success = success and self.queue_usage_event(
                external_customer_id=tenant_id,
                code=MetricCode.LLM_TOKENS_INPUT,
                properties={"token_count": input_tokens},
            )
        if output_tokens > 0:
            success = success and self.queue_usage_event(
                external_customer_id=tenant_id,
                code=MetricCode.LLM_TOKENS_OUTPUT,
                properties={"token_count": output_tokens},
            )
        return success

    def track_connection(self, tenant_id: str, duration_minutes: float) -> bool:
        """Track WebSocket connection duration for billing.

        Args:
            tenant_id: Tenant ID
            duration_minutes: Connection duration in minutes

        Returns:
            True if queued successfully
        """
        return self.queue_usage_event(
            external_customer_id=tenant_id,
            code=MetricCode.CONNECTION_MINUTES,
            properties={"duration_minutes": duration_minutes},
        )

    def track_concurrent_connections(
        self,
        tenant_id: str,
        connection_count: int,
    ) -> bool:
        """Track peak concurrent connections for billing.

        Args:
            tenant_id: Tenant ID
            connection_count: Number of concurrent connections

        Returns:
            True if queued successfully
        """
        return self.queue_usage_event(
            external_customer_id=tenant_id,
            code=MetricCode.CONCURRENT_CONNECTIONS,
            properties={"connection_count": connection_count},
        )

    # =========================================================================
    # Webhook Handling
    # =========================================================================

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify webhook signature from Lago.

        Args:
            payload: Raw request body
            signature: X-Lago-Signature header value

        Returns:
            True if signature is valid
        """
        import hashlib
        import hmac

        if not self.config.webhook_secret:
            logger.warning("Webhook secret not configured, skipping verification")
            return True

        expected = hmac.new(
            self.config.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)


# =============================================================================
# Singleton Pattern for Dependency Injection
# =============================================================================

_lago_service: Optional[LagoService] = None


def get_lago_service() -> LagoService:
    """Get or create Lago service singleton."""
    global _lago_service
    if _lago_service is None:
        _lago_service = LagoService()
    return _lago_service


async def init_lago_service() -> LagoService:
    """Initialize Lago service and start metering pipeline.

    Call on app startup.
    """
    global _lago_service
    _lago_service = LagoService()
    await _lago_service.start()
    return _lago_service


async def close_lago_service() -> None:
    """Close Lago service and stop metering pipeline.

    Call on app shutdown.
    """
    global _lago_service
    if _lago_service:
        await _lago_service.close()
        _lago_service = None

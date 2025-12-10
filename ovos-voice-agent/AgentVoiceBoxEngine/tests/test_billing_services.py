"""Unit Tests for Billing Services (Lago and Payment).

Tests for:
- LagoService: Customer, subscription, invoice, and metering operations
- PaymentService: Stripe and PayPal provider logic
- DunningService: Failed payment handling and suspension flow

Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.8, 20.9, 22.1, 22.2, 22.7

NOTE: These are unit tests that verify internal logic and data structures.
Integration tests with real Lago/Stripe/PayPal APIs are in Phase 13 (Task 30.2).
"""

from __future__ import annotations

import hashlib
import hmac

# Import the modules under test
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.lago_service import (  # noqa: E402
    InvoiceStatus,
    LagoConfig,
    LagoCustomer,
    LagoInvoice,
    LagoService,
    LagoSubscription,
    MetricCode,
    PlanCode,
    SubscriptionStatus,
    UsageEvent,
)
from app.services.payment_service import (  # noqa: E402
    DunningService,
    Payment,
    PaymentMethod,
    PaymentProvider,
    PaymentService,
    PaymentStatus,
    PayPalProvider,
    Refund,
    RefundStatus,
    StripeProvider,
)

# =============================================================================
# LagoService Tests
# =============================================================================


class TestLagoConfig:
    """Tests for LagoConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = LagoConfig()

        assert config.api_url == "http://localhost:3000"
        assert config.api_key == ""
        assert config.webhook_secret == ""
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_from_env_with_defaults(self):
        """Verify from_env uses defaults when env vars not set."""
        with patch.dict("os.environ", {}, clear=True):
            config = LagoConfig.from_env()
            assert config.api_url == "http://localhost:3000"

    def test_from_env_with_values(self):
        """Verify from_env reads environment variables."""
        env = {
            "LAGO_API_URL": "https://lago.example.com",
            "LAGO_API_KEY": "test-api-key",
            "LAGO_WEBHOOK_SECRET": "webhook-secret",
            "LAGO_TIMEOUT": "60",
            "LAGO_MAX_RETRIES": "5",
            "LAGO_RETRY_DELAY": "2.0",
        }
        with patch.dict("os.environ", env, clear=True):
            config = LagoConfig.from_env()

            assert config.api_url == "https://lago.example.com"
            assert config.api_key == "test-api-key"
            assert config.webhook_secret == "webhook-secret"
            assert config.timeout == 60.0
            assert config.max_retries == 5
            assert config.retry_delay == 2.0


class TestLagoCustomer:
    """Tests for LagoCustomer dataclass."""

    def test_from_dict_full(self):
        """Verify customer creation from full API response."""
        data = {
            "customer": {
                "lago_id": "lago-123",
                "external_id": "tenant-456",
                "name": "Test Company",
                "email": "billing@test.com",
                "currency": "EUR",
                "timezone": "Europe/London",
                "billing_configuration": {"payment_provider": "stripe"},
                "metadata": [{"key": "tier", "value": "pro"}],
                "created_at": "2024-01-15T10:30:00Z",
            }
        }

        customer = LagoCustomer.from_dict(data)

        assert customer.lago_id == "lago-123"
        assert customer.external_id == "tenant-456"
        assert customer.name == "Test Company"
        assert customer.email == "billing@test.com"
        assert customer.currency == "EUR"
        assert customer.timezone == "Europe/London"
        assert customer.billing_configuration == {"payment_provider": "stripe"}
        assert customer.metadata == [{"key": "tier", "value": "pro"}]
        assert customer.created_at is not None

    def test_from_dict_minimal(self):
        """Verify customer creation from minimal API response."""
        data = {
            "lago_id": "lago-123",
            "external_id": "tenant-456",
            "name": "Test",
            "email": "test@test.com",
        }

        customer = LagoCustomer.from_dict(data)

        assert customer.lago_id == "lago-123"
        assert customer.currency == "USD"  # Default
        assert customer.timezone == "UTC"  # Default
        assert customer.created_at is None


class TestLagoSubscription:
    """Tests for LagoSubscription dataclass."""

    def test_from_dict_active(self):
        """Verify subscription creation from active subscription response."""
        data = {
            "subscription": {
                "lago_id": "sub-lago-123",
                "external_id": "sub-ext-456",
                "external_customer_id": "tenant-789",
                "plan_code": "pro",
                "status": "active",
                "name": "Pro Plan",
                "started_at": "2024-01-01T00:00:00Z",
                "created_at": "2024-01-01T00:00:00Z",
            }
        }

        sub = LagoSubscription.from_dict(data)

        assert sub.lago_id == "sub-lago-123"
        assert sub.external_id == "sub-ext-456"
        assert sub.plan_code == "pro"
        assert sub.status == SubscriptionStatus.ACTIVE
        assert sub.started_at is not None

    def test_from_dict_canceled(self):
        """Verify subscription creation from canceled subscription."""
        data = {
            "subscription": {
                "lago_id": "sub-123",
                "external_id": "sub-456",
                "external_customer_id": "tenant-789",
                "plan_code": "free",
                "status": "canceled",
                "canceled_at": "2024-02-01T00:00:00Z",
            }
        }

        sub = LagoSubscription.from_dict(data)

        assert sub.status == SubscriptionStatus.CANCELED
        assert sub.canceled_at is not None


class TestLagoInvoice:
    """Tests for LagoInvoice dataclass."""

    def test_from_dict_finalized(self):
        """Verify invoice creation from finalized invoice response."""
        data = {
            "invoice": {
                "lago_id": "inv-123",
                "sequential_id": 42,
                "number": "INV-2024-0042",
                "status": "finalized",
                "payment_status": "succeeded",
                "currency": "USD",
                "total_amount_cents": 4900,
                "taxes_amount_cents": 0,
                "sub_total_excluding_taxes_amount_cents": 4900,
                "issuing_date": "2024-02-01T00:00:00Z",
                "payment_due_date": "2024-02-15T00:00:00Z",
                "file_url": "https://lago.example.com/invoices/inv-123.pdf",
            }
        }

        invoice = LagoInvoice.from_dict(data)

        assert invoice.lago_id == "inv-123"
        assert invoice.number == "INV-2024-0042"
        assert invoice.status == InvoiceStatus.FINALIZED
        assert invoice.total_amount_cents == 4900
        assert invoice.file_url is not None


class TestUsageEvent:
    """Tests for UsageEvent dataclass."""

    def test_to_dict_basic(self):
        """Verify usage event serialization."""
        test_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        event = UsageEvent(
            transaction_id="evt-123",
            external_customer_id="tenant-456",
            code="api_requests",
            timestamp=test_time,
            properties={"count": 1},
        )

        data = event.to_dict()

        assert data["transaction_id"] == "evt-123"
        assert data["external_customer_id"] == "tenant-456"
        assert data["code"] == "api_requests"
        assert data["timestamp"] == int(test_time.timestamp())  # Unix timestamp
        assert data["properties"] == {"count": 1}

    def test_to_dict_with_subscription(self):
        """Verify usage event with subscription ID."""
        event = UsageEvent(
            transaction_id="evt-123",
            external_customer_id="tenant-456",
            code="audio_minutes_input",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            properties={"duration_minutes": 5.5},
            external_subscription_id="sub-789",
        )

        data = event.to_dict()

        assert data["external_subscription_id"] == "sub-789"


class TestLagoServiceMeteringQueue:
    """Tests for LagoService metering queue functionality."""

    def test_queue_usage_event_success(self):
        """Verify usage event is queued successfully."""
        config = LagoConfig(api_key="test-key")
        service = LagoService(config)

        result = service.queue_usage_event(
            external_customer_id="tenant-123",
            code=MetricCode.API_REQUESTS,
        )

        assert result is True
        assert service._event_queue.qsize() == 1

    def test_queue_usage_event_with_properties(self):
        """Verify usage event with properties is queued."""
        config = LagoConfig(api_key="test-key")
        service = LagoService(config)

        result = service.queue_usage_event(
            external_customer_id="tenant-123",
            code=MetricCode.AUDIO_MINUTES_INPUT,
            properties={"duration_minutes": 2.5},
        )

        assert result is True

    def test_track_api_request(self):
        """Verify track_api_request convenience method."""
        config = LagoConfig(api_key="test-key")
        service = LagoService(config)

        result = service.track_api_request("tenant-123")

        assert result is True
        assert service._event_queue.qsize() == 1

    def test_track_audio_input(self):
        """Verify track_audio_input convenience method."""
        config = LagoConfig(api_key="test-key")
        service = LagoService(config)

        result = service.track_audio_input("tenant-123", duration_minutes=3.5)

        assert result is True

    def test_track_audio_output(self):
        """Verify track_audio_output convenience method."""
        config = LagoConfig(api_key="test-key")
        service = LagoService(config)

        result = service.track_audio_output("tenant-123", duration_minutes=2.0)

        assert result is True

    def test_track_llm_tokens(self):
        """Verify track_llm_tokens queues both input and output events."""
        config = LagoConfig(api_key="test-key")
        service = LagoService(config)

        result = service.track_llm_tokens(
            tenant_id="tenant-123",
            input_tokens=100,
            output_tokens=50,
        )

        assert result is True
        assert service._event_queue.qsize() == 2  # Two events queued

    def test_track_connection(self):
        """Verify track_connection convenience method."""
        config = LagoConfig(api_key="test-key")
        service = LagoService(config)

        result = service.track_connection("tenant-123", duration_minutes=15.0)

        assert result is True


class TestLagoServiceWebhook:
    """Tests for LagoService webhook verification."""

    def test_verify_webhook_signature_valid(self):
        """Verify valid webhook signature passes."""
        secret = "test-webhook-secret"
        config = LagoConfig(api_key="test-key", webhook_secret=secret)
        service = LagoService(config)

        payload = b'{"event": "invoice.created"}'
        expected_sig = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        result = service.verify_webhook_signature(payload, expected_sig)

        assert result is True

    def test_verify_webhook_signature_invalid(self):
        """Verify invalid webhook signature fails."""
        config = LagoConfig(api_key="test-key", webhook_secret="real-secret")
        service = LagoService(config)

        payload = b'{"event": "invoice.created"}'

        result = service.verify_webhook_signature(payload, "invalid-signature")

        assert result is False

    def test_verify_webhook_no_secret_configured(self):
        """Verify webhook passes when no secret configured (dev mode)."""
        config = LagoConfig(api_key="test-key", webhook_secret="")
        service = LagoService(config)

        result = service.verify_webhook_signature(b"payload", "any-sig")

        assert result is True  # Passes in dev mode


# =============================================================================
# PaymentService Tests
# =============================================================================


class TestPaymentDataclasses:
    """Tests for payment-related dataclasses."""

    def test_payment_method_card(self):
        """Verify PaymentMethod for card."""
        pm = PaymentMethod(
            id="pm_123",
            provider=PaymentProvider.STRIPE,
            type="card",
            last_four="4242",
            brand="visa",
            exp_month=12,
            exp_year=2025,
            is_default=True,
        )

        assert pm.id == "pm_123"
        assert pm.provider == PaymentProvider.STRIPE
        assert pm.last_four == "4242"
        assert pm.brand == "visa"
        assert pm.is_default is True

    def test_payment_succeeded(self):
        """Verify Payment dataclass."""
        payment = Payment(
            id="pi_123",
            provider=PaymentProvider.STRIPE,
            amount_cents=4900,
            currency="USD",
            status=PaymentStatus.SUCCEEDED,
            customer_id="cus_123",
            payment_method_id="pm_456",
            description="Pro subscription",
        )

        assert payment.amount_cents == 4900
        assert payment.status == PaymentStatus.SUCCEEDED

    def test_refund_dataclass(self):
        """Verify Refund dataclass."""
        refund = Refund(
            id="re_123",
            provider=PaymentProvider.STRIPE,
            payment_id="pi_456",
            amount_cents=2450,
            currency="USD",
            status=RefundStatus.SUCCEEDED,
            reason="customer_request",
        )

        assert refund.amount_cents == 2450
        assert refund.status == RefundStatus.SUCCEEDED


class TestStripeProviderStatusMapping:
    """Tests for Stripe status mapping."""

    def test_map_status_succeeded(self):
        """Verify succeeded status mapping."""
        provider = StripeProvider(api_key="test")

        assert provider._map_status("succeeded") == PaymentStatus.SUCCEEDED

    def test_map_status_processing(self):
        """Verify processing status mapping."""
        provider = StripeProvider(api_key="test")

        assert provider._map_status("processing") == PaymentStatus.PROCESSING

    def test_map_status_requires_action(self):
        """Verify requires_action maps to processing."""
        provider = StripeProvider(api_key="test")

        assert provider._map_status("requires_action") == PaymentStatus.PROCESSING

    def test_map_status_canceled(self):
        """Verify canceled status mapping."""
        provider = StripeProvider(api_key="test")

        assert provider._map_status("canceled") == PaymentStatus.CANCELED

    def test_map_status_unknown(self):
        """Verify unknown status defaults to pending."""
        provider = StripeProvider(api_key="test")

        assert provider._map_status("unknown_status") == PaymentStatus.PENDING


class TestStripeWebhookVerification:
    """Tests for Stripe webhook signature verification."""

    def test_verify_valid_signature(self):
        """Verify valid Stripe webhook signature."""
        secret = "whsec_test_secret"
        provider = StripeProvider(api_key="test", webhook_secret=secret)

        timestamp = "1704067200"
        payload = b'{"type": "payment_intent.succeeded"}'
        signed_payload = f"{timestamp}.{payload.decode()}"

        sig = hmac.new(
            secret.encode(),
            signed_payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        signature_header = f"t={timestamp},v1={sig}"

        result = provider.verify_webhook_signature(payload, signature_header)

        assert result is True

    def test_verify_invalid_signature(self):
        """Verify invalid Stripe webhook signature fails."""
        provider = StripeProvider(api_key="test", webhook_secret="real_secret")

        result = provider.verify_webhook_signature(b"payload", "t=123,v1=invalid_sig")

        assert result is False


class TestPayPalProviderStatusMapping:
    """Tests for PayPal status mapping."""

    def test_map_status_completed(self):
        """Verify COMPLETED maps to succeeded."""
        provider = PayPalProvider(client_id="test", client_secret="test")

        assert provider._map_status("COMPLETED") == PaymentStatus.SUCCEEDED

    def test_map_status_created(self):
        """Verify CREATED maps to pending."""
        provider = PayPalProvider(client_id="test", client_secret="test")

        assert provider._map_status("CREATED") == PaymentStatus.PENDING

    def test_map_status_approved(self):
        """Verify APPROVED maps to processing."""
        provider = PayPalProvider(client_id="test", client_secret="test")

        assert provider._map_status("APPROVED") == PaymentStatus.PROCESSING


class TestPaymentServiceProviderSelection:
    """Tests for PaymentService provider selection."""

    def test_get_default_provider(self):
        """Verify default provider is Stripe."""
        service = PaymentService()

        provider = service.get_provider()

        assert isinstance(provider, StripeProvider)

    def test_get_stripe_provider(self):
        """Verify explicit Stripe provider selection."""
        service = PaymentService()

        provider = service.get_provider(PaymentProvider.STRIPE)

        assert isinstance(provider, StripeProvider)

    def test_get_paypal_provider(self):
        """Verify explicit PayPal provider selection."""
        service = PaymentService()

        provider = service.get_provider(PaymentProvider.PAYPAL)

        assert isinstance(provider, PayPalProvider)


# =============================================================================
# DunningService Tests
# =============================================================================


class TestDunningService:
    """Tests for DunningService payment failure handling."""

    @pytest.fixture
    def dunning_service(self):
        """Create DunningService instance for testing."""
        payment_service = PaymentService()
        return DunningService(payment_service=payment_service)

    @pytest.mark.asyncio
    async def test_first_retry_scheduled(self, dunning_service):
        """Verify first retry is scheduled after 1 day."""
        result = await dunning_service.handle_payment_failed(
            tenant_id="tenant-123",
            payment_id="pi_456",
            attempt_number=0,
        )

        assert result["action"] == "retry_scheduled"
        assert result["attempt"] == 1
        assert result["retry_in_days"] == 1

    @pytest.mark.asyncio
    async def test_second_retry_scheduled(self, dunning_service):
        """Verify second retry is scheduled after 3 days."""
        result = await dunning_service.handle_payment_failed(
            tenant_id="tenant-123",
            payment_id="pi_456",
            attempt_number=1,
        )

        assert result["action"] == "retry_scheduled"
        assert result["attempt"] == 2
        assert result["retry_in_days"] == 3

    @pytest.mark.asyncio
    async def test_third_retry_scheduled(self, dunning_service):
        """Verify third retry is scheduled after 7 days."""
        result = await dunning_service.handle_payment_failed(
            tenant_id="tenant-123",
            payment_id="pi_456",
            attempt_number=2,
        )

        assert result["action"] == "retry_scheduled"
        assert result["attempt"] == 3
        assert result["retry_in_days"] == 7

    @pytest.mark.asyncio
    async def test_grace_period_after_all_retries(self, dunning_service):
        """Verify grace period starts after all retries exhausted."""
        result = await dunning_service.handle_payment_failed(
            tenant_id="tenant-123",
            payment_id="pi_456",
            attempt_number=3,  # All retries exhausted
        )

        assert result["action"] == "grace_period_started"
        assert result["grace_hours"] == 48

    @pytest.mark.asyncio
    async def test_suspend_tenant(self, dunning_service):
        """Verify tenant suspension after grace period."""
        result = await dunning_service.suspend_tenant("tenant-123")

        assert result["action"] == "tenant_suspended"
        assert result["tenant_id"] == "tenant-123"
        assert result["reason"] == "payment_failure"


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for billing-related enums."""

    def test_plan_codes(self):
        """Verify all plan codes are defined."""
        assert PlanCode.FREE.value == "free"
        assert PlanCode.PRO.value == "pro"
        assert PlanCode.PRO_ANNUAL.value == "pro_annual"
        assert PlanCode.ENTERPRISE.value == "enterprise"

    def test_metric_codes(self):
        """Verify all metric codes are defined."""
        assert MetricCode.API_REQUESTS.value == "api_requests"
        assert MetricCode.AUDIO_MINUTES_INPUT.value == "audio_minutes_input"
        assert MetricCode.AUDIO_MINUTES_OUTPUT.value == "audio_minutes_output"
        assert MetricCode.LLM_TOKENS_INPUT.value == "llm_tokens_input"
        assert MetricCode.LLM_TOKENS_OUTPUT.value == "llm_tokens_output"
        assert MetricCode.CONCURRENT_CONNECTIONS.value == "concurrent_connections"
        assert MetricCode.CONNECTION_MINUTES.value == "connection_minutes"

    def test_subscription_statuses(self):
        """Verify all subscription statuses are defined."""
        assert SubscriptionStatus.ACTIVE.value == "active"
        assert SubscriptionStatus.PENDING.value == "pending"
        assert SubscriptionStatus.TERMINATED.value == "terminated"
        assert SubscriptionStatus.CANCELED.value == "canceled"

    def test_payment_providers(self):
        """Verify all payment providers are defined."""
        assert PaymentProvider.STRIPE.value == "stripe"
        assert PaymentProvider.PAYPAL.value == "paypal"

    def test_payment_statuses(self):
        """Verify all payment statuses are defined."""
        assert PaymentStatus.PENDING.value == "pending"
        assert PaymentStatus.PROCESSING.value == "processing"
        assert PaymentStatus.SUCCEEDED.value == "succeeded"
        assert PaymentStatus.FAILED.value == "failed"
        assert PaymentStatus.CANCELED.value == "canceled"
        assert PaymentStatus.REFUNDED.value == "refunded"
        assert PaymentStatus.PARTIALLY_REFUNDED.value == "partially_refunded"

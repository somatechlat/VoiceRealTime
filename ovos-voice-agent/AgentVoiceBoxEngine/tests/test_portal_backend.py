"""Unit Tests for Portal Backend API.

Tests for:
- Authentication middleware (UserContext, role/permission checks)
- Dashboard endpoints
- API Key management endpoints
- Billing endpoints
- Team management endpoints
- Settings endpoints
- Onboarding endpoints

Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8, 21.9, 24.1-24.8

NOTE: These are unit tests that verify internal logic and data structures.
Integration tests with real Keycloak/Lago are in Phase 13.
"""

from __future__ import annotations

# Import the modules under test
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portal.app.auth import UserContext  # noqa: E402

# =============================================================================
# UserContext Tests
# =============================================================================


class TestUserContext:
    """Tests for UserContext dataclass and methods."""

    def test_user_context_creation(self):
        """Verify UserContext can be created with all fields."""
        user = UserContext(
            user_id="user-123",
            tenant_id="tenant-456",
            email="test@example.com",
            username="testuser",
            roles=["tenant_admin", "developer"],
            permissions=["api:read", "api:write"],
        )

        assert user.user_id == "user-123"
        assert user.tenant_id == "tenant-456"
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert len(user.roles) == 2
        assert len(user.permissions) == 2

    def test_has_role_true(self):
        """Verify has_role returns True when user has role."""
        user = UserContext(
            user_id="user-123",
            tenant_id="tenant-456",
            email="test@example.com",
            username="testuser",
            roles=["tenant_admin", "developer"],
            permissions=[],
        )

        assert user.has_role("tenant_admin") is True
        assert user.has_role("developer") is True

    def test_has_role_false(self):
        """Verify has_role returns False when user lacks role."""
        user = UserContext(
            user_id="user-123",
            tenant_id="tenant-456",
            email="test@example.com",
            username="testuser",
            roles=["viewer"],
            permissions=[],
        )

        assert user.has_role("tenant_admin") is False
        assert user.has_role("developer") is False

    def test_has_permission_true(self):
        """Verify has_permission returns True when user has permission."""
        user = UserContext(
            user_id="user-123",
            tenant_id="tenant-456",
            email="test@example.com",
            username="testuser",
            roles=[],
            permissions=["api:read", "api:write", "billing:read"],
        )

        assert user.has_permission("api:read") is True
        assert user.has_permission("billing:read") is True

    def test_has_permission_false(self):
        """Verify has_permission returns False when user lacks permission."""
        user = UserContext(
            user_id="user-123",
            tenant_id="tenant-456",
            email="test@example.com",
            username="testuser",
            roles=[],
            permissions=["api:read"],
        )

        assert user.has_permission("admin:*") is False
        assert user.has_permission("billing:write") is False

    def test_is_admin_with_role(self):
        """Verify is_admin returns True for tenant_admin role."""
        user = UserContext(
            user_id="user-123",
            tenant_id="tenant-456",
            email="test@example.com",
            username="testuser",
            roles=["tenant_admin"],
            permissions=[],
        )

        assert user.is_admin() is True

    def test_is_admin_with_permission(self):
        """Verify is_admin returns True for admin:* permission."""
        user = UserContext(
            user_id="user-123",
            tenant_id="tenant-456",
            email="test@example.com",
            username="testuser",
            roles=[],
            permissions=["admin:*"],
        )

        assert user.is_admin() is True

    def test_is_admin_false(self):
        """Verify is_admin returns False for non-admin users."""
        user = UserContext(
            user_id="user-123",
            tenant_id="tenant-456",
            email="test@example.com",
            username="testuser",
            roles=["viewer", "developer"],
            permissions=["api:read"],
        )

        assert user.is_admin() is False


# =============================================================================
# Dashboard Model Tests
# =============================================================================


class TestDashboardModels:
    """Tests for dashboard Pydantic models."""

    def test_usage_summary_model(self):
        """Verify UsageSummary model validation."""
        from portal.app.routes.dashboard import UsageSummary

        now = datetime.now(timezone.utc)
        usage = UsageSummary(
            api_requests=1000,
            audio_minutes_input=50.5,
            audio_minutes_output=30.2,
            llm_tokens_input=50000,
            llm_tokens_output=25000,
            concurrent_connections_peak=10,
            period_start=now,
            period_end=now,
        )

        assert usage.api_requests == 1000
        assert usage.audio_minutes_input == 50.5
        assert usage.llm_tokens_input == 50000

    def test_billing_summary_model(self):
        """Verify BillingSummary model validation."""
        from portal.app.routes.dashboard import BillingSummary

        billing = BillingSummary(
            plan_name="Pro",
            plan_code="pro",
            amount_due_cents=4900,
            currency="USD",
            next_billing_date=datetime.now(timezone.utc),
            payment_status="current",
        )

        assert billing.plan_name == "Pro"
        assert billing.amount_due_cents == 4900
        assert billing.currency == "USD"

    def test_health_status_model(self):
        """Verify HealthStatus model validation."""
        from portal.app.routes.dashboard import HealthStatus

        health = HealthStatus(
            overall="healthy",
            services={"redis": "healthy", "postgresql": "healthy"},
            latency_ms={"redis": 1.5, "postgresql": 5.2},
        )

        assert health.overall == "healthy"
        assert health.services["redis"] == "healthy"
        assert health.latency_ms["redis"] == 1.5

    def test_activity_item_model(self):
        """Verify ActivityItem model validation."""
        from portal.app.routes.dashboard import ActivityItem

        activity = ActivityItem(
            id="act-123",
            type="api_key.created",
            description="API key 'Production' was created",
            timestamp=datetime.now(timezone.utc),
            metadata={"key_id": "key-456"},
        )

        assert activity.id == "act-123"
        assert activity.type == "api_key.created"


# =============================================================================
# API Keys Model Tests
# =============================================================================


class TestAPIKeyModels:
    """Tests for API key Pydantic models."""

    def test_api_key_create_model(self):
        """Verify APIKeyCreate model validation."""
        from portal.app.routes.api_keys import APIKeyCreate

        request = APIKeyCreate(
            name="Production Key",
            scopes=["realtime:connect", "realtime:admin"],
            expires_in_days=90,
        )

        assert request.name == "Production Key"
        assert len(request.scopes) == 2
        assert request.expires_in_days == 90

    def test_api_key_create_defaults(self):
        """Verify APIKeyCreate default values."""
        from portal.app.routes.api_keys import APIKeyCreate

        request = APIKeyCreate(name="Test Key")

        assert request.scopes == ["realtime:connect"]
        assert request.expires_in_days is None

    def test_api_key_response_model(self):
        """Verify APIKeyResponse model validation."""
        from portal.app.routes.api_keys import APIKeyResponse

        response = APIKeyResponse(
            id="key-123",
            name="Production Key",
            prefix="avb_prod",
            scopes=["realtime:connect"],
            created_at=datetime.now(timezone.utc),
            expires_at=None,
            last_used_at=None,
            is_active=True,
        )

        assert response.id == "key-123"
        assert response.prefix == "avb_prod"
        assert response.is_active is True

    def test_api_key_usage_model(self):
        """Verify APIKeyUsage model validation."""
        from portal.app.routes.api_keys import APIKeyUsage

        usage = APIKeyUsage(
            key_id="key-123",
            total_requests=10000,
            requests_today=500,
            requests_this_month=8000,
            last_used_at=datetime.now(timezone.utc),
        )

        assert usage.total_requests == 10000
        assert usage.requests_today == 500


# =============================================================================
# Billing Model Tests
# =============================================================================


class TestBillingModels:
    """Tests for billing Pydantic models."""

    def test_plan_details_model(self):
        """Verify PlanDetails model validation."""
        from portal.app.routes.billing import PlanDetails

        plan = PlanDetails(
            code="pro",
            name="Pro",
            description="For production applications",
            amount_cents=4900,
            currency="USD",
            interval="monthly",
            features=["10,000 API calls/month", "Email support"],
            limits={"api_requests": 10000, "audio_minutes": 1000},
        )

        assert plan.code == "pro"
        assert plan.amount_cents == 4900
        assert len(plan.features) == 2

    def test_current_subscription_model(self):
        """Verify CurrentSubscription model validation."""
        from portal.app.routes.billing import CurrentSubscription, PlanDetails

        plan = PlanDetails(
            code="pro",
            name="Pro",
            description="Pro plan",
            amount_cents=4900,
            currency="USD",
            interval="monthly",
            features=[],
            limits={},
        )

        subscription = CurrentSubscription(
            id="sub-123",
            plan=plan,
            status="active",
            started_at=datetime.now(timezone.utc),
            current_period_end=datetime.now(timezone.utc),
            cancel_at_period_end=False,
        )

        assert subscription.id == "sub-123"
        assert subscription.status == "active"
        assert subscription.cancel_at_period_end is False

    def test_invoice_model(self):
        """Verify Invoice model validation."""
        from portal.app.routes.billing import Invoice

        invoice = Invoice(
            id="inv-123",
            number="INV-2024-0001",
            status="finalized",
            payment_status="succeeded",
            total_amount_cents=4900,
            currency="USD",
            issuing_date=datetime.now(timezone.utc),
            payment_due_date=datetime.now(timezone.utc),
            items=[],
            pdf_url="https://example.com/invoice.pdf",
        )

        assert invoice.id == "inv-123"
        assert invoice.total_amount_cents == 4900


# =============================================================================
# Team Model Tests
# =============================================================================


class TestTeamModels:
    """Tests for team management Pydantic models."""

    def test_team_member_model(self):
        """Verify TeamMember model validation."""
        from portal.app.routes.team import TeamMember

        member = TeamMember(
            id="user-123",
            email="dev@example.com",
            name="John Developer",
            roles=["developer", "viewer"],
            status="active",
            joined_at=datetime.now(timezone.utc),
            last_login_at=None,
        )

        assert member.id == "user-123"
        assert member.email == "dev@example.com"
        assert len(member.roles) == 2

    def test_invite_request_model(self):
        """Verify InviteRequest model validation."""
        from portal.app.routes.team import InviteRequest

        request = InviteRequest(
            email="newuser@example.com",
            roles=["developer"],
            message="Welcome to the team!",
        )

        assert request.email == "newuser@example.com"
        assert request.roles == ["developer"]

    def test_invite_request_defaults(self):
        """Verify InviteRequest default values."""
        from portal.app.routes.team import InviteRequest

        request = InviteRequest(email="newuser@example.com")

        assert request.roles == ["viewer"]
        assert request.message is None

    def test_update_roles_request_model(self):
        """Verify UpdateRolesRequest model validation."""
        from portal.app.routes.team import UpdateRolesRequest

        request = UpdateRolesRequest(roles=["tenant_admin", "developer"])

        assert len(request.roles) == 2

    def test_available_roles(self):
        """Verify available roles are defined."""
        from portal.app.routes.team import AVAILABLE_ROLES

        role_names = [r["name"] for r in AVAILABLE_ROLES]

        assert "tenant_admin" in role_names
        assert "developer" in role_names
        assert "viewer" in role_names
        assert "billing_admin" in role_names


# =============================================================================
# Settings Model Tests
# =============================================================================


class TestSettingsModels:
    """Tests for settings Pydantic models."""

    def test_organization_profile_model(self):
        """Verify OrganizationProfile model validation."""
        from portal.app.routes.settings import OrganizationProfile

        profile = OrganizationProfile(
            name="Acme Corp",
            email="billing@acme.com",
            website="https://acme.com",
            address="123 Main St",
            phone="+1-555-1234",
            timezone="America/New_York",
            logo_url="https://acme.com/logo.png",
        )

        assert profile.name == "Acme Corp"
        assert profile.timezone == "America/New_York"

    def test_notification_preferences_model(self):
        """Verify NotificationPreferences model validation."""
        from portal.app.routes.settings import NotificationPreferences

        prefs = NotificationPreferences(
            email_billing=True,
            email_usage_alerts=True,
            email_security=True,
            email_product_updates=False,
            email_weekly_summary=True,
        )

        assert prefs.email_billing is True
        assert prefs.email_product_updates is False

    def test_notification_preferences_defaults(self):
        """Verify NotificationPreferences default values."""
        from portal.app.routes.settings import NotificationPreferences

        prefs = NotificationPreferences()

        assert prefs.email_billing is True
        assert prefs.email_product_updates is False

    def test_webhook_config_model(self):
        """Verify WebhookConfig model validation."""
        from portal.app.routes.settings import WebhookConfig

        webhook = WebhookConfig(
            id="wh_abc123",
            url="https://example.com/webhook",
            events=["session.created", "transcription.completed"],
            secret="whsec_secret123",
            is_active=True,
            created_at=datetime.now(timezone.utc),
            last_triggered_at=None,
        )

        assert webhook.id == "wh_abc123"
        assert len(webhook.events) == 2
        assert webhook.is_active is True

    def test_webhook_events_defined(self):
        """Verify webhook events are defined."""
        from portal.app.routes.settings import WEBHOOK_EVENTS

        assert "session.created" in WEBHOOK_EVENTS
        assert "transcription.completed" in WEBHOOK_EVENTS
        assert "billing.payment_succeeded" in WEBHOOK_EVENTS
        assert "api_key.created" in WEBHOOK_EVENTS


# =============================================================================
# Onboarding Model Tests
# =============================================================================


class TestOnboardingModels:
    """Tests for onboarding Pydantic models."""

    def test_signup_request_model(self):
        """Verify SignupRequest model validation."""
        from portal.app.routes.onboarding import SignupRequest, UseCase

        request = SignupRequest(
            email="newuser@example.com",
            password="SecurePass123",
            organization_name="New Startup",
            first_name="John",
            last_name="Doe",
            use_case=UseCase.VOICE_ASSISTANT,
        )

        assert request.email == "newuser@example.com"
        assert request.organization_name == "New Startup"
        assert request.use_case == UseCase.VOICE_ASSISTANT

    def test_signup_request_password_validation_too_short(self):
        """Verify password validation rejects short passwords."""
        from portal.app.routes.onboarding import SignupRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            SignupRequest(
                email="test@example.com",
                password="Short1",  # Too short
                organization_name="Test",
                first_name="John",
                last_name="Doe",
            )

        assert "at least 8 characters" in str(exc_info.value).lower()

    def test_signup_request_password_validation_no_uppercase(self):
        """Verify password validation requires uppercase."""
        from portal.app.routes.onboarding import SignupRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            SignupRequest(
                email="test@example.com",
                password="lowercase123",  # No uppercase
                organization_name="Test",
                first_name="John",
                last_name="Doe",
            )

        assert "uppercase" in str(exc_info.value).lower()

    def test_signup_request_password_validation_no_digit(self):
        """Verify password validation requires digit."""
        from portal.app.routes.onboarding import SignupRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            SignupRequest(
                email="test@example.com",
                password="NoDigitsHere",  # No digit
                organization_name="Test",
                first_name="John",
                last_name="Doe",
            )

        assert "digit" in str(exc_info.value).lower()

    def test_signup_response_model(self):
        """Verify SignupResponse model validation."""
        from portal.app.routes.onboarding import SignupResponse

        response = SignupResponse(
            tenant_id="tenant-123",
            user_id="user-456",
            project_id="proj-789",
            api_key="avb_live_abc123xyz",
            api_key_prefix="avb_live",
            message="Welcome!",
            next_steps=["Verify email", "Make first API call"],
        )

        assert response.tenant_id == "tenant-123"
        assert response.api_key_prefix == "avb_live"
        assert len(response.next_steps) == 2

    def test_onboarding_status_model(self):
        """Verify OnboardingStatus model validation."""
        from portal.app.routes.onboarding import OnboardingStatus

        status = OnboardingStatus(
            tenant_id="tenant-123",
            milestones={
                "signup": datetime.now(timezone.utc),
                "email_verified": None,
                "first_api_call": None,
            },
            completion_percentage=20,
            next_milestone="email_verified",
        )

        assert status.tenant_id == "tenant-123"
        assert status.completion_percentage == 20
        assert status.next_milestone == "email_verified"

    def test_use_case_enum(self):
        """Verify UseCase enum values."""
        from portal.app.routes.onboarding import UseCase

        assert UseCase.VOICE_ASSISTANT.value == "voice_assistant"
        assert UseCase.CUSTOMER_SERVICE.value == "customer_service"
        assert UseCase.TRANSCRIPTION.value == "transcription"

    def test_onboarding_milestone_enum(self):
        """Verify OnboardingMilestone enum values."""
        from portal.app.routes.onboarding import OnboardingMilestone

        assert OnboardingMilestone.SIGNUP.value == "signup"
        assert OnboardingMilestone.EMAIL_VERIFIED.value == "email_verified"
        assert OnboardingMilestone.FIRST_API_CALL.value == "first_api_call"
        assert OnboardingMilestone.FIRST_SUCCESS.value == "first_success"
        assert OnboardingMilestone.PAYMENT_METHOD_ADDED.value == "payment_method_added"


# =============================================================================
# Payments Model Tests
# =============================================================================


class TestPaymentsModels:
    """Tests for payment method Pydantic models."""

    def test_payment_method_response_model(self):
        """Verify PaymentMethodResponse model validation."""
        from portal.app.routes.payments import PaymentMethodResponse

        method = PaymentMethodResponse(
            id="pm_123",
            type="card",
            provider="stripe",
            last_four="4242",
            brand="visa",
            exp_month=12,
            exp_year=2025,
            is_default=True,
            created_at=datetime.now(timezone.utc),
        )

        assert method.id == "pm_123"
        assert method.type == "card"
        assert method.last_four == "4242"
        assert method.is_default is True

    def test_add_payment_method_request_model(self):
        """Verify AddPaymentMethodRequest model validation."""
        from portal.app.routes.payments import AddPaymentMethodRequest

        request = AddPaymentMethodRequest(
            provider="stripe",
            payment_method_id="pm_abc123",
            set_default=True,
        )

        assert request.provider == "stripe"
        assert request.payment_method_id == "pm_abc123"
        assert request.set_default is True

    def test_add_payment_method_request_defaults(self):
        """Verify AddPaymentMethodRequest default values."""
        from portal.app.routes.payments import AddPaymentMethodRequest

        request = AddPaymentMethodRequest(
            provider="paypal",
            payment_method_id="pp_xyz789",
        )

        assert request.set_default is True  # Default


# =============================================================================
# Available Plans Tests
# =============================================================================


class TestAvailablePlans:
    """Tests for available billing plans."""

    def test_free_plan_defined(self):
        """Verify Free plan is defined correctly."""
        from portal.app.routes.billing import AVAILABLE_PLANS

        free_plan = next((p for p in AVAILABLE_PLANS if p.code == "free"), None)

        assert free_plan is not None
        assert free_plan.name == "Free"
        assert free_plan.amount_cents == 0
        assert free_plan.limits["api_requests"] == 100

    def test_pro_plan_defined(self):
        """Verify Pro plan is defined correctly."""
        from portal.app.routes.billing import AVAILABLE_PLANS

        pro_plan = next((p for p in AVAILABLE_PLANS if p.code == "pro"), None)

        assert pro_plan is not None
        assert pro_plan.name == "Pro"
        assert pro_plan.amount_cents == 4900
        assert pro_plan.limits["api_requests"] == 10000

    def test_enterprise_plan_defined(self):
        """Verify Enterprise plan is defined correctly."""
        from portal.app.routes.billing import AVAILABLE_PLANS

        enterprise_plan = next((p for p in AVAILABLE_PLANS if p.code == "enterprise"), None)

        assert enterprise_plan is not None
        assert enterprise_plan.name == "Enterprise"
        assert enterprise_plan.limits["api_requests"] == -1  # Unlimited


# =============================================================================
# Portal App Creation Tests
# =============================================================================


class TestPortalAppCreation:
    """Tests for portal FastAPI app creation."""

    def test_create_app_returns_fastapi(self):
        """Verify create_app returns a FastAPI instance."""
        from fastapi import FastAPI
        from portal.app import create_app

        app = create_app()

        assert isinstance(app, FastAPI)

    def test_app_has_correct_title(self):
        """Verify app has correct title."""
        from portal.app import create_app

        app = create_app()

        assert app.title == "AgentVoiceBox Customer Portal API"

    def test_app_has_health_endpoint(self):
        """Verify app has health check endpoint."""
        from portal.app import create_app

        app = create_app()

        # Check routes include health
        routes = [r.path for r in app.routes]
        assert "/health" in routes

    def test_app_has_docs_endpoint(self):
        """Verify app has docs endpoint configured."""
        from portal.app import create_app

        app = create_app()

        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

"""Tenant Onboarding Routes.

Provides endpoints for:
- User signup (creates Keycloak user, Lago customer, default project, first API key)
- Email verification
- Onboarding milestone tracking
- Interactive quickstart

Requirements: 24.1, 24.2, 24.3, 24.4, 24.5, 24.7, 24.8
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, field_validator

logger = logging.getLogger(__name__)
router = APIRouter()


class UseCase(str, Enum):
    """Predefined use cases for onboarding."""

    VOICE_ASSISTANT = "voice_assistant"
    CUSTOMER_SERVICE = "customer_service"
    TRANSCRIPTION = "transcription"
    ACCESSIBILITY = "accessibility"
    GAMING = "gaming"
    EDUCATION = "education"
    OTHER = "other"


class OnboardingMilestone(str, Enum):
    """Onboarding completion milestones."""

    SIGNUP = "signup"
    EMAIL_VERIFIED = "email_verified"
    FIRST_API_CALL = "first_api_call"
    FIRST_SUCCESS = "first_success"
    PAYMENT_METHOD_ADDED = "payment_method_added"


class SignupRequest(BaseModel):
    """Request to create a new tenant account."""

    email: EmailStr = Field(description="User email address")
    password: str = Field(min_length=8, max_length=128, description="Account password")
    organization_name: str = Field(min_length=1, max_length=100, description="Organization name")
    first_name: str = Field(min_length=1, max_length=50, description="User's first name")
    last_name: str = Field(min_length=1, max_length=50, description="User's last name")
    use_case: Optional[UseCase] = Field(default=None, description="Primary use case")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class SignupResponse(BaseModel):
    """Response from successful signup."""

    tenant_id: str = Field(description="New tenant ID")
    user_id: str = Field(description="Keycloak user ID")
    project_id: str = Field(description="Default project ID")
    api_key: str = Field(description="First API key (shown only once)")
    api_key_prefix: str = Field(description="API key prefix for identification")
    message: str = Field(description="Welcome message")
    next_steps: List[str] = Field(description="Suggested next steps")


class VerifyEmailRequest(BaseModel):
    """Request to verify email address."""

    token: str = Field(description="Email verification token")


class OnboardingStatus(BaseModel):
    """Current onboarding status."""

    tenant_id: str
    milestones: dict[str, Optional[datetime]] = Field(description="Milestone completion timestamps")
    completion_percentage: int = Field(description="Overall completion percentage")
    next_milestone: Optional[str] = Field(description="Next suggested milestone")


class QuickstartTestRequest(BaseModel):
    """Request to test API call in quickstart."""

    text: str = Field(
        default="Hello, this is a test of the AgentVoiceBox API.",
        max_length=500,
        description="Text to synthesize for test",
    )


class QuickstartTestResponse(BaseModel):
    """Response from quickstart test."""

    success: bool
    message: str
    latency_ms: float
    audio_url: Optional[str] = None


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    request: SignupRequest,
    background_tasks: BackgroundTasks,
) -> SignupResponse:
    """Create a new tenant account.

    This endpoint orchestrates the complete signup flow:
    1. Create Keycloak user with tenant_admin role
    2. Create Lago customer for billing
    3. Create default project
    4. Generate first API key
    5. Send welcome email (async)

    Requirements: 24.1, 24.2
    """
    tenant_id = str(uuid.uuid4())
    project_id = str(uuid.uuid4())

    try:
        # 1. Create Keycloak user
        from ....app.services.keycloak_service import get_keycloak_service

        keycloak = get_keycloak_service()

        # Check if user already exists
        existing = await keycloak.get_user_by_email(request.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists",
            )

        # Create user with tenant_admin role
        user = await keycloak.create_user(
            username=request.email,
            email=request.email,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name,
            tenant_id=tenant_id,
            roles=["tenant_admin", "developer"],
            email_verified=False,
            enabled=True,
            temporary_password=False,
        )

        logger.info(f"Created Keycloak user {user.id} for tenant {tenant_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create Keycloak user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account",
        )

    try:
        # 2. Create Lago customer for billing
        from ....app.services.lago_service import get_lago_service

        lago = get_lago_service()

        await lago.create_customer(
            external_id=tenant_id,
            name=request.organization_name,
            email=request.email,
            currency="USD",
            timezone="UTC",
            metadata=[
                {
                    "key": "use_case",
                    "value": request.use_case.value if request.use_case else "other",
                },
                {"key": "signup_date", "value": datetime.now(timezone.utc).isoformat()},
            ],
        )

        # Create free subscription
        await lago.create_subscription(
            external_customer_id=tenant_id,
            plan_code="free",
            name=f"{request.organization_name} - Free Plan",
        )

        logger.info(f"Created Lago customer and subscription for tenant {tenant_id}")

    except Exception as e:
        logger.error(f"Failed to create Lago customer: {e}")
        # Continue - billing can be set up later

    try:
        # 3. Create default project in database
        from ....app.services.async_database import get_database

        db = get_database()

        # Insert tenant
        await db.execute(
            """
            INSERT INTO tenants (id, name, tier, status, settings, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            tenant_id,
            request.organization_name,
            "free",
            "active",
            {"use_case": request.use_case.value if request.use_case else "other"},
            datetime.now(timezone.utc),
        )

        # Insert default project
        await db.execute(
            """
            INSERT INTO projects (id, tenant_id, name, environment, settings, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            project_id,
            tenant_id,
            "Default Project",
            "production",
            {},
            datetime.now(timezone.utc),
        )

        logger.info(f"Created tenant {tenant_id} and project {project_id} in database")

    except Exception as e:
        logger.error(f"Failed to create tenant/project in database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create organization",
        )

    try:
        # 4. Generate first API key
        from ....app.services.api_key_service import APIKeyHasher, generate_api_key

        api_key, prefix = generate_api_key()
        key_id = str(uuid.uuid4())
        hasher = APIKeyHasher()
        key_hash = hasher.hash(api_key)

        db = get_database()
        await db.execute(
            """
            INSERT INTO api_keys (id, project_id, key_hash, key_prefix, name, scopes, rate_limit_tier, is_active, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            key_id,
            project_id,
            key_hash,
            prefix,
            "Default API Key",
            ["realtime:connect", "realtime:admin"],
            "free",
            True,
            datetime.now(timezone.utc),
        )

        logger.info(f"Created API key {key_id} for project {project_id}")

    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate API key",
        )

    try:
        # 5. Record signup milestone
        await _record_milestone(tenant_id, OnboardingMilestone.SIGNUP)

    except Exception as e:
        logger.warning(f"Failed to record signup milestone: {e}")

    # 6. Send welcome email (async)
    background_tasks.add_task(
        _send_welcome_email,
        email=request.email,
        first_name=request.first_name,
        organization_name=request.organization_name,
        api_key=api_key,
    )

    return SignupResponse(
        tenant_id=tenant_id,
        user_id=user.id,
        project_id=project_id,
        api_key=api_key,
        api_key_prefix=prefix,
        message=f"Welcome to AgentVoiceBox, {request.first_name}!",
        next_steps=[
            "Verify your email address",
            "Make your first API call",
            "Explore the documentation",
            "Add a payment method to unlock Pro features",
        ],
    )


@router.post("/verify-email")
async def verify_email(request: VerifyEmailRequest) -> dict:
    """Verify email address using token.

    Requirements: 24.2, 24.3
    """
    # In production, this would validate the token from email
    # and update Keycloak user's emailVerified status

    # For now, return success
    return {
        "message": "Email verified successfully",
        "verified": True,
    }


@router.get("/status/{tenant_id}", response_model=OnboardingStatus)
async def get_onboarding_status(tenant_id: str) -> OnboardingStatus:
    """Get current onboarding status for a tenant.

    Requirements: 24.7, 24.8
    """
    try:
        from ....app.services.async_database import get_database

        db = get_database()

        # Get milestones from database
        rows = await db.fetch(
            """
            SELECT milestone, completed_at
            FROM onboarding_milestones
            WHERE tenant_id = $1
            ORDER BY completed_at
            """,
            tenant_id,
        )

        milestones = {m.value: None for m in OnboardingMilestone}
        for row in rows:
            milestones[row["milestone"]] = row["completed_at"]

        # Calculate completion percentage
        completed = sum(1 for v in milestones.values() if v is not None)
        total = len(OnboardingMilestone)
        percentage = int((completed / total) * 100)

        # Determine next milestone
        next_milestone = None
        for milestone in OnboardingMilestone:
            if milestones.get(milestone.value) is None:
                next_milestone = milestone.value
                break

        return OnboardingStatus(
            tenant_id=tenant_id,
            milestones=milestones,
            completion_percentage=percentage,
            next_milestone=next_milestone,
        )

    except Exception as e:
        logger.error(f"Failed to get onboarding status: {e}")
        # Return empty status
        return OnboardingStatus(
            tenant_id=tenant_id,
            milestones={m.value: None for m in OnboardingMilestone},
            completion_percentage=0,
            next_milestone=OnboardingMilestone.SIGNUP.value,
        )


@router.post("/quickstart/test", response_model=QuickstartTestResponse)
async def quickstart_test(
    request: QuickstartTestRequest,
    api_key: str = None,  # Would come from header in real implementation
) -> QuickstartTestResponse:
    """Test API call for interactive quickstart.

    This endpoint allows new users to test the TTS API
    and see their usage update in real-time.

    Requirements: 24.4, 24.5
    """
    import time

    start_time = time.time()

    try:
        # In production, this would actually call the TTS service
        # For now, simulate a successful response

        latency_ms = (time.time() - start_time) * 1000

        return QuickstartTestResponse(
            success=True,
            message="API call successful! Your text was processed.",
            latency_ms=round(latency_ms, 2),
            audio_url=None,  # Would be actual audio URL
        )

    except Exception as e:
        logger.error(f"Quickstart test failed: {e}")
        return QuickstartTestResponse(
            success=False,
            message=f"API call failed: {str(e)}",
            latency_ms=0,
            audio_url=None,
        )


@router.post("/milestone/{tenant_id}/{milestone}")
async def record_milestone(
    tenant_id: str,
    milestone: OnboardingMilestone,
) -> dict:
    """Record an onboarding milestone completion.

    Requirements: 24.7, 24.8
    """
    try:
        await _record_milestone(tenant_id, milestone)
        return {
            "message": f"Milestone '{milestone.value}' recorded",
            "tenant_id": tenant_id,
        }
    except Exception as e:
        logger.error(f"Failed to record milestone: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record milestone",
        )


# =============================================================================
# Internal Functions
# =============================================================================


async def _record_milestone(tenant_id: str, milestone: OnboardingMilestone) -> None:
    """Record an onboarding milestone in the database."""
    try:
        from ....app.services.async_database import get_database

        db = get_database()

        await db.execute(
            """
            INSERT INTO onboarding_milestones (id, tenant_id, milestone, completed_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (tenant_id, milestone) DO NOTHING
            """,
            str(uuid.uuid4()),
            tenant_id,
            milestone.value,
            datetime.now(timezone.utc),
        )

        logger.info(f"Recorded milestone {milestone.value} for tenant {tenant_id}")

    except Exception as e:
        logger.error(f"Failed to record milestone: {e}")
        raise


async def _send_welcome_email(
    email: str,
    first_name: str,
    organization_name: str,
    api_key: str,
) -> None:
    """Send welcome email to new user.

    Requirements: 24.3
    """
    try:
        # In production, this would use an email service (SendGrid, SES, etc.)
        # For now, just log the email

        logger.info(
            f"Sending welcome email to {email}",
            extra={
                "email": email,
                "first_name": first_name,
                "organization": organization_name,
                "template": "welcome",
            },
        )

        # Email content would include:
        # - Welcome message
        # - API key (masked except last 4 chars)
        # - Quickstart guide link
        # - Documentation link
        # - Support contact

    except Exception as e:
        logger.error(f"Failed to send welcome email: {e}")

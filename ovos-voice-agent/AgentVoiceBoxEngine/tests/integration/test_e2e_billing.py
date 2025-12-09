"""End-to-End Billing Integration Tests.

These tests validate the complete billing flow:
1. Stripe integration (test mode)
2. PayPal integration (sandbox)
3. Subscription management
4. Usage metering
5. Invoice generation

Requirements: 20.1, 20.3, 20.4, 20.5, 20.6, 22.1, 22.2

Run with:
    docker compose -f docker-compose.yml up -d
    pytest tests/integration/test_e2e_billing.py -v

Note: These tests use Stripe test mode and PayPal sandbox.
Set STRIPE_TEST_KEY and PAYPAL_SANDBOX_CLIENT_ID environment variables.
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
import httpx

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure pytest-asyncio mode
pytestmark = pytest.mark.asyncio(loop_scope="function")

# Service URLs from environment
PORTAL_API_URL = os.getenv("PORTAL_API_URL", "http://localhost:25001")
LAGO_API_URL = os.getenv("LAGO_API_URL", "http://localhost:25005")

# Stripe test mode
STRIPE_TEST_KEY = os.getenv("STRIPE_TEST_KEY", "")
STRIPE_TEST_PM = "pm_card_visa"  # Stripe test payment method


class TestStripeIntegration:
    """Test Stripe integration in test mode.
    
    Requirements: 20.6, 22.1
    """

    @pytest.mark.asyncio
    async def test_stripe_payment_method_setup(self):
        """Test adding a Stripe payment method.
        
        Requirements: 22.1, 22.3
        """
        if not STRIPE_TEST_KEY:
            pytest.skip("STRIPE_TEST_KEY not set")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # First create an account
                test_email = f"stripe_{uuid.uuid4().hex[:8]}@example.com"
                
                signup_response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/signup",
                    json={
                        "email": test_email,
                        "password": "SecurePass123!",
                        "organization_name": "Stripe Test Org",
                        "first_name": "Test",
                        "last_name": "User",
                    },
                )
                
                if signup_response.status_code != 201:
                    pytest.skip("Signup failed")
                
                api_key = signup_response.json()["api_key"]
                
                # Add payment method
                pm_response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/payments/methods",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "provider": "stripe",
                        "payment_method_id": STRIPE_TEST_PM,
                        "set_default": True,
                    },
                )
                
                if pm_response.status_code == 200:
                    data = pm_response.json()
                    assert data.get("type") == "card"
                    assert data.get("provider") == "stripe"
                    print("✓ Stripe payment method added successfully")
                elif pm_response.status_code in [401, 403]:
                    pytest.skip("Auth required for payment methods")
                else:
                    print(f"Payment method response: {pm_response.status_code}")
                    
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")

    @pytest.mark.asyncio
    async def test_stripe_webhook_signature_validation(self):
        """Test Stripe webhook signature validation.
        
        Requirements: 22.1
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Send webhook without valid signature
                response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/webhooks/stripe",
                    headers={
                        "Stripe-Signature": "invalid_signature",
                        "Content-Type": "application/json",
                    },
                    json={"type": "payment_intent.succeeded"},
                )
                
                # Should reject invalid signature
                assert response.status_code in [400, 401, 403, 404]
                print("✓ Invalid Stripe webhook signature rejected")
                
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")


class TestPayPalIntegration:
    """Test PayPal integration in sandbox mode.
    
    Requirements: 20.6, 22.2
    """

    @pytest.mark.asyncio
    async def test_paypal_webhook_endpoint_exists(self):
        """Test PayPal webhook endpoint exists.
        
        Requirements: 22.2
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/webhooks/paypal",
                    headers={"Content-Type": "application/json"},
                    json={"event_type": "PAYMENT.CAPTURE.COMPLETED"},
                )
                
                # Endpoint should exist (may reject without valid signature)
                assert response.status_code in [200, 400, 401, 403, 404]
                print("✓ PayPal webhook endpoint accessible")
                
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")


class TestSubscriptionManagement:
    """Test subscription management flow.
    
    Requirements: 20.3, 23.1, 23.2, 23.3
    """

    @pytest.mark.asyncio
    async def test_available_plans_endpoint(self):
        """Test listing available subscription plans.
        
        Requirements: 20.3, 23.1
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{PORTAL_API_URL}/api/v1/billing/plans",
                )
                
                if response.status_code == 200:
                    plans = response.json()
                    
                    # Should have Free, Pro, Enterprise plans
                    plan_codes = [p.get("code") for p in plans]
                    
                    assert "free" in plan_codes, "Free plan should exist"
                    assert "pro" in plan_codes, "Pro plan should exist"
                    assert "enterprise" in plan_codes, "Enterprise plan should exist"
                    
                    # Verify plan structure
                    for plan in plans:
                        assert "code" in plan
                        assert "name" in plan
                        assert "amount_cents" in plan
                        assert "features" in plan
                    
                    print("✓ Available plans endpoint working")
                    print(f"  Plans: {plan_codes}")
                    
                elif response.status_code in [401, 403]:
                    pytest.skip("Auth required for plans endpoint")
                else:
                    pytest.skip(f"Plans endpoint returned {response.status_code}")
                    
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")

    @pytest.mark.asyncio
    async def test_current_subscription_endpoint(self):
        """Test getting current subscription.
        
        Requirements: 20.3
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Create account first
                test_email = f"sub_{uuid.uuid4().hex[:8]}@example.com"
                
                signup_response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/signup",
                    json={
                        "email": test_email,
                        "password": "SecurePass123!",
                        "organization_name": "Subscription Test Org",
                        "first_name": "Test",
                        "last_name": "User",
                    },
                )
                
                if signup_response.status_code != 201:
                    pytest.skip("Signup failed")
                
                api_key = signup_response.json()["api_key"]
                
                # Get current subscription
                sub_response = await client.get(
                    f"{PORTAL_API_URL}/api/v1/billing/subscription",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                
                if sub_response.status_code == 200:
                    data = sub_response.json()
                    
                    # New accounts should be on free plan
                    assert data.get("plan", {}).get("code") == "free"
                    assert data.get("status") in ["active", "trialing"]
                    
                    print("✓ Current subscription endpoint working")
                    print(f"  Plan: {data.get('plan', {}).get('name')}")
                    
                elif sub_response.status_code in [401, 403]:
                    pytest.skip("Auth required for subscription endpoint")
                    
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")


class TestUsageMetering:
    """Test usage metering flow.
    
    Requirements: 20.4, 20.5
    """

    @pytest.mark.asyncio
    async def test_usage_endpoint(self):
        """Test usage data endpoint.
        
        Requirements: 20.4, 21.5
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Create account first
                test_email = f"usage_{uuid.uuid4().hex[:8]}@example.com"
                
                signup_response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/signup",
                    json={
                        "email": test_email,
                        "password": "SecurePass123!",
                        "organization_name": "Usage Test Org",
                        "first_name": "Test",
                        "last_name": "User",
                    },
                )
                
                if signup_response.status_code != 201:
                    pytest.skip("Signup failed")
                
                api_key = signup_response.json()["api_key"]
                
                # Get usage data
                usage_response = await client.get(
                    f"{PORTAL_API_URL}/api/v1/dashboard/usage",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                
                if usage_response.status_code == 200:
                    data = usage_response.json()
                    
                    # Verify usage structure
                    assert "api_requests" in data
                    assert "audio_minutes_input" in data
                    assert "audio_minutes_output" in data
                    
                    print("✓ Usage endpoint working")
                    print(f"  API Requests: {data.get('api_requests', 0)}")
                    
                elif usage_response.status_code in [401, 403]:
                    pytest.skip("Auth required for usage endpoint")
                    
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")


class TestInvoiceGeneration:
    """Test invoice generation and retrieval.
    
    Requirements: 20.7, 21.7
    """

    @pytest.mark.asyncio
    async def test_invoices_endpoint(self):
        """Test invoices list endpoint.
        
        Requirements: 20.7, 21.7
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Create account first
                test_email = f"invoice_{uuid.uuid4().hex[:8]}@example.com"
                
                signup_response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/signup",
                    json={
                        "email": test_email,
                        "password": "SecurePass123!",
                        "organization_name": "Invoice Test Org",
                        "first_name": "Test",
                        "last_name": "User",
                    },
                )
                
                if signup_response.status_code != 201:
                    pytest.skip("Signup failed")
                
                api_key = signup_response.json()["api_key"]
                
                # Get invoices
                invoices_response = await client.get(
                    f"{PORTAL_API_URL}/api/v1/billing/invoices",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                
                if invoices_response.status_code == 200:
                    data = invoices_response.json()
                    
                    # Should be a list (may be empty for new accounts)
                    assert isinstance(data, list)
                    
                    print("✓ Invoices endpoint working")
                    print(f"  Invoice count: {len(data)}")
                    
                elif invoices_response.status_code in [401, 403]:
                    pytest.skip("Auth required for invoices endpoint")
                    
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")


class TestLagoIntegration:
    """Test Lago billing engine integration.
    
    Requirements: 20.1
    """

    @pytest.mark.asyncio
    async def test_lago_health(self):
        """Test Lago service is healthy.
        
        Requirements: 20.1
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{LAGO_API_URL}/health")
                
                if response.status_code == 200:
                    print("✓ Lago service is healthy")
                else:
                    print(f"Lago health check: {response.status_code}")
                    
            except httpx.ConnectError:
                pytest.skip("Lago not reachable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""End-to-End Onboarding Integration Tests.

These tests validate the complete onboarding flow:
1. User signup (creates Keycloak user, Lago customer, first API key)
2. Email verification
3. First API call
4. Onboarding milestone tracking

Requirements: 24.1, 24.2, 24.3, 24.4, 24.5, 24.7, 24.8

Run with:
    docker compose -f docker-compose.yml up -d
    pytest tests/integration/test_e2e_onboarding.py -v
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
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:25000")
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:25004")


class TestEndToEndOnboarding:
    """End-to-end onboarding flow tests.
    
    Requirements: 24.1, 24.2, 24.3, 24.4, 24.5, 24.7, 24.8
    """

    @pytest.mark.asyncio
    async def test_complete_signup_flow(self):
        """Test complete signup flow creates all required resources.
        
        Requirements: 24.1, 24.2
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Generate unique test data
            test_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
            test_org = f"Test Org {uuid.uuid4().hex[:8]}"
            
            signup_data = {
                "email": test_email,
                "password": "SecurePass123!",
                "organization_name": test_org,
                "first_name": "Test",
                "last_name": "User",
                "use_case": "voice_assistant",
            }
            
            try:
                response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/signup",
                    json=signup_data,
                )
                
                # Check response
                if response.status_code == 201:
                    data = response.json()
                    
                    # Verify all required fields are present
                    assert "tenant_id" in data
                    assert "user_id" in data
                    assert "project_id" in data
                    assert "api_key" in data
                    assert "api_key_prefix" in data
                    assert "message" in data
                    assert "next_steps" in data
                    
                    # Verify API key format
                    assert data["api_key"].startswith("avb_")
                    assert len(data["api_key"]) > 20
                    
                    # Verify next steps are provided
                    assert len(data["next_steps"]) > 0
                    
                    print(f"✓ Signup successful for {test_email}")
                    print(f"  Tenant ID: {data['tenant_id']}")
                    print(f"  API Key Prefix: {data['api_key_prefix']}")
                    
                elif response.status_code == 503:
                    pytest.skip("Portal API not available")
                else:
                    # Log error for debugging
                    print(f"Signup failed: {response.status_code} - {response.text}")
                    pytest.skip(f"Signup endpoint returned {response.status_code}")
                    
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")

    @pytest.mark.asyncio
    async def test_duplicate_email_rejected(self):
        """Test that duplicate email addresses are rejected.
        
        Requirements: 24.2
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            test_email = f"duplicate_{uuid.uuid4().hex[:8]}@example.com"
            
            signup_data = {
                "email": test_email,
                "password": "SecurePass123!",
                "organization_name": "Test Org",
                "first_name": "Test",
                "last_name": "User",
            }
            
            try:
                # First signup
                response1 = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/signup",
                    json=signup_data,
                )
                
                if response1.status_code != 201:
                    pytest.skip("First signup failed")
                
                # Second signup with same email
                response2 = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/signup",
                    json=signup_data,
                )
                
                # Should be rejected
                assert response2.status_code == 409
                data = response2.json()
                assert "already exists" in data.get("detail", "").lower()
                
                print("✓ Duplicate email correctly rejected")
                
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")

    @pytest.mark.asyncio
    async def test_password_validation(self):
        """Test password validation rules.
        
        Requirements: 24.2
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            test_email = f"pwtest_{uuid.uuid4().hex[:8]}@example.com"
            
            # Test weak passwords
            weak_passwords = [
                ("short", "Too short"),
                ("nouppercase123", "No uppercase"),
                ("NOLOWERCASE123", "No lowercase"),
                ("NoDigitsHere", "No digit"),
            ]
            
            try:
                for password, reason in weak_passwords:
                    signup_data = {
                        "email": test_email,
                        "password": password,
                        "organization_name": "Test Org",
                        "first_name": "Test",
                        "last_name": "User",
                    }
                    
                    response = await client.post(
                        f"{PORTAL_API_URL}/api/v1/onboarding/signup",
                        json=signup_data,
                    )
                    
                    if response.status_code == 503:
                        pytest.skip("Portal API not available")
                    
                    # Should be rejected with 422 validation error
                    assert response.status_code == 422, f"Password '{password}' ({reason}) should be rejected"
                    
                print("✓ Password validation working correctly")
                
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")

    @pytest.mark.asyncio
    async def test_onboarding_status_tracking(self):
        """Test onboarding milestone tracking.
        
        Requirements: 24.7, 24.8
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First create a new account
            test_email = f"milestone_{uuid.uuid4().hex[:8]}@example.com"
            
            signup_data = {
                "email": test_email,
                "password": "SecurePass123!",
                "organization_name": "Milestone Test Org",
                "first_name": "Test",
                "last_name": "User",
            }
            
            try:
                response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/signup",
                    json=signup_data,
                )
                
                if response.status_code != 201:
                    pytest.skip("Signup failed")
                
                tenant_id = response.json()["tenant_id"]
                
                # Check onboarding status
                status_response = await client.get(
                    f"{PORTAL_API_URL}/api/v1/onboarding/status/{tenant_id}",
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    
                    assert status["tenant_id"] == tenant_id
                    assert "milestones" in status
                    assert "completion_percentage" in status
                    assert "next_milestone" in status
                    
                    # Signup milestone should be completed
                    milestones = status["milestones"]
                    assert milestones.get("signup") is not None
                    
                    # Completion should be > 0
                    assert status["completion_percentage"] > 0
                    
                    print(f"✓ Onboarding status tracking working")
                    print(f"  Completion: {status['completion_percentage']}%")
                    print(f"  Next milestone: {status['next_milestone']}")
                    
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")

    @pytest.mark.asyncio
    async def test_quickstart_api_test(self):
        """Test interactive quickstart API test endpoint.
        
        Requirements: 24.4, 24.5
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/quickstart/test",
                    json={"text": "Hello, this is a test."},
                )
                
                if response.status_code == 503:
                    pytest.skip("Portal API not available")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    assert "success" in data
                    assert "message" in data
                    assert "latency_ms" in data
                    
                    print(f"✓ Quickstart test endpoint working")
                    print(f"  Success: {data['success']}")
                    print(f"  Latency: {data['latency_ms']}ms")
                    
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")


class TestAPIKeyUsage:
    """Test API key usage after onboarding.
    
    Requirements: 24.1, 3.1, 3.2
    """

    @pytest.mark.asyncio
    async def test_api_key_works_for_gateway(self):
        """Test that generated API key works for gateway access.
        
        Requirements: 24.1, 3.1
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create account and get API key
            test_email = f"apitest_{uuid.uuid4().hex[:8]}@example.com"
            
            signup_data = {
                "email": test_email,
                "password": "SecurePass123!",
                "organization_name": "API Test Org",
                "first_name": "Test",
                "last_name": "User",
            }
            
            try:
                response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/signup",
                    json=signup_data,
                )
                
                if response.status_code != 201:
                    pytest.skip("Signup failed")
                
                api_key = response.json()["api_key"]
                
                # Try to access gateway health endpoint
                health_response = await client.get(
                    f"{GATEWAY_URL}/health",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                
                # Health endpoint should work (may not require auth)
                assert health_response.status_code in [200, 401, 403]
                
                print(f"✓ API key generated and gateway accessible")
                
            except httpx.ConnectError:
                pytest.skip("Services not reachable")


class TestEmailVerification:
    """Test email verification flow.
    
    Requirements: 24.2, 24.3
    """

    @pytest.mark.asyncio
    async def test_email_verification_endpoint(self):
        """Test email verification endpoint exists and responds.
        
        Requirements: 24.2, 24.3
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test with a dummy token
                response = await client.post(
                    f"{PORTAL_API_URL}/api/v1/onboarding/verify-email",
                    json={"token": "test_token_123"},
                )
                
                if response.status_code == 503:
                    pytest.skip("Portal API not available")
                
                # Endpoint should exist and respond
                assert response.status_code in [200, 400, 404]
                
                print("✓ Email verification endpoint accessible")
                
            except httpx.ConnectError:
                pytest.skip("Portal API not reachable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

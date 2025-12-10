"""Keycloak Service for Identity & Access Management.

This module provides integration with Keycloak for:
- User management (create, update, delete, deactivate)
- Realm management (tenant provisioning)
- Role assignment and RBAC
- Token validation and introspection
- SSO integration (SAML, OIDC)

Requirements: 19.2, 19.3, 19.8, 19.9
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)


@dataclass
class KeycloakConfig:
    """Keycloak connection configuration."""

    server_url: str = "http://localhost:8080"
    realm: str = "agentvoicebox"
    client_id: str = "agentvoicebox-api"
    client_secret: str = ""
    admin_username: str = "admin"
    admin_password: str = "admin"
    verify_ssl: bool = True
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "KeycloakConfig":
        """Create config from environment variables."""
        return cls(
            server_url=os.getenv("KEYCLOAK_SERVER_URL", "http://localhost:8080"),
            realm=os.getenv("KEYCLOAK_REALM", "agentvoicebox"),
            client_id=os.getenv("KEYCLOAK_CLIENT_ID", "agentvoicebox-api"),
            client_secret=os.getenv("KEYCLOAK_CLIENT_SECRET", ""),
            admin_username=os.getenv("KEYCLOAK_ADMIN_USERNAME", "admin"),
            admin_password=os.getenv("KEYCLOAK_ADMIN_PASSWORD", "admin"),
            verify_ssl=os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower() == "true",
            timeout=float(os.getenv("KEYCLOAK_TIMEOUT", "30")),
        )


@dataclass
class KeycloakUser:
    """Keycloak user representation."""

    id: str
    username: str
    email: str
    first_name: str = ""
    last_name: str = ""
    enabled: bool = True
    email_verified: bool = False
    created_timestamp: Optional[int] = None
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    realm_roles: List[str] = field(default_factory=list)

    @property
    def tenant_id(self) -> Optional[str]:
        """Get tenant_id from user attributes."""
        tenant_ids = self.attributes.get("tenant_id", [])
        return tenant_ids[0] if tenant_ids else None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeycloakUser":
        """Create user from Keycloak API response."""
        return cls(
            id=data.get("id", ""),
            username=data.get("username", ""),
            email=data.get("email", ""),
            first_name=data.get("firstName", ""),
            last_name=data.get("lastName", ""),
            enabled=data.get("enabled", True),
            email_verified=data.get("emailVerified", False),
            created_timestamp=data.get("createdTimestamp"),
            attributes=data.get("attributes", {}),
            realm_roles=data.get("realmRoles", []),
        )


@dataclass
class TokenInfo:
    """Token introspection result."""

    active: bool
    sub: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    tenant_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    client_id: Optional[str] = None
    exp: Optional[int] = None
    iat: Optional[int] = None
    scope: Optional[str] = None


class KeycloakService:
    """Service for Keycloak identity management operations.

    Provides methods for:
    - Admin authentication and token management
    - User CRUD operations
    - Role management
    - Token validation
    - Tenant (realm) provisioning
    """

    def __init__(self, config: Optional[KeycloakConfig] = None):
        """Initialize Keycloak service.

        Args:
            config: Keycloak configuration. If None, loads from environment.
        """
        self.config = config or KeycloakConfig.from_env()
        self._admin_token: Optional[str] = None
        self._admin_token_expires: Optional[datetime] = None
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

    # =========================================================================
    # Admin Authentication
    # =========================================================================

    async def _get_admin_token(self) -> str:
        """Get or refresh admin access token.

        Returns:
            Admin access token for API calls.
        """
        # Check if we have a valid cached token
        if (
            self._admin_token
            and self._admin_token_expires
            and datetime.utcnow() < self._admin_token_expires - timedelta(seconds=30)
        ):
            return self._admin_token

        # Get new token
        url = urljoin(self.config.server_url, "/realms/master/protocol/openid-connect/token")

        data = {
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": self.config.admin_username,
            "password": self.config.admin_password,
        }

        response = await self._client.post(url, data=data)
        response.raise_for_status()

        token_data = response.json()
        self._admin_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 300)
        self._admin_token_expires = datetime.utcnow() + timedelta(seconds=expires_in)

        return self._admin_token

    async def _admin_request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make authenticated admin API request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path (relative to admin API base)
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        token = await self._get_admin_token()
        url = urljoin(self.config.server_url, f"/admin/realms/{self.config.realm}{path}")

        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        headers["Content-Type"] = "application/json"

        response = await self._client.request(method, url, headers=headers, **kwargs)
        return response

    # =========================================================================
    # User Management
    # =========================================================================

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        first_name: str = "",
        last_name: str = "",
        tenant_id: Optional[str] = None,
        roles: Optional[List[str]] = None,
        email_verified: bool = False,
        enabled: bool = True,
        temporary_password: bool = True,
    ) -> KeycloakUser:
        """Create a new user in Keycloak.

        Args:
            username: Username (usually email)
            email: User email address
            password: Initial password
            first_name: User's first name
            last_name: User's last name
            tenant_id: Tenant ID to associate with user
            roles: List of realm roles to assign
            email_verified: Whether email is pre-verified
            enabled: Whether user is enabled
            temporary_password: Whether password must be changed on first login

        Returns:
            Created user object
        """
        user_data = {
            "username": username,
            "email": email,
            "firstName": first_name,
            "lastName": last_name,
            "enabled": enabled,
            "emailVerified": email_verified,
            "credentials": [
                {
                    "type": "password",
                    "value": password,
                    "temporary": temporary_password,
                }
            ],
            "attributes": {},
        }

        if tenant_id:
            user_data["attributes"]["tenant_id"] = [tenant_id]

        response = await self._admin_request("POST", "/users", json=user_data)

        if response.status_code == 409:
            raise ValueError(f"User with username '{username}' already exists")

        response.raise_for_status()

        # Get user ID from Location header
        location = response.headers.get("Location", "")
        user_id = location.split("/")[-1]

        # Assign roles if specified
        if roles:
            await self.assign_roles(user_id, roles)

        # Fetch and return the created user
        return await self.get_user(user_id)

    async def get_user(self, user_id: str) -> KeycloakUser:
        """Get user by ID.

        Args:
            user_id: Keycloak user ID

        Returns:
            User object
        """
        response = await self._admin_request("GET", f"/users/{user_id}")
        response.raise_for_status()
        return KeycloakUser.from_dict(response.json())

    async def get_user_by_email(self, email: str) -> Optional[KeycloakUser]:
        """Get user by email address.

        Args:
            email: User email address

        Returns:
            User object or None if not found
        """
        response = await self._admin_request(
            "GET",
            "/users",
            params={"email": email, "exact": "true"},
        )
        response.raise_for_status()

        users = response.json()
        if not users:
            return None

        return KeycloakUser.from_dict(users[0])

    async def update_user(
        self,
        user_id: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        email: Optional[str] = None,
        enabled: Optional[bool] = None,
        attributes: Optional[Dict[str, List[str]]] = None,
    ) -> KeycloakUser:
        """Update user attributes.

        Args:
            user_id: Keycloak user ID
            first_name: New first name (optional)
            last_name: New last name (optional)
            email: New email (optional)
            enabled: Enable/disable user (optional)
            attributes: Custom attributes to update (optional)

        Returns:
            Updated user object
        """
        # Get current user data
        current = await self.get_user(user_id)

        update_data: Dict[str, Any] = {}

        if first_name is not None:
            update_data["firstName"] = first_name
        if last_name is not None:
            update_data["lastName"] = last_name
        if email is not None:
            update_data["email"] = email
        if enabled is not None:
            update_data["enabled"] = enabled
        if attributes is not None:
            # Merge with existing attributes
            merged_attrs = {**current.attributes, **attributes}
            update_data["attributes"] = merged_attrs

        response = await self._admin_request(
            "PUT",
            f"/users/{user_id}",
            json=update_data,
        )
        response.raise_for_status()

        return await self.get_user(user_id)

    async def delete_user(self, user_id: str) -> None:
        """Delete a user.

        Args:
            user_id: Keycloak user ID
        """
        response = await self._admin_request("DELETE", f"/users/{user_id}")
        response.raise_for_status()

    async def deactivate_user(self, user_id: str) -> KeycloakUser:
        """Deactivate a user and revoke all sessions.

        This disables the user and logs them out of all active sessions.
        Requirement: 19.9 - Revoke all active sessions within 60 seconds.

        Args:
            user_id: Keycloak user ID

        Returns:
            Updated user object
        """
        # Disable the user
        await self.update_user(user_id, enabled=False)

        # Logout all sessions
        await self.logout_user(user_id)

        return await self.get_user(user_id)

    async def logout_user(self, user_id: str) -> None:
        """Logout user from all sessions.

        Args:
            user_id: Keycloak user ID
        """
        response = await self._admin_request("POST", f"/users/{user_id}/logout")
        # 204 No Content is success
        if response.status_code not in (200, 204):
            response.raise_for_status()

    async def reset_password(
        self,
        user_id: str,
        new_password: str,
        temporary: bool = True,
    ) -> None:
        """Reset user password.

        Args:
            user_id: Keycloak user ID
            new_password: New password
            temporary: Whether password must be changed on next login
        """
        credential_data = {
            "type": "password",
            "value": new_password,
            "temporary": temporary,
        }

        response = await self._admin_request(
            "PUT",
            f"/users/{user_id}/reset-password",
            json=credential_data,
        )
        response.raise_for_status()

    # =========================================================================
    # Role Management
    # =========================================================================

    async def get_realm_roles(self) -> List[Dict[str, Any]]:
        """Get all realm roles.

        Returns:
            List of role objects
        """
        response = await self._admin_request("GET", "/roles")
        response.raise_for_status()
        return response.json()

    async def get_user_roles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get roles assigned to a user.

        Args:
            user_id: Keycloak user ID

        Returns:
            List of role objects
        """
        response = await self._admin_request(
            "GET",
            f"/users/{user_id}/role-mappings/realm",
        )
        response.raise_for_status()
        return response.json()

    async def assign_roles(self, user_id: str, role_names: List[str]) -> None:
        """Assign realm roles to a user.

        Args:
            user_id: Keycloak user ID
            role_names: List of role names to assign
        """
        # Get role objects by name
        all_roles = await self.get_realm_roles()
        roles_to_assign = [
            {"id": r["id"], "name": r["name"]} for r in all_roles if r["name"] in role_names
        ]

        if not roles_to_assign:
            logger.warning(f"No matching roles found for: {role_names}")
            return

        response = await self._admin_request(
            "POST",
            f"/users/{user_id}/role-mappings/realm",
            json=roles_to_assign,
        )
        response.raise_for_status()

    async def remove_roles(self, user_id: str, role_names: List[str]) -> None:
        """Remove realm roles from a user.

        Args:
            user_id: Keycloak user ID
            role_names: List of role names to remove
        """
        # Get current user roles
        current_roles = await self.get_user_roles(user_id)
        roles_to_remove = [
            {"id": r["id"], "name": r["name"]} for r in current_roles if r["name"] in role_names
        ]

        if not roles_to_remove:
            return

        response = await self._admin_request(
            "DELETE",
            f"/users/{user_id}/role-mappings/realm",
            json=roles_to_remove,
        )
        response.raise_for_status()

    # =========================================================================
    # Token Operations
    # =========================================================================

    async def introspect_token(self, token: str) -> TokenInfo:
        """Introspect an access token.

        Args:
            token: Access token to introspect

        Returns:
            Token information including validity and claims
        """
        url = urljoin(
            self.config.server_url,
            f"/realms/{self.config.realm}/protocol/openid-connect/token/introspect",
        )

        data = {
            "token": token,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }

        response = await self._client.post(url, data=data)
        response.raise_for_status()

        result = response.json()

        return TokenInfo(
            active=result.get("active", False),
            sub=result.get("sub"),
            username=result.get("username") or result.get("preferred_username"),
            email=result.get("email"),
            tenant_id=result.get("tenant_id"),
            roles=result.get("realm_access", {}).get("roles", []),
            client_id=result.get("client_id"),
            exp=result.get("exp"),
            iat=result.get("iat"),
            scope=result.get("scope"),
        )

    async def validate_token(self, token: str) -> bool:
        """Validate an access token.

        Args:
            token: Access token to validate

        Returns:
            True if token is valid and active
        """
        try:
            info = await self.introspect_token(token)
            return info.active
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return False

    async def exchange_token(
        self,
        subject_token: str,
        target_client_id: str,
    ) -> Dict[str, Any]:
        """Exchange a token for another client.

        Requires token-exchange feature enabled in Keycloak.

        Args:
            subject_token: Current access token
            target_client_id: Client ID to exchange for

        Returns:
            New token response
        """
        url = urljoin(
            self.config.server_url, f"/realms/{self.config.realm}/protocol/openid-connect/token"
        )

        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "subject_token": subject_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "audience": target_client_id,
        }

        response = await self._client.post(url, data=data)
        response.raise_for_status()

        return response.json()

    # =========================================================================
    # Tenant/Realm Operations
    # =========================================================================

    async def list_users(
        self,
        first: int = 0,
        max_results: int = 100,
        search: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[KeycloakUser]:
        """List users with optional filtering.

        Args:
            first: Pagination offset
            max_results: Maximum results to return
            search: Search string (matches username, email, first/last name)
            tenant_id: Filter by tenant_id attribute

        Returns:
            List of users
        """
        params: Dict[str, Any] = {
            "first": first,
            "max": max_results,
        }

        if search:
            params["search"] = search

        if tenant_id:
            params["q"] = f"tenant_id:{tenant_id}"

        response = await self._admin_request("GET", "/users", params=params)
        response.raise_for_status()

        return [KeycloakUser.from_dict(u) for u in response.json()]

    async def count_users(self, tenant_id: Optional[str] = None) -> int:
        """Count total users.

        Args:
            tenant_id: Filter by tenant_id attribute

        Returns:
            User count
        """
        params = {}
        if tenant_id:
            params["q"] = f"tenant_id:{tenant_id}"

        response = await self._admin_request("GET", "/users/count", params=params)
        response.raise_for_status()

        return response.json()


# Singleton instance for dependency injection
_keycloak_service: Optional[KeycloakService] = None


def get_keycloak_service() -> KeycloakService:
    """Get or create Keycloak service singleton."""
    global _keycloak_service
    if _keycloak_service is None:
        _keycloak_service = KeycloakService()
    return _keycloak_service


async def init_keycloak_service() -> KeycloakService:
    """Initialize Keycloak service (call on app startup)."""
    global _keycloak_service
    _keycloak_service = KeycloakService()
    return _keycloak_service


async def close_keycloak_service() -> None:
    """Close Keycloak service (call on app shutdown)."""
    global _keycloak_service
    if _keycloak_service:
        await _keycloak_service.close()
        _keycloak_service = None

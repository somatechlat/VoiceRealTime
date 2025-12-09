"""Keycloak Authentication Middleware for Portal API.

Provides JWT validation and user context extraction from Keycloak tokens.

Requirements: 21.1, 21.2
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


@dataclass
class UserContext:
    """Authenticated user context from Keycloak token."""
    
    user_id: str
    tenant_id: str
    email: str
    username: str
    roles: List[str]
    permissions: List[str]
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions
    
    def is_admin(self) -> bool:
        """Check if user is a tenant admin."""
        return "tenant_admin" in self.roles or "admin:*" in self.permissions


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> UserContext:
    """Extract and validate user from Keycloak JWT token.
    
    Args:
        request: FastAPI request
        credentials: Bearer token credentials
        
    Returns:
        Authenticated user context
        
    Raises:
        HTTPException: If authentication fails
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    try:
        # Import JWT validator
        from ...app.services.jwt_validator import JWTValidator
        
        validator = JWTValidator()
        claims = await validator.validate_token(token)
        
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Extract user context from claims
        return UserContext(
            user_id=claims.get("sub", ""),
            tenant_id=claims.get("tenant_id", claims.get("azp", "")),
            email=claims.get("email", ""),
            username=claims.get("preferred_username", claims.get("sub", "")),
            roles=claims.get("realm_access", {}).get("roles", []),
            permissions=claims.get("resource_access", {}).get(
                "agentvoicebox-api", {}
            ).get("roles", []),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(role: str):
    """Dependency to require a specific role.
    
    Args:
        role: Required role name
        
    Returns:
        Dependency function
    """
    async def check_role(user: UserContext = Depends(get_current_user)) -> UserContext:
        if not user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required",
            )
        return user
    
    return check_role


def require_permission(permission: str):
    """Dependency to require a specific permission.
    
    Args:
        permission: Required permission name
        
    Returns:
        Dependency function
    """
    async def check_permission(user: UserContext = Depends(get_current_user)) -> UserContext:
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required",
            )
        return user
    
    return check_permission


def require_admin():
    """Dependency to require tenant admin role."""
    return require_role("tenant_admin")

"""JWT Validation Service for Keycloak tokens.

This module provides JWT validation for the gateway:
- JWKS-based signature verification
- Token claims extraction
- Role-based access control
- Caching of JWKS keys

Requirements: 19.4, 19.7
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

# PyJWT is required for JWT validation
try:
    import jwt
    from jwt import PyJWKClient, PyJWKClientError
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not installed - JWT validation unavailable")


@dataclass
class JWTConfig:
    """JWT validation configuration."""
    
    issuer: str = "http://localhost:8080/realms/agentvoicebox"
    audience: str = "agentvoicebox-api"
    jwks_uri: str = "http://localhost:8080/realms/agentvoicebox/protocol/openid-connect/certs"
    algorithms: List[str] = field(default_factory=lambda: ["RS256"])
    verify_exp: bool = True
    verify_aud: bool = True
    verify_iss: bool = True
    leeway: int = 30  # seconds of leeway for exp/iat validation
    cache_ttl: int = 3600  # JWKS cache TTL in seconds
    
    @classmethod
    def from_env(cls) -> "JWTConfig":
        """Create config from environment variables."""
        keycloak_url = os.getenv("KEYCLOAK_SERVER_URL", "http://localhost:8080")
        realm = os.getenv("KEYCLOAK_REALM", "agentvoicebox")
        
        return cls(
            issuer=os.getenv(
                "JWT_ISSUER",
                f"{keycloak_url}/realms/{realm}"
            ),
            audience=os.getenv("JWT_AUDIENCE", "agentvoicebox-api"),
            jwks_uri=os.getenv(
                "JWT_JWKS_URI",
                f"{keycloak_url}/realms/{realm}/protocol/openid-connect/certs"
            ),
            algorithms=os.getenv("JWT_ALGORITHMS", "RS256").split(","),
            verify_exp=os.getenv("JWT_VERIFY_EXP", "true").lower() == "true",
            verify_aud=os.getenv("JWT_VERIFY_AUD", "true").lower() == "true",
            verify_iss=os.getenv("JWT_VERIFY_ISS", "true").lower() == "true",
            leeway=int(os.getenv("JWT_LEEWAY", "30")),
            cache_ttl=int(os.getenv("JWT_CACHE_TTL", "3600")),
        )


@dataclass
class JWTClaims:
    """Validated JWT claims."""
    
    sub: str  # Subject (user ID)
    iss: str  # Issuer
    aud: str  # Audience
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp
    
    # Custom claims
    tenant_id: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None
    name: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    client_roles: Dict[str, List[str]] = field(default_factory=dict)
    scope: Optional[str] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.exp
    
    @property
    def scopes(self) -> Set[str]:
        """Get scopes as a set."""
        if self.scope:
            return set(self.scope.split())
        return set()
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific realm role."""
        return role in self.roles
    
    def has_any_role(self, roles: List[str]) -> bool:
        """Check if user has any of the specified roles."""
        return bool(set(roles) & set(self.roles))
    
    def has_all_roles(self, roles: List[str]) -> bool:
        """Check if user has all specified roles."""
        return set(roles).issubset(set(self.roles))
    
    def has_client_role(self, client_id: str, role: str) -> bool:
        """Check if user has a specific client role."""
        client_roles = self.client_roles.get(client_id, [])
        return role in client_roles
    
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "JWTClaims":
        """Create claims from decoded JWT payload."""
        # Extract realm roles
        realm_access = payload.get("realm_access", {})
        roles = realm_access.get("roles", [])
        
        # Also check for roles in custom claim
        if not roles:
            roles = payload.get("roles", [])
        
        # Extract client roles
        resource_access = payload.get("resource_access", {})
        client_roles = {
            client: access.get("roles", [])
            for client, access in resource_access.items()
        }
        
        return cls(
            sub=payload.get("sub", ""),
            iss=payload.get("iss", ""),
            aud=payload.get("aud", ""),
            exp=payload.get("exp", 0),
            iat=payload.get("iat", 0),
            tenant_id=payload.get("tenant_id"),
            email=payload.get("email"),
            username=payload.get("preferred_username") or payload.get("username"),
            name=payload.get("name"),
            roles=roles,
            client_roles=client_roles,
            scope=payload.get("scope"),
        )


class JWTValidationError(Exception):
    """JWT validation error."""
    
    def __init__(self, message: str, code: str = "invalid_token"):
        super().__init__(message)
        self.code = code


class JWTValidator:
    """JWT validator with JWKS support.
    
    Validates Keycloak-issued JWTs using JWKS for signature verification.
    Caches JWKS keys for performance.
    """
    
    def __init__(self, config: Optional[JWTConfig] = None):
        """Initialize JWT validator.
        
        Args:
            config: JWT validation configuration. If None, loads from environment.
        """
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT is required for JWT validation")
        
        self.config = config or JWTConfig.from_env()
        self._jwks_client: Optional[PyJWKClient] = None
        self._jwks_last_refresh: float = 0
    
    def _get_jwks_client(self) -> PyJWKClient:
        """Get or create JWKS client with caching."""
        now = time.time()
        
        # Refresh JWKS client if cache expired
        if (
            self._jwks_client is None
            or now - self._jwks_last_refresh > self.config.cache_ttl
        ):
            self._jwks_client = PyJWKClient(
                self.config.jwks_uri,
                cache_keys=True,
                lifespan=self.config.cache_ttl,
            )
            self._jwks_last_refresh = now
        
        return self._jwks_client
    
    def validate(self, token: str) -> JWTClaims:
        """Validate a JWT and return claims.
        
        Args:
            token: JWT access token
            
        Returns:
            Validated JWT claims
            
        Raises:
            JWTValidationError: If token is invalid
        """
        try:
            # Get signing key from JWKS
            jwks_client = self._get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            
            # Decode and validate token
            options = {
                "verify_exp": self.config.verify_exp,
                "verify_aud": self.config.verify_aud,
                "verify_iss": self.config.verify_iss,
                "require": ["exp", "iat", "sub"],
            }
            
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.config.algorithms,
                audience=self.config.audience if self.config.verify_aud else None,
                issuer=self.config.issuer if self.config.verify_iss else None,
                leeway=self.config.leeway,
                options=options,
            )
            
            return JWTClaims.from_payload(payload)
            
        except jwt.ExpiredSignatureError:
            raise JWTValidationError("Token has expired", "token_expired")
        except jwt.InvalidAudienceError:
            raise JWTValidationError("Invalid audience", "invalid_audience")
        except jwt.InvalidIssuerError:
            raise JWTValidationError("Invalid issuer", "invalid_issuer")
        except jwt.InvalidSignatureError:
            raise JWTValidationError("Invalid signature", "invalid_signature")
        except PyJWKClientError as e:
            logger.error(f"JWKS client error: {e}")
            raise JWTValidationError("Unable to verify token signature", "jwks_error")
        except jwt.PyJWTError as e:
            logger.error(f"JWT validation error: {e}")
            raise JWTValidationError(f"Token validation failed: {e}", "validation_error")
        except Exception as e:
            logger.error(f"Unexpected JWT validation error: {e}")
            raise JWTValidationError("Token validation failed", "unknown_error")
    
    def validate_optional(self, token: Optional[str]) -> Optional[JWTClaims]:
        """Validate a JWT if provided, return None if not.
        
        Args:
            token: JWT access token or None
            
        Returns:
            Validated JWT claims or None
        """
        if not token:
            return None
        
        try:
            return self.validate(token)
        except JWTValidationError:
            return None
    
    def decode_unverified(self, token: str) -> Dict[str, Any]:
        """Decode JWT without verification (for debugging only).
        
        WARNING: Do not use for authentication!
        
        Args:
            token: JWT access token
            
        Returns:
            Decoded payload (unverified)
        """
        return jwt.decode(token, options={"verify_signature": False})


# Role-based access control helpers

def require_roles(claims: JWTClaims, required_roles: List[str]) -> None:
    """Require user to have all specified roles.
    
    Args:
        claims: Validated JWT claims
        required_roles: List of required role names
        
    Raises:
        JWTValidationError: If user lacks required roles
    """
    if not claims.has_all_roles(required_roles):
        missing = set(required_roles) - set(claims.roles)
        raise JWTValidationError(
            f"Missing required roles: {missing}",
            "insufficient_permissions"
        )


def require_any_role(claims: JWTClaims, allowed_roles: List[str]) -> None:
    """Require user to have at least one of the specified roles.
    
    Args:
        claims: Validated JWT claims
        allowed_roles: List of allowed role names
        
    Raises:
        JWTValidationError: If user lacks all allowed roles
    """
    if not claims.has_any_role(allowed_roles):
        raise JWTValidationError(
            f"Requires one of roles: {allowed_roles}",
            "insufficient_permissions"
        )


def require_tenant(claims: JWTClaims, tenant_id: str) -> None:
    """Require user to belong to a specific tenant.
    
    Args:
        claims: Validated JWT claims
        tenant_id: Required tenant ID
        
    Raises:
        JWTValidationError: If user is not in the tenant
    """
    if claims.tenant_id != tenant_id:
        raise JWTValidationError(
            "Access denied to this tenant",
            "tenant_mismatch"
        )


# Singleton instance for dependency injection
_jwt_validator: Optional[JWTValidator] = None


def get_jwt_validator() -> JWTValidator:
    """Get or create JWT validator singleton."""
    global _jwt_validator
    if _jwt_validator is None:
        _jwt_validator = JWTValidator()
    return _jwt_validator


def init_jwt_validator(config: Optional[JWTConfig] = None) -> JWTValidator:
    """Initialize JWT validator (call on app startup)."""
    global _jwt_validator
    _jwt_validator = JWTValidator(config)
    return _jwt_validator

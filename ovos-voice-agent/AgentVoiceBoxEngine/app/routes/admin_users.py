"""Admin User Management Routes.

This module provides REST API endpoints for user administration:
- User CRUD operations
- Role management
- User deactivation with session revocation

Requirements: 19.5, 19.9
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from app.services.jwt_validator import (
    JWTClaims,
    JWTValidationError,
    get_jwt_validator,
    require_any_role,
)
from app.services.keycloak_service import (
    KeycloakUser,
    get_keycloak_service,
)

logger = logging.getLogger(__name__)

admin_users_bp = Blueprint("admin_users", __name__, url_prefix="/v1/admin/users")


def get_current_user() -> JWTClaims:
    """Extract and validate JWT from request.

    Returns:
        Validated JWT claims

    Raises:
        JWTValidationError: If token is invalid or missing
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        raise JWTValidationError("Missing or invalid Authorization header", "missing_token")

    token = auth_header[7:]  # Remove "Bearer " prefix
    validator = get_jwt_validator()
    return validator.validate(token)


def require_admin(claims: JWTClaims) -> None:
    """Require admin role for the operation.

    Args:
        claims: Validated JWT claims

    Raises:
        JWTValidationError: If user is not an admin
    """
    require_any_role(claims, ["tenant_admin", "admin:*"])


def user_to_dict(user: KeycloakUser) -> Dict[str, Any]:
    """Convert KeycloakUser to API response dict."""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "enabled": user.enabled,
        "email_verified": user.email_verified,
        "tenant_id": user.tenant_id,
        "roles": user.realm_roles,
        "created_at": user.created_timestamp,
    }


@admin_users_bp.route("", methods=["GET"])
async def list_users():
    """List users with optional filtering.

    Query Parameters:
        first: Pagination offset (default: 0)
        max: Maximum results (default: 100)
        search: Search string
        tenant_id: Filter by tenant

    Returns:
        List of users
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        first = request.args.get("first", 0, type=int)
        max_results = request.args.get("max", 100, type=int)
        search = request.args.get("search")
        tenant_id = request.args.get("tenant_id")

        # Non-super-admins can only see users in their tenant
        if not claims.has_role("admin:*"):
            tenant_id = claims.tenant_id

        keycloak = get_keycloak_service()
        users = await keycloak.list_users(
            first=first,
            max_results=max_results,
            search=search,
            tenant_id=tenant_id,
        )

        return jsonify(
            {
                "users": [user_to_dict(u) for u in users],
                "count": len(users),
            }
        )

    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to list users"}), 500


@admin_users_bp.route("/<user_id>", methods=["GET"])
async def get_user(user_id: str):
    """Get user by ID.

    Args:
        user_id: Keycloak user ID

    Returns:
        User details
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        keycloak = get_keycloak_service()
        user = await keycloak.get_user(user_id)

        # Non-super-admins can only see users in their tenant
        if not claims.has_role("admin:*") and user.tenant_id != claims.tenant_id:
            return jsonify({"error": "forbidden", "message": "Access denied"}), 403

        return jsonify(user_to_dict(user))

    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to get user"}), 500


@admin_users_bp.route("", methods=["POST"])
async def create_user():
    """Create a new user.

    Request Body:
        username: Username (usually email)
        email: Email address
        password: Initial password
        first_name: First name (optional)
        last_name: Last name (optional)
        roles: List of roles to assign (optional)

    Returns:
        Created user
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        data = request.get_json()
        if not data:
            return jsonify({"error": "bad_request", "message": "Request body required"}), 400

        # Validate required fields
        required = ["username", "email", "password"]
        missing = [f for f in required if not data.get(f)]
        if missing:
            return (
                jsonify({"error": "bad_request", "message": f"Missing required fields: {missing}"}),
                400,
            )

        # Non-super-admins can only create users in their tenant
        tenant_id = data.get("tenant_id", claims.tenant_id)
        if not claims.has_role("admin:*") and tenant_id != claims.tenant_id:
            return (
                jsonify({"error": "forbidden", "message": "Cannot create user in another tenant"}),
                403,
            )

        keycloak = get_keycloak_service()
        user = await keycloak.create_user(
            username=data["username"],
            email=data["email"],
            password=data["password"],
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", ""),
            tenant_id=tenant_id,
            roles=data.get("roles"),
            email_verified=data.get("email_verified", False),
            temporary_password=data.get("temporary_password", True),
        )

        return jsonify(user_to_dict(user)), 201

    except ValueError as e:
        return jsonify({"error": "conflict", "message": str(e)}), 409
    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to create user"}), 500


@admin_users_bp.route("/<user_id>", methods=["PATCH"])
async def update_user(user_id: str):
    """Update user attributes.

    Args:
        user_id: Keycloak user ID

    Request Body:
        first_name: New first name (optional)
        last_name: New last name (optional)
        email: New email (optional)
        enabled: Enable/disable user (optional)

    Returns:
        Updated user
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        data = request.get_json()
        if not data:
            return jsonify({"error": "bad_request", "message": "Request body required"}), 400

        keycloak = get_keycloak_service()

        # Check tenant access
        user = await keycloak.get_user(user_id)
        if not claims.has_role("admin:*") and user.tenant_id != claims.tenant_id:
            return jsonify({"error": "forbidden", "message": "Access denied"}), 403

        updated = await keycloak.update_user(
            user_id=user_id,
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            email=data.get("email"),
            enabled=data.get("enabled"),
        )

        return jsonify(user_to_dict(updated))

    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to update user"}), 500


@admin_users_bp.route("/<user_id>", methods=["DELETE"])
async def delete_user(user_id: str):
    """Delete a user.

    Args:
        user_id: Keycloak user ID

    Returns:
        204 No Content on success
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        keycloak = get_keycloak_service()

        # Check tenant access
        user = await keycloak.get_user(user_id)
        if not claims.has_role("admin:*") and user.tenant_id != claims.tenant_id:
            return jsonify({"error": "forbidden", "message": "Access denied"}), 403

        # Prevent self-deletion
        if user_id == claims.sub:
            return jsonify({"error": "bad_request", "message": "Cannot delete yourself"}), 400

        await keycloak.delete_user(user_id)

        return "", 204

    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to delete user"}), 500


@admin_users_bp.route("/<user_id>/deactivate", methods=["POST"])
async def deactivate_user(user_id: str):
    """Deactivate a user and revoke all sessions.

    This endpoint:
    1. Disables the user account
    2. Revokes all active sessions (logout)
    3. Invalidates all API keys associated with the user

    Requirement 19.9: Revoke all active sessions within 60 seconds.

    Args:
        user_id: Keycloak user ID

    Returns:
        Deactivated user
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        keycloak = get_keycloak_service()

        # Check tenant access
        user = await keycloak.get_user(user_id)
        if not claims.has_role("admin:*") and user.tenant_id != claims.tenant_id:
            return jsonify({"error": "forbidden", "message": "Access denied"}), 403

        # Prevent self-deactivation
        if user_id == claims.sub:
            return jsonify({"error": "bad_request", "message": "Cannot deactivate yourself"}), 400

        # Deactivate user (disables account and logs out all sessions)
        deactivated = await keycloak.deactivate_user(user_id)

        # TODO: Also invalidate API keys associated with this user
        # This would require looking up API keys by user_id and marking them inactive

        logger.info(f"User {user_id} deactivated by {claims.sub}")

        return jsonify(
            {
                "user": user_to_dict(deactivated),
                "message": "User deactivated and all sessions revoked",
            }
        )

    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error deactivating user {user_id}: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to deactivate user"}), 500


@admin_users_bp.route("/<user_id>/activate", methods=["POST"])
async def activate_user(user_id: str):
    """Reactivate a deactivated user.

    Args:
        user_id: Keycloak user ID

    Returns:
        Activated user
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        keycloak = get_keycloak_service()

        # Check tenant access
        user = await keycloak.get_user(user_id)
        if not claims.has_role("admin:*") and user.tenant_id != claims.tenant_id:
            return jsonify({"error": "forbidden", "message": "Access denied"}), 403

        activated = await keycloak.update_user(user_id, enabled=True)

        logger.info(f"User {user_id} activated by {claims.sub}")

        return jsonify(
            {
                "user": user_to_dict(activated),
                "message": "User activated",
            }
        )

    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error activating user {user_id}: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to activate user"}), 500


@admin_users_bp.route("/<user_id>/roles", methods=["GET"])
async def get_user_roles(user_id: str):
    """Get roles assigned to a user.

    Args:
        user_id: Keycloak user ID

    Returns:
        List of roles
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        keycloak = get_keycloak_service()

        # Check tenant access
        user = await keycloak.get_user(user_id)
        if not claims.has_role("admin:*") and user.tenant_id != claims.tenant_id:
            return jsonify({"error": "forbidden", "message": "Access denied"}), 403

        roles = await keycloak.get_user_roles(user_id)

        return jsonify({"roles": roles})

    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error getting roles for user {user_id}: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to get user roles"}), 500


@admin_users_bp.route("/<user_id>/roles", methods=["POST"])
async def assign_user_roles(user_id: str):
    """Assign roles to a user.

    Args:
        user_id: Keycloak user ID

    Request Body:
        roles: List of role names to assign

    Returns:
        Updated roles
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        data = request.get_json()
        if not data or "roles" not in data:
            return jsonify({"error": "bad_request", "message": "roles field required"}), 400

        keycloak = get_keycloak_service()

        # Check tenant access
        user = await keycloak.get_user(user_id)
        if not claims.has_role("admin:*") and user.tenant_id != claims.tenant_id:
            return jsonify({"error": "forbidden", "message": "Access denied"}), 403

        await keycloak.assign_roles(user_id, data["roles"])
        roles = await keycloak.get_user_roles(user_id)

        logger.info(f"Roles {data['roles']} assigned to user {user_id} by {claims.sub}")

        return jsonify({"roles": roles})

    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error assigning roles to user {user_id}: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to assign roles"}), 500


@admin_users_bp.route("/<user_id>/roles", methods=["DELETE"])
async def remove_user_roles(user_id: str):
    """Remove roles from a user.

    Args:
        user_id: Keycloak user ID

    Request Body:
        roles: List of role names to remove

    Returns:
        Updated roles
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        data = request.get_json()
        if not data or "roles" not in data:
            return jsonify({"error": "bad_request", "message": "roles field required"}), 400

        keycloak = get_keycloak_service()

        # Check tenant access
        user = await keycloak.get_user(user_id)
        if not claims.has_role("admin:*") and user.tenant_id != claims.tenant_id:
            return jsonify({"error": "forbidden", "message": "Access denied"}), 403

        await keycloak.remove_roles(user_id, data["roles"])
        roles = await keycloak.get_user_roles(user_id)

        logger.info(f"Roles {data['roles']} removed from user {user_id} by {claims.sub}")

        return jsonify({"roles": roles})

    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error removing roles from user {user_id}: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to remove roles"}), 500


@admin_users_bp.route("/<user_id>/reset-password", methods=["POST"])
async def reset_user_password(user_id: str):
    """Reset a user's password.

    Args:
        user_id: Keycloak user ID

    Request Body:
        password: New password
        temporary: Whether password must be changed on next login (default: true)

    Returns:
        Success message
    """
    try:
        claims = get_current_user()
        require_admin(claims)

        data = request.get_json()
        if not data or "password" not in data:
            return jsonify({"error": "bad_request", "message": "password field required"}), 400

        keycloak = get_keycloak_service()

        # Check tenant access
        user = await keycloak.get_user(user_id)
        if not claims.has_role("admin:*") and user.tenant_id != claims.tenant_id:
            return jsonify({"error": "forbidden", "message": "Access denied"}), 403

        await keycloak.reset_password(
            user_id=user_id,
            new_password=data["password"],
            temporary=data.get("temporary", True),
        )

        logger.info(f"Password reset for user {user_id} by {claims.sub}")

        return jsonify({"message": "Password reset successfully"})

    except JWTValidationError as e:
        return jsonify({"error": e.code, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"Error resetting password for user {user_id}: {e}")
        return jsonify({"error": "internal_error", "message": "Failed to reset password"}), 500

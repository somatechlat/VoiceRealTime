"""Team Management Routes.

Provides endpoints for:
- List team members
- Invite users
- Update user roles
- Remove users
- Transfer ownership

Requirements: 21.9
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

from ..auth import UserContext, get_current_user, require_admin

logger = logging.getLogger(__name__)
router = APIRouter()


class TeamMember(BaseModel):
    """Team member details."""
    
    id: str = Field(description="User ID")
    email: str = Field(description="Email address")
    name: str = Field(description="Display name")
    roles: List[str] = Field(description="Assigned roles")
    status: str = Field(description="User status (active, invited, disabled)")
    joined_at: Optional[datetime] = Field(description="Join date")
    last_login_at: Optional[datetime] = Field(description="Last login")


class InviteRequest(BaseModel):
    """Request to invite a new team member."""
    
    email: EmailStr = Field(description="Email address to invite")
    roles: List[str] = Field(
        default=["viewer"],
        description="Roles to assign",
    )
    message: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional invitation message",
    )


class InviteResponse(BaseModel):
    """Response from invitation."""
    
    id: str = Field(description="Invitation ID")
    email: str = Field(description="Invited email")
    roles: List[str] = Field(description="Assigned roles")
    expires_at: datetime = Field(description="Invitation expiration")
    status: str = Field(description="Invitation status")


class UpdateRolesRequest(BaseModel):
    """Request to update user roles."""
    
    roles: List[str] = Field(description="New roles to assign")


class TransferOwnershipRequest(BaseModel):
    """Request to transfer tenant ownership."""
    
    new_owner_id: str = Field(description="User ID of new owner")
    confirm: bool = Field(description="Confirmation flag")


# Available roles
AVAILABLE_ROLES = [
    {"name": "tenant_admin", "description": "Full administrative access"},
    {"name": "developer", "description": "API access and key management"},
    {"name": "viewer", "description": "Read-only access to dashboard"},
    {"name": "billing_admin", "description": "Billing and payment management"},
]


@router.get("/team/roles")
async def list_available_roles(
    user: UserContext = Depends(get_current_user),
) -> List[dict]:
    """List available roles that can be assigned."""
    return AVAILABLE_ROLES


@router.get("/team/members", response_model=List[TeamMember])
async def list_team_members(
    user: UserContext = Depends(get_current_user),
    include_disabled: bool = Query(default=False),
) -> List[TeamMember]:
    """List all team members for the tenant."""
    try:
        from ....app.services.keycloak_service import get_keycloak_service
        
        keycloak = get_keycloak_service()
        users = await keycloak.list_users(tenant_id=user.tenant_id)
        
        members = []
        for u in users:
            if not include_disabled and not u.enabled:
                continue
            
            members.append(TeamMember(
                id=u.id,
                email=u.email,
                name=f"{u.first_name} {u.last_name}".strip() or u.username,
                roles=u.realm_roles,
                status="active" if u.enabled else "disabled",
                joined_at=datetime.fromtimestamp(u.created_timestamp / 1000) if u.created_timestamp else None,
                last_login_at=None,  # Would come from Keycloak sessions
            ))
        
        return members
        
    except Exception as e:
        logger.error(f"Failed to list team members: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list team members",
        )


@router.post("/team/invite", response_model=InviteResponse, status_code=status.HTTP_201_CREATED)
async def invite_team_member(
    request: InviteRequest,
    user: UserContext = Depends(require_admin()),
) -> InviteResponse:
    """Invite a new team member.
    
    Requires tenant_admin role.
    """
    try:
        from ....app.services.keycloak_service import get_keycloak_service
        import secrets
        
        keycloak = get_keycloak_service()
        
        # Check if user already exists
        existing = await keycloak.get_user_by_email(request.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists",
            )
        
        # Create user with temporary password
        temp_password = secrets.token_urlsafe(16)
        new_user = await keycloak.create_user(
            username=request.email,
            email=request.email,
            password=temp_password,
            tenant_id=user.tenant_id,
            roles=request.roles,
            email_verified=False,
            enabled=True,
            temporary_password=True,
        )
        
        # In production, send invitation email here
        logger.info(f"Invited user {request.email} to tenant {user.tenant_id}")
        
        return InviteResponse(
            id=new_user.id,
            email=request.email,
            roles=request.roles,
            expires_at=datetime.now(),  # Would be actual expiration
            status="pending",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to invite team member: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to invite team member",
        )


@router.get("/team/members/{member_id}", response_model=TeamMember)
async def get_team_member(
    member_id: str,
    user: UserContext = Depends(get_current_user),
) -> TeamMember:
    """Get team member details."""
    try:
        from ....app.services.keycloak_service import get_keycloak_service
        
        keycloak = get_keycloak_service()
        member = await keycloak.get_user(member_id)
        
        # Verify member belongs to same tenant
        if member.tenant_id != user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team member not found",
            )
        
        return TeamMember(
            id=member.id,
            email=member.email,
            name=f"{member.first_name} {member.last_name}".strip() or member.username,
            roles=member.realm_roles,
            status="active" if member.enabled else "disabled",
            joined_at=datetime.fromtimestamp(member.created_timestamp / 1000) if member.created_timestamp else None,
            last_login_at=None,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get team member: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get team member",
        )


@router.patch("/team/members/{member_id}/roles", response_model=TeamMember)
async def update_member_roles(
    member_id: str,
    request: UpdateRolesRequest,
    user: UserContext = Depends(require_admin()),
) -> TeamMember:
    """Update team member roles.
    
    Requires tenant_admin role.
    """
    try:
        from ....app.services.keycloak_service import get_keycloak_service
        
        keycloak = get_keycloak_service()
        
        # Get member
        member = await keycloak.get_user(member_id)
        if member.tenant_id != user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team member not found",
            )
        
        # Can't modify own roles
        if member_id == user.user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot modify your own roles",
            )
        
        # Update roles
        current_roles = await keycloak.get_user_roles(member_id)
        current_role_names = [r["name"] for r in current_roles]
        
        # Remove old roles
        roles_to_remove = [r for r in current_role_names if r not in request.roles]
        if roles_to_remove:
            await keycloak.remove_roles(member_id, roles_to_remove)
        
        # Add new roles
        roles_to_add = [r for r in request.roles if r not in current_role_names]
        if roles_to_add:
            await keycloak.assign_roles(member_id, roles_to_add)
        
        # Get updated member
        updated = await keycloak.get_user(member_id)
        
        return TeamMember(
            id=updated.id,
            email=updated.email,
            name=f"{updated.first_name} {updated.last_name}".strip() or updated.username,
            roles=request.roles,
            status="active" if updated.enabled else "disabled",
            joined_at=datetime.fromtimestamp(updated.created_timestamp / 1000) if updated.created_timestamp else None,
            last_login_at=None,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update member roles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update member roles",
        )


@router.delete("/team/members/{member_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def remove_team_member(
    member_id: str,
    user: UserContext = Depends(require_admin()),
):
    """Remove a team member.
    
    Requires tenant_admin role.
    """
    try:
        from ....app.services.keycloak_service import get_keycloak_service
        
        keycloak = get_keycloak_service()
        
        # Get member
        member = await keycloak.get_user(member_id)
        if member.tenant_id != user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team member not found",
            )
        
        # Can't remove yourself
        if member_id == user.user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove yourself",
            )
        
        # Deactivate user (don't delete, for audit trail)
        await keycloak.deactivate_user(member_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove team member: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove team member",
        )


@router.post("/team/transfer-ownership")
async def transfer_ownership(
    request: TransferOwnershipRequest,
    user: UserContext = Depends(require_admin()),
) -> dict:
    """Transfer tenant ownership to another user.
    
    Requires tenant_admin role and confirmation.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation required",
        )
    
    try:
        from ....app.services.keycloak_service import get_keycloak_service
        
        keycloak = get_keycloak_service()
        
        # Verify new owner exists and is in same tenant
        new_owner = await keycloak.get_user(request.new_owner_id)
        if new_owner.tenant_id != user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        # Grant tenant_admin to new owner
        await keycloak.assign_roles(request.new_owner_id, ["tenant_admin"])
        
        # Remove tenant_admin from current owner (optional)
        # await keycloak.remove_roles(user.user_id, ["tenant_admin"])
        
        return {
            "message": "Ownership transferred successfully",
            "new_owner_id": request.new_owner_id,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to transfer ownership: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to transfer ownership",
        )

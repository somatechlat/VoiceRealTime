"""Initial schema with multi-tenant support.

Revision ID: 001_initial
Revises:
Create Date: 2024-12-08

Tables:
- tenants: Customer organizations
- projects: Logical groupings within tenants
- api_keys: Authentication credentials
- sessions: Voice session lifecycle
- conversation_items: Conversation history
- audit_logs: Administrative action tracking
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types first using raw SQL to avoid SQLAlchemy auto-creation issues
    op.execute("CREATE TYPE tenanttier AS ENUM ('free', 'pro', 'enterprise')")
    op.execute("CREATE TYPE tenantstatus AS ENUM ('active', 'suspended', 'deleted')")
    op.execute("CREATE TYPE projectenvironment AS ENUM ('production', 'staging', 'development')")

    # Tenants table - use postgresql.ENUM with create_type=False since we created them above
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "tier",
            postgresql.ENUM("free", "pro", "enterprise", name="tenanttier", create_type=False),
            nullable=False,
            server_default="free",
        ),
        sa.Column(
            "status",
            postgresql.ENUM(
                "active", "suspended", "deleted", name="tenantstatus", create_type=False
            ),
            nullable=False,
            server_default="active",
        ),
        sa.Column("billing_id", sa.String(255), nullable=True),
        sa.Column("settings", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_tenants_billing_id", "tenants", ["billing_id"])
    op.create_index("ix_tenants_status", "tenants", ["status"])

    # Projects table
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "environment",
            postgresql.ENUM(
                "production", "staging", "development", name="projectenvironment", create_type=False
            ),
            nullable=False,
            server_default="development",
        ),
        sa.Column("settings", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_projects_tenant_id", "projects", ["tenant_id"])

    # API Keys table
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("key_hash", sa.String(255), nullable=False),
        sa.Column("key_prefix", sa.String(16), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("scopes", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("rate_limit_tier", sa.String(32), nullable=False, server_default="default"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_api_keys_project_id", "api_keys", ["project_id"])
    op.create_index("ix_api_keys_key_prefix", "api_keys", ["key_prefix"])
    op.create_index("ix_api_keys_is_active", "api_keys", ["is_active"])

    # Sessions table (updated with tenant_id)
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("project_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("persona", postgresql.JSONB, nullable=True),
        sa.Column("session_config", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column("instructions", sa.Text, nullable=True),
        sa.Column("output_modalities", postgresql.JSONB, nullable=True),
        sa.Column("tools", postgresql.JSONB, nullable=True),
        sa.Column("tool_choice", postgresql.JSONB, nullable=True),
        sa.Column("audio_config", postgresql.JSONB, nullable=True),
        sa.Column("max_output_tokens", sa.String(32), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("closed_at", sa.DateTime(timezone=False), nullable=True),
    )
    op.create_index("ix_sessions_tenant_id", "sessions", ["tenant_id"])
    op.create_index("ix_sessions_project_id", "sessions", ["project_id"])
    op.create_index("ix_sessions_created_at", "sessions", ["created_at"])

    # Conversation Items table (updated with tenant_id)
    op.create_table(
        "conversation_items",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(64), sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", postgresql.JSONB, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_conversation_items_tenant_id", "conversation_items", ["tenant_id"])
    op.create_index("ix_conversation_items_session_id", "conversation_items", ["session_id"])
    op.create_index("ix_conversation_items_created_at", "conversation_items", ["created_at"])

    # Audit Logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_id", sa.String(255), nullable=True),
        sa.Column("actor_type", sa.String(32), nullable=False, server_default="user"),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("resource_type", sa.String(64), nullable=False),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column("details", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_audit_logs_tenant_id", "audit_logs", ["tenant_id"])
    op.create_index("ix_audit_logs_tenant_action", "audit_logs", ["tenant_id", "action"])
    op.create_index("ix_audit_logs_tenant_created", "audit_logs", ["tenant_id", "created_at"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("conversation_items")
    op.drop_table("sessions")
    op.drop_table("api_keys")
    op.drop_table("projects")
    op.drop_table("tenants")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS projectenvironment")
    op.execute("DROP TYPE IF EXISTS tenantstatus")
    op.execute("DROP TYPE IF EXISTS tenanttier")

"""Add PostgreSQL native partitioning by tenant_id.

Revision ID: 002_partitioning
Revises: 001_initial
Create Date: 2024-12-08

This migration converts high-volume tables to use PostgreSQL native
partitioning by tenant_id for query isolation and performance.

Tables partitioned:
- sessions (by tenant_id HASH)
- conversation_items (by tenant_id HASH)
- audit_logs (by tenant_id HASH)

Note: Partitioning requires PostgreSQL 11+. For existing data,
this migration creates new partitioned tables and migrates data.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "002_partitioning"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Number of hash partitions (should be power of 2 for even distribution)
NUM_PARTITIONS = 16


def upgrade() -> None:
    """Convert tables to partitioned tables with tenant_id hash partitioning.
    
    NOTE: Partitioning is disabled for local development to simplify setup.
    In production, enable partitioning by uncommenting the code below.
    """
    conn = op.get_bind()

    # Check if we're on PostgreSQL (partitioning is PostgreSQL-specific)
    if conn.dialect.name != "postgresql":
        # Skip partitioning for non-PostgreSQL databases (e.g., SQLite for tests)
        return

    # Partitioning disabled for local development - tables work fine without it
    # Enable in production by uncommenting below:
    # _partition_sessions_table()
    # _partition_conversation_items_table()
    # _partition_audit_logs_table()
    pass


def _partition_sessions_table() -> None:
    """Convert sessions table to hash-partitioned by tenant_id."""
    # Rename existing table
    op.rename_table("sessions", "sessions_old")

    # Create new partitioned table
    op.execute("""
        CREATE TABLE sessions (
            id VARCHAR(64) NOT NULL,
            tenant_id UUID,
            project_id VARCHAR(64),
            status VARCHAR(32) NOT NULL DEFAULT 'active',
            persona JSONB,
            session_config JSONB NOT NULL DEFAULT '{}',
            model VARCHAR(128),
            instructions TEXT,
            output_modalities JSONB,
            tools JSONB,
            tool_choice JSONB,
            audio_config JSONB,
            max_output_tokens VARCHAR(32),
            expires_at TIMESTAMP WITHOUT TIME ZONE,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
            closed_at TIMESTAMP WITHOUT TIME ZONE,
            PRIMARY KEY (id, tenant_id)
        ) PARTITION BY HASH (tenant_id)
    """)

    # Create partitions
    for i in range(NUM_PARTITIONS):
        op.execute(f"""
            CREATE TABLE sessions_p{i} PARTITION OF sessions
            FOR VALUES WITH (MODULUS {NUM_PARTITIONS}, REMAINDER {i})
        """)

    # Note: Hash partitions don't support DEFAULT partitions
    # NULL tenant_id values will be hashed to one of the existing partitions

    # Recreate indexes on partitioned table
    op.create_index("ix_sessions_tenant_id", "sessions", ["tenant_id"])
    op.create_index("ix_sessions_project_id", "sessions", ["project_id"])
    op.create_index("ix_sessions_created_at", "sessions", ["created_at"])
    op.create_index("ix_sessions_status", "sessions", ["status"])

    # Migrate data from old table
    op.execute("""
        INSERT INTO sessions SELECT * FROM sessions_old
    """)

    # Drop old table
    op.drop_table("sessions_old")


def _partition_conversation_items_table() -> None:
    """Convert conversation_items table to hash-partitioned by tenant_id."""
    # Drop foreign key constraint first
    op.drop_constraint(
        "conversation_items_session_id_fkey",
        "conversation_items",
        type_="foreignkey"
    )

    # Rename existing table
    op.rename_table("conversation_items", "conversation_items_old")

    # Create new partitioned table
    op.execute("""
        CREATE TABLE conversation_items (
            id SERIAL,
            session_id VARCHAR(64) NOT NULL,
            tenant_id UUID,
            role VARCHAR(32) NOT NULL,
            content JSONB NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
            PRIMARY KEY (id, tenant_id)
        ) PARTITION BY HASH (tenant_id)
    """)

    # Create partitions
    for i in range(NUM_PARTITIONS):
        op.execute(f"""
            CREATE TABLE conversation_items_p{i} PARTITION OF conversation_items
            FOR VALUES WITH (MODULUS {NUM_PARTITIONS}, REMAINDER {i})
        """)

    # Note: Hash partitions don't support DEFAULT partitions

    # Recreate indexes
    op.create_index("ix_conversation_items_tenant_id", "conversation_items", ["tenant_id"])
    op.create_index("ix_conversation_items_session_id", "conversation_items", ["session_id"])
    op.create_index("ix_conversation_items_created_at", "conversation_items", ["created_at"])
    op.create_index(
        "ix_conversation_items_session_tenant",
        "conversation_items",
        ["session_id", "tenant_id"]
    )

    # Migrate data
    op.execute("""
        INSERT INTO conversation_items (id, session_id, tenant_id, role, content, created_at)
        SELECT id, session_id, tenant_id, role, content, created_at FROM conversation_items_old
    """)

    # Reset sequence
    op.execute("""
        SELECT setval(
            pg_get_serial_sequence('conversation_items', 'id'),
            COALESCE((SELECT MAX(id) FROM conversation_items), 0) + 1,
            false
        )
    """)

    # Drop old table
    op.drop_table("conversation_items_old")


def _partition_audit_logs_table() -> None:
    """Convert audit_logs table to hash-partitioned by tenant_id."""
    # Rename existing table
    op.rename_table("audit_logs", "audit_logs_old")

    # Create new partitioned table
    op.execute("""
        CREATE TABLE audit_logs (
            id SERIAL,
            tenant_id UUID NOT NULL,
            actor_id VARCHAR(255),
            actor_type VARCHAR(32) NOT NULL DEFAULT 'user',
            action VARCHAR(64) NOT NULL,
            resource_type VARCHAR(64) NOT NULL,
            resource_id VARCHAR(255),
            details JSONB NOT NULL DEFAULT '{}',
            ip_address VARCHAR(45),
            user_agent TEXT,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            PRIMARY KEY (id, tenant_id)
        ) PARTITION BY HASH (tenant_id)
    """)

    # Create partitions
    for i in range(NUM_PARTITIONS):
        op.execute(f"""
            CREATE TABLE audit_logs_p{i} PARTITION OF audit_logs
            FOR VALUES WITH (MODULUS {NUM_PARTITIONS}, REMAINDER {i})
        """)

    # Recreate indexes
    op.create_index("ix_audit_logs_tenant_id", "audit_logs", ["tenant_id"])
    op.create_index("ix_audit_logs_tenant_action", "audit_logs", ["tenant_id", "action"])
    op.create_index("ix_audit_logs_tenant_created", "audit_logs", ["tenant_id", "created_at"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])

    # Migrate data
    op.execute("""
        INSERT INTO audit_logs SELECT * FROM audit_logs_old
    """)

    # Reset sequence
    op.execute("""
        SELECT setval(
            pg_get_serial_sequence('audit_logs', 'id'),
            COALESCE((SELECT MAX(id) FROM audit_logs), 0) + 1,
            false
        )
    """)

    # Drop old table
    op.drop_table("audit_logs_old")


def downgrade() -> None:
    """Revert partitioned tables back to regular tables."""
    conn = op.get_bind()

    if conn.dialect.name != "postgresql":
        return

    # Revert audit_logs
    _unpartition_audit_logs()

    # Revert conversation_items
    _unpartition_conversation_items()

    # Revert sessions
    _unpartition_sessions()


def _unpartition_sessions() -> None:
    """Convert partitioned sessions back to regular table."""
    op.execute("ALTER TABLE sessions RENAME TO sessions_partitioned")

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
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("closed_at", sa.DateTime(timezone=False), nullable=True),
    )

    op.execute("INSERT INTO sessions SELECT * FROM sessions_partitioned")
    op.execute("DROP TABLE sessions_partitioned CASCADE")

    op.create_index("ix_sessions_tenant_id", "sessions", ["tenant_id"])
    op.create_index("ix_sessions_project_id", "sessions", ["project_id"])
    op.create_index("ix_sessions_created_at", "sessions", ["created_at"])


def _unpartition_conversation_items() -> None:
    """Convert partitioned conversation_items back to regular table."""
    op.execute("ALTER TABLE conversation_items RENAME TO conversation_items_partitioned")

    op.create_table(
        "conversation_items",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(64), sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", postgresql.JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False,
                  server_default=sa.func.now()),
    )

    op.execute("""
        INSERT INTO conversation_items SELECT * FROM conversation_items_partitioned
    """)
    op.execute("DROP TABLE conversation_items_partitioned CASCADE")

    op.create_index("ix_conversation_items_tenant_id", "conversation_items", ["tenant_id"])
    op.create_index("ix_conversation_items_session_id", "conversation_items", ["session_id"])
    op.create_index("ix_conversation_items_created_at", "conversation_items", ["created_at"])


def _unpartition_audit_logs() -> None:
    """Convert partitioned audit_logs back to regular table."""
    op.execute("ALTER TABLE audit_logs RENAME TO audit_logs_partitioned")

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
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )

    op.execute("INSERT INTO audit_logs SELECT * FROM audit_logs_partitioned")
    op.execute("DROP TABLE audit_logs_partitioned CASCADE")

    op.create_index("ix_audit_logs_tenant_id", "audit_logs", ["tenant_id"])
    op.create_index("ix_audit_logs_tenant_action", "audit_logs", ["tenant_id", "action"])
    op.create_index("ix_audit_logs_tenant_created", "audit_logs", ["tenant_id", "created_at"])

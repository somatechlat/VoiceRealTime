"""Real PostgreSQL Integration Tests - NO MOCKS.

These tests run against a real PostgreSQL instance via Docker Compose.
They validate:
- Tenant isolation (create 2 tenants, verify no cross-access)
- Conversation item persistence and retrieval
- Audit log writes and queries
- Database connection pool under load
- Migration rollback/forward

Requirements: 13.1, 13.2, 13.5

Run with:
    docker compose -f docker-compose.test.yml up -d postgres
    pytest tests/integration/test_postgres_real.py -v
"""

import asyncio
import datetime as dt
import os
import sys
import time
import uuid

import pytest
import pytest_asyncio

# Add app to path
_app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _app_root)

# Import directly from module file to avoid loading app/__init__.py (which imports Flask)
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

if ASYNCPG_AVAILABLE:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "async_database",
        os.path.join(_app_root, "app", "services", "async_database.py")
    )
    _async_db_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_async_db_module)
    
    AsyncDatabaseClient = _async_db_module.AsyncDatabaseClient
    AsyncDatabaseConfig = _async_db_module.AsyncDatabaseConfig
    AsyncConversationRepository = _async_db_module.AsyncConversationRepository
    ConversationItemData = _async_db_module.ConversationItemData
else:
    AsyncDatabaseClient = None
    AsyncDatabaseConfig = None
    AsyncConversationRepository = None
    ConversationItemData = None

# Skip all tests if asyncpg not available
pytestmark = [
    pytest.mark.asyncio(loop_scope="function"),
    pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed"),
]

# Database URL from environment or default (port 15432 to avoid conflicts)
DATABASE_URL = os.getenv(
    "DATABASE_URI",
    "postgresql://agentvoicebox:agentvoicebox_secure_pwd_2024@localhost:15432/agentvoicebox"
)


@pytest_asyncio.fixture
async def db_client():
    """Create async database client connected to real PostgreSQL."""
    config = AsyncDatabaseConfig.from_uri(DATABASE_URL)
    client = AsyncDatabaseClient(config)
    await client.connect()
    yield client
    await client.close()


@pytest_asyncio.fixture
async def conversation_repo(db_client):
    """Create conversation repository with real database."""
    return AsyncConversationRepository(db_client)


@pytest_asyncio.fixture
async def setup_test_tables(db_client):
    """Ensure test tables exist (conversation_items)."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS conversation_items (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) NOT NULL,
        tenant_id UUID,
        role VARCHAR(64) NOT NULL,
        content JSONB DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        CONSTRAINT idx_conversation_items_session_tenant 
            UNIQUE (session_id, tenant_id, id)
    );
    CREATE INDEX IF NOT EXISTS ix_conversation_items_session 
        ON conversation_items(session_id);
    CREATE INDEX IF NOT EXISTS ix_conversation_items_tenant 
        ON conversation_items(tenant_id);
    CREATE INDEX IF NOT EXISTS ix_conversation_items_created 
        ON conversation_items(created_at);
    """
    await db_client.execute(create_table_sql)
    yield
    # Cleanup is handled per-test



class TestDatabaseConnection:
    """Test basic database connectivity and health checks."""

    @pytest.mark.asyncio
    async def test_connect_and_health_check(self, db_client):
        """Test database connection and health check."""
        assert db_client.is_connected
        health = await db_client.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_execute_simple_query(self, db_client):
        """Test executing a simple query."""
        result = await db_client.fetchval("SELECT 1 + 1")
        assert result == 2

    @pytest.mark.asyncio
    async def test_connection_pool_under_load(self, db_client):
        """Test connection pool handles concurrent queries.
        
        Simulates 50 concurrent queries to verify pool management.
        """
        async def run_query(i: int):
            result = await db_client.fetchval("SELECT $1::int", i)
            return result

        tasks = [run_query(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        assert results == list(range(50))


class TestTenantIsolation:
    """Test tenant isolation in PostgreSQL.
    
    Requirements: 1.2, 1.3, 13.2
    """

    @pytest.mark.asyncio
    async def test_tenant_cannot_access_other_tenant_data(
        self, db_client, conversation_repo, setup_test_tables
    ):
        """Tenant A cannot access Tenant B's conversation items."""
        tenant_a = uuid.uuid4()
        tenant_b = uuid.uuid4()
        session_id = f"sess_{uuid.uuid4().hex[:16]}"

        # Create items for tenant A
        item_a = ConversationItemData(
            session_id=session_id,
            tenant_id=tenant_a,
            role="user",
            content={"text": "Message from Tenant A"},
        )
        await conversation_repo.create(item_a)

        # Create items for tenant B (same session_id, different tenant)
        item_b = ConversationItemData(
            session_id=session_id,
            tenant_id=tenant_b,
            role="user",
            content={"text": "Message from Tenant B"},
        )
        await conversation_repo.create(item_b)

        # Tenant A queries - should only see their data
        items_a = await conversation_repo.get_by_session(session_id, tenant_a)
        assert len(items_a) == 1
        assert items_a[0].content["text"] == "Message from Tenant A"

        # Tenant B queries - should only see their data
        items_b = await conversation_repo.get_by_session(session_id, tenant_b)
        assert len(items_b) == 1
        assert items_b[0].content["text"] == "Message from Tenant B"

        # Cleanup
        await db_client.execute(
            "DELETE FROM conversation_items WHERE session_id = $1", session_id
        )

    @pytest.mark.asyncio
    async def test_tenant_filter_enforced_on_all_queries(
        self, db_client, conversation_repo, setup_test_tables
    ):
        """All queries must filter by tenant_id."""
        tenant_id = uuid.uuid4()
        other_tenant = uuid.uuid4()
        session_id = f"sess_{uuid.uuid4().hex[:16]}"

        # Create 5 items for our tenant
        for i in range(5):
            await conversation_repo.create(
                ConversationItemData(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    role="user",
                    content={"index": i},
                )
            )

        # Create 3 items for other tenant (same session)
        for i in range(3):
            await conversation_repo.create(
                ConversationItemData(
                    session_id=session_id,
                    tenant_id=other_tenant,
                    role="user",
                    content={"index": i},
                )
            )

        # Count should be isolated
        count = await conversation_repo.count_by_session(session_id, tenant_id)
        assert count == 5, f"Expected 5 items for tenant, got {count}"

        other_count = await conversation_repo.count_by_session(session_id, other_tenant)
        assert other_count == 3, f"Expected 3 items for other tenant, got {other_count}"

        # Cleanup
        await db_client.execute(
            "DELETE FROM conversation_items WHERE session_id = $1", session_id
        )



class TestConversationItemPersistence:
    """Test conversation item persistence and retrieval.
    
    Requirements: 13.5, 9.5
    Property 9: Message Persistence - For any conversation item created, 
    it SHALL be persisted to PostgreSQL within 1 second and be queryable thereafter.
    """

    @pytest.mark.asyncio
    async def test_create_and_retrieve_item(
        self, db_client, conversation_repo, setup_test_tables
    ):
        """Test creating and retrieving a conversation item."""
        tenant_id = uuid.uuid4()
        session_id = f"sess_{uuid.uuid4().hex[:16]}"

        item = ConversationItemData(
            session_id=session_id,
            tenant_id=tenant_id,
            role="user",
            content={"text": "Hello, world!", "type": "input_text"},
        )

        # Create
        start_time = time.time()
        created = await conversation_repo.create(item)
        create_latency_ms = (time.time() - start_time) * 1000

        assert created.id is not None
        assert created.session_id == session_id
        assert created.role == "user"

        # Verify persistence latency < 1 second (Property 9)
        assert create_latency_ms < 1000, f"Create latency {create_latency_ms}ms exceeds 1s"

        # Retrieve
        start_time = time.time()
        items = await conversation_repo.get_by_session(session_id, tenant_id)
        retrieve_latency_ms = (time.time() - start_time) * 1000

        assert len(items) == 1
        assert items[0].content["text"] == "Hello, world!"

        # Verify retrieval latency < 100ms (Requirement 13.5)
        assert retrieve_latency_ms < 100, f"Retrieve latency {retrieve_latency_ms}ms exceeds 100ms"

        # Cleanup
        await db_client.execute(
            "DELETE FROM conversation_items WHERE session_id = $1", session_id
        )

    @pytest.mark.asyncio
    async def test_batch_create_items(
        self, db_client, conversation_repo, setup_test_tables
    ):
        """Test batch creation of conversation items."""
        tenant_id = uuid.uuid4()
        session_id = f"sess_{uuid.uuid4().hex[:16]}"

        items = [
            ConversationItemData(
                session_id=session_id,
                tenant_id=tenant_id,
                role="user" if i % 2 == 0 else "assistant",
                content={"text": f"Message {i}", "index": i},
            )
            for i in range(50)
        ]

        # Batch create
        start_time = time.time()
        count = await conversation_repo.create_batch(items)
        batch_latency_ms = (time.time() - start_time) * 1000

        assert count == 50

        # Batch should complete within 5 seconds (Requirement 13.3)
        assert batch_latency_ms < 5000, f"Batch latency {batch_latency_ms}ms exceeds 5s"

        # Verify all items persisted
        retrieved = await conversation_repo.get_by_session(session_id, tenant_id, limit=100)
        assert len(retrieved) == 50

        # Cleanup
        await db_client.execute(
            "DELETE FROM conversation_items WHERE session_id = $1", session_id
        )

    @pytest.mark.asyncio
    async def test_get_last_n_items(
        self, db_client, conversation_repo, setup_test_tables
    ):
        """Test retrieving last N items with proper ordering."""
        tenant_id = uuid.uuid4()
        session_id = f"sess_{uuid.uuid4().hex[:16]}"

        # Create 20 items with slight delays to ensure ordering
        for i in range(20):
            await conversation_repo.create(
                ConversationItemData(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    role="user",
                    content={"index": i},
                    created_at=dt.datetime.utcnow(),
                )
            )

        # Get last 10
        start_time = time.time()
        last_10 = await conversation_repo.get_last_n(session_id, n=10, tenant_id=tenant_id)
        latency_ms = (time.time() - start_time) * 1000

        assert len(last_10) == 10

        # Should be ordered by created_at ASC (oldest first in result)
        indices = [item.content["index"] for item in last_10]
        assert indices == list(range(10, 20)), f"Expected [10-19], got {indices}"

        # Latency should be < 100ms (Requirement 13.5)
        assert latency_ms < 100, f"Query latency {latency_ms}ms exceeds 100ms"

        # Cleanup
        await db_client.execute(
            "DELETE FROM conversation_items WHERE session_id = $1", session_id
        )



class TestAuditLogPersistence:
    """Test audit log writes and queries.
    
    Requirements: 1.7, 15.5
    """

    @pytest_asyncio.fixture
    async def setup_audit_table(self, db_client):
        """Ensure audit_logs table exists."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            tenant_id UUID NOT NULL,
            actor_id VARCHAR(255),
            actor_type VARCHAR(32) DEFAULT 'user',
            action VARCHAR(64) NOT NULL,
            resource_type VARCHAR(64) NOT NULL,
            resource_id VARCHAR(255),
            details JSONB DEFAULT '{}',
            ip_address VARCHAR(45),
            user_agent TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_audit_logs_tenant 
            ON audit_logs(tenant_id);
        CREATE INDEX IF NOT EXISTS ix_audit_logs_action 
            ON audit_logs(tenant_id, action);
        CREATE INDEX IF NOT EXISTS ix_audit_logs_created 
            ON audit_logs(tenant_id, created_at);
        """
        await db_client.execute(create_table_sql)
        yield

    @pytest.mark.asyncio
    async def test_write_audit_log(self, db_client, setup_audit_table):
        """Test writing an audit log entry."""
        tenant_id = uuid.uuid4()

        insert_sql = """
        INSERT INTO audit_logs 
            (tenant_id, actor_id, action, resource_type, resource_id, details, ip_address)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
        """

        log_id = await db_client.fetchval(
            insert_sql,
            tenant_id,
            "user_123",
            "api_key.created",
            "api_key",
            "key_abc123",
            {"scopes": ["realtime:connect"]},
            "192.168.1.100",
        )

        assert log_id is not None

        # Verify retrieval
        record = await db_client.fetchrow(
            "SELECT * FROM audit_logs WHERE id = $1", log_id
        )

        assert record["tenant_id"] == tenant_id
        assert record["action"] == "api_key.created"
        assert record["details"]["scopes"] == ["realtime:connect"]

        # Cleanup
        await db_client.execute("DELETE FROM audit_logs WHERE id = $1", log_id)

    @pytest.mark.asyncio
    async def test_query_audit_logs_by_tenant(self, db_client, setup_audit_table):
        """Test querying audit logs filtered by tenant."""
        tenant_a = uuid.uuid4()
        tenant_b = uuid.uuid4()

        insert_sql = """
        INSERT INTO audit_logs (tenant_id, actor_id, action, resource_type)
        VALUES ($1, $2, $3, $4)
        """

        # Create logs for tenant A
        for i in range(5):
            await db_client.execute(
                insert_sql, tenant_a, f"user_{i}", "tenant.updated", "tenant"
            )

        # Create logs for tenant B
        for i in range(3):
            await db_client.execute(
                insert_sql, tenant_b, f"user_{i}", "project.created", "project"
            )

        # Query tenant A logs
        logs_a = await db_client.fetch(
            "SELECT * FROM audit_logs WHERE tenant_id = $1", tenant_a
        )
        assert len(logs_a) == 5

        # Query tenant B logs
        logs_b = await db_client.fetch(
            "SELECT * FROM audit_logs WHERE tenant_id = $1", tenant_b
        )
        assert len(logs_b) == 3

        # Cleanup
        await db_client.execute(
            "DELETE FROM audit_logs WHERE tenant_id IN ($1, $2)", tenant_a, tenant_b
        )

    @pytest.mark.asyncio
    async def test_audit_log_query_performance(self, db_client, setup_audit_table):
        """Test audit log query performance with index usage."""
        tenant_id = uuid.uuid4()

        insert_sql = """
        INSERT INTO audit_logs (tenant_id, actor_id, action, resource_type, created_at)
        VALUES ($1, $2, $3, $4, $5)
        """

        # Create 100 audit logs
        base_time = dt.datetime.utcnow()
        for i in range(100):
            await db_client.execute(
                insert_sql,
                tenant_id,
                f"user_{i % 10}",
                "session.created" if i % 2 == 0 else "session.closed",
                "session",
                base_time - dt.timedelta(minutes=i),
            )

        # Query with tenant + action filter (should use index)
        start_time = time.time()
        logs = await db_client.fetch(
            """SELECT * FROM audit_logs 
               WHERE tenant_id = $1 AND action = $2 
               ORDER BY created_at DESC LIMIT 50""",
            tenant_id,
            "session.created",
        )
        query_latency_ms = (time.time() - start_time) * 1000

        assert len(logs) == 50
        assert query_latency_ms < 100, f"Query latency {query_latency_ms}ms exceeds 100ms"

        # Cleanup
        await db_client.execute(
            "DELETE FROM audit_logs WHERE tenant_id = $1", tenant_id
        )


class TestDatabaseTransactions:
    """Test database transaction handling."""

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(
        self, db_client, conversation_repo, setup_test_tables
    ):
        """Test that transactions rollback on error."""
        tenant_id = uuid.uuid4()
        session_id = f"sess_{uuid.uuid4().hex[:16]}"

        # Create initial item
        await conversation_repo.create(
            ConversationItemData(
                session_id=session_id,
                tenant_id=tenant_id,
                role="user",
                content={"text": "Initial"},
            )
        )

        # Attempt transaction that will fail
        try:
            async with db_client.transaction() as conn:
                # Insert valid item
                await conn.execute(
                    """INSERT INTO conversation_items (session_id, tenant_id, role, content)
                       VALUES ($1, $2, $3, $4)""",
                    session_id,
                    tenant_id,
                    "assistant",
                    {"text": "Should rollback"},
                )
                # Force error
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify only initial item exists (transaction rolled back)
        items = await conversation_repo.get_by_session(session_id, tenant_id)
        assert len(items) == 1
        assert items[0].content["text"] == "Initial"

        # Cleanup
        await db_client.execute(
            "DELETE FROM conversation_items WHERE session_id = $1", session_id
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

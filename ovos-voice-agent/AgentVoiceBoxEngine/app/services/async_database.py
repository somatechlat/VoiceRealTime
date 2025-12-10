"""Async database client using asyncpg for worker services.

This module provides async PostgreSQL access for:
- STT/TTS/LLM workers that need to persist data
- Redis-to-PostgreSQL overflow for conversation items
- Async audit logging from workers

Requirements: 13.1, 13.3, 13.5
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    import asyncpg
    from asyncpg import Connection, Pool, Record

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    Pool = None
    Connection = None
    Record = None
    logger.warning("asyncpg not installed - async database unavailable")


@dataclass
class AsyncDatabaseConfig:
    """Database connection configuration for asyncpg."""

    host: str = "localhost"
    port: int = 5432
    database: str = "voice_agent"
    user: str = "voice_agent"
    password: str = "voice_agent"
    min_connections: int = 2
    max_connections: int = 10
    command_timeout: float = 60.0
    statement_cache_size: int = 100

    @classmethod
    def from_env(cls) -> "AsyncDatabaseConfig":
        uri = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URI", "")
        if uri:
            return cls.from_uri(uri)
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "voice_agent"),
            user=os.getenv("DB_USER", "voice_agent"),
            password=os.getenv("DB_PASSWORD", "voice_agent"),
        )

    @classmethod
    def from_uri(cls, uri: str) -> "AsyncDatabaseConfig":
        uri = uri.replace("+psycopg", "").replace("+asyncpg", "")
        parsed = urlparse(uri)
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=(parsed.path or "/voice_agent").lstrip("/"),
            user=parsed.username or "voice_agent",
            password=parsed.password or "voice_agent",
        )


class AsyncDatabaseClient:
    """Async PostgreSQL client with connection pooling."""

    def __init__(self, config: AsyncDatabaseConfig):
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg is required but not installed")
        self.config = config
        self._pool: Optional[Pool] = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
            command_timeout=self.config.command_timeout,
            statement_cache_size=self.config.statement_cache_size,
            init=self._init_connection,
        )
        logger.info("Async database pool created")

    async def _init_connection(self, conn: Connection) -> None:
        await conn.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )
        await conn.set_type_codec(
            "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Async database pool closed")

    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Connection]:
        if self._pool is None:
            raise RuntimeError("Database pool not initialized")
        async with self._pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Connection]:
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def execute(self, query: str, *args: Any, timeout: float = None) -> str:
        async with self.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)

    async def fetch(self, query: str, *args: Any, timeout: float = None) -> List[Record]:
        async with self.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(self, query: str, *args: Any, timeout: float = None) -> Optional[Record]:
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)

    async def fetchval(self, query: str, *args: Any, column: int = 0, timeout: float = None) -> Any:
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)

    async def health_check(self) -> bool:
        try:
            result = await self.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error("Database health check failed: %s", e)
            return False


@dataclass
class ConversationItemData:
    """Conversation item for PostgreSQL persistence."""

    session_id: str
    tenant_id: Optional[uuid.UUID] = None
    role: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[dt.datetime] = None
    id: Optional[int] = None


class AsyncConversationRepository:
    """Async repository for conversation item persistence."""

    def __init__(self, db: AsyncDatabaseClient):
        self.db = db

    async def create(self, item: ConversationItemData) -> ConversationItemData:
        query = """
            INSERT INTO conversation_items (session_id, tenant_id, role, content, created_at)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, session_id, tenant_id, role, content, created_at
        """
        record = await self.db.fetchrow(
            query,
            item.session_id,
            item.tenant_id,
            item.role,
            item.content,
            item.created_at or dt.datetime.utcnow(),
        )
        if record:
            return ConversationItemData(
                id=record["id"],
                session_id=record["session_id"],
                tenant_id=record["tenant_id"],
                role=record["role"],
                content=record["content"],
                created_at=record["created_at"],
            )
        return item

    async def create_batch(self, items: List[ConversationItemData]) -> int:
        if not items:
            return 0
        async with self.db.transaction() as conn:
            query = """INSERT INTO conversation_items (session_id, tenant_id, role, content, created_at) VALUES ($1, $2, $3, $4, $5)"""
            count = 0
            for item in items:
                await conn.execute(
                    query,
                    item.session_id,
                    item.tenant_id,
                    item.role,
                    item.content,
                    item.created_at or dt.datetime.utcnow(),
                )
                count += 1
            return count

    async def get_by_session(
        self, session_id: str, tenant_id: Optional[uuid.UUID] = None, limit: int = 100
    ) -> List[ConversationItemData]:
        if tenant_id:
            query = """SELECT id, session_id, tenant_id, role, content, created_at FROM conversation_items WHERE session_id = $1 AND tenant_id = $2 ORDER BY created_at ASC LIMIT $3"""
            records = await self.db.fetch(query, session_id, tenant_id, limit)
        else:
            query = """SELECT id, session_id, tenant_id, role, content, created_at FROM conversation_items WHERE session_id = $1 ORDER BY created_at ASC LIMIT $2"""
            records = await self.db.fetch(query, session_id, limit)
        return [
            ConversationItemData(
                id=r["id"],
                session_id=r["session_id"],
                tenant_id=r["tenant_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in records
        ]

    async def count_by_session(self, session_id: str, tenant_id: Optional[uuid.UUID] = None) -> int:
        if tenant_id:
            return await self.db.fetchval(
                "SELECT COUNT(*) FROM conversation_items WHERE session_id = $1 AND tenant_id = $2",
                session_id,
                tenant_id,
            )
        return await self.db.fetchval(
            "SELECT COUNT(*) FROM conversation_items WHERE session_id = $1", session_id
        )


class ConversationOverflowHandler:
    """Handles Redis-to-PostgreSQL overflow for conversation items."""

    MAX_REDIS_ITEMS = 100
    OVERFLOW_BATCH_SIZE = 50

    def __init__(self, db: AsyncDatabaseClient, redis_client: Any):
        self.db = db
        self.redis = redis_client
        self._repo = AsyncConversationRepository(db)

    async def check_and_overflow(self, session_id: str, tenant_id: str) -> int:
        items_key = f"session:{tenant_id}:{session_id}:items"
        client = self.redis.client
        count = await client.llen(items_key)
        if count <= self.MAX_REDIS_ITEMS:
            return 0
        overflow_count = count - self.MAX_REDIS_ITEMS
        items_json = await client.lrange(items_key, 0, overflow_count - 1)
        if not items_json:
            return 0
        items_to_persist = []
        tenant_uuid = uuid.UUID(tenant_id) if tenant_id else None
        for item_str in items_json:
            try:
                item_data = json.loads(item_str)
                items_to_persist.append(
                    ConversationItemData(
                        session_id=session_id,
                        tenant_id=tenant_uuid,
                        role=item_data.get("role", ""),
                        content=item_data,
                        created_at=dt.datetime.utcnow(),
                    )
                )
            except json.JSONDecodeError:
                logger.warning("Failed to parse overflow item")
        if items_to_persist:
            persisted = await self._repo.create_batch(items_to_persist)
            await client.ltrim(items_key, overflow_count, -1)
            logger.info("Overflow: persisted %d items to PostgreSQL", persisted)
            return persisted
        return 0


_async_db_client: Optional[AsyncDatabaseClient] = None


async def get_async_database(config: Optional[AsyncDatabaseConfig] = None) -> AsyncDatabaseClient:
    global _async_db_client
    if _async_db_client is None:
        if config is None:
            config = AsyncDatabaseConfig.from_env()
        _async_db_client = AsyncDatabaseClient(config)
        await _async_db_client.connect()
    return _async_db_client


def get_database() -> AsyncDatabaseClient:
    global _async_db_client
    if _async_db_client is None:
        raise RuntimeError("Database not initialized")
    return _async_db_client


async def close_async_database() -> None:
    global _async_db_client
    if _async_db_client is not None:
        await _async_db_client.close()
        _async_db_client = None


__all__ = [
    "ASYNCPG_AVAILABLE",
    "AsyncDatabaseConfig",
    "AsyncDatabaseClient",
    "ConversationItemData",
    "AsyncConversationRepository",
    "ConversationOverflowHandler",
    "get_async_database",
    "get_database",
    "close_async_database",
]

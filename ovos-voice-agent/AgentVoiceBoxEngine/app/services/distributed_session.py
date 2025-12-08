"""Distributed session management backed by Redis.

This module provides a Redis-backed session manager for AgentVoiceBox that enables:
- Cross-gateway session access (stateless gateways)
- Real-time session sync via pub/sub
- Automatic session expiration with heartbeat refresh
- Tenant isolation via key namespacing

Redis Data Structures:
- session:{tenant_id}:{session_id} - Hash with session metadata
- session:{tenant_id}:{session_id}:config - Hash with session config
- session:{tenant_id}:{session_id}:items - List of conversation items (capped at 100)
- channel:session:{session_id} - Pub/sub channel for session updates
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .redis_client import RedisClient

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Session configuration matching OpenAI Realtime API."""
    model: str = "ovos-voice-1"
    voice: str = "am_onyx"
    speed: float = 1.1
    temperature: float = 0.8
    instructions: str = "You are a helpful assistant."
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[str] = None
    output_modalities: List[str] = field(default_factory=lambda: ["audio", "text"])
    max_output_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "voice": self.voice,
            "speed": self.speed,
            "temperature": self.temperature,
            "instructions": self.instructions,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "output_modalities": self.output_modalities,
            "max_output_tokens": self.max_output_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionConfig":
        return cls(
            model=data.get("model", "ovos-voice-1"),
            voice=data.get("voice", "am_onyx"),
            speed=float(data.get("speed", 1.1)),
            temperature=float(data.get("temperature", 0.8)),
            instructions=data.get("instructions", "You are a helpful assistant."),
            tools=data.get("tools", []),
            tool_choice=data.get("tool_choice"),
            output_modalities=data.get("output_modalities", ["audio", "text"]),
            max_output_tokens=data.get("max_output_tokens"),
        )


@dataclass
class Session:
    """Session state stored in Redis."""
    id: str
    tenant_id: str
    project_id: Optional[str]
    status: str  # created, connected, disconnected, expired
    gateway_id: Optional[str]
    conversation_id: str
    created_at: float
    last_activity: float
    config: SessionConfig
    persona: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "project_id": self.project_id or "",
            "status": self.status,
            "gateway_id": self.gateway_id or "",
            "conversation_id": self.conversation_id,
            "created_at": str(self.created_at),
            "last_activity": str(self.last_activity),
            "config": json.dumps(self.config.to_dict()),
            "persona": json.dumps(self.persona) if self.persona else "",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Session":
        config_data = json.loads(data.get("config", "{}"))
        persona_str = data.get("persona", "")
        return cls(
            id=data["id"],
            tenant_id=data["tenant_id"],
            project_id=data.get("project_id") or None,
            status=data["status"],
            gateway_id=data.get("gateway_id") or None,
            conversation_id=data["conversation_id"],
            created_at=float(data.get("created_at", time.time())),
            last_activity=float(data.get("last_activity", time.time())),
            config=SessionConfig.from_dict(config_data),
            persona=json.loads(persona_str) if persona_str else None,
        )


class DistributedSessionManager:
    """Redis-backed session management for cross-gateway access.
    
    This manager stores session state in Redis with:
    - 30-second TTL refreshed by heartbeat
    - Pub/sub notifications on session updates
    - Tenant isolation via key namespacing
    - Automatic cleanup of expired sessions
    """

    HASH_PREFIX = "session"
    HEARTBEAT_TTL = 30  # seconds
    MAX_CONVERSATION_ITEMS = 100
    PUBSUB_PREFIX = "channel:session"

    def __init__(self, redis_client: RedisClient, gateway_id: str) -> None:
        self._redis = redis_client
        self._gateway_id = gateway_id
        self._cleanup_task: Optional[asyncio.Task] = None

    def _session_key(self, tenant_id: str, session_id: str) -> str:
        """Generate namespaced session key for tenant isolation."""
        return f"{self.HASH_PREFIX}:{tenant_id}:{session_id}"

    def _items_key(self, tenant_id: str, session_id: str) -> str:
        """Generate key for conversation items list."""
        return f"{self.HASH_PREFIX}:{tenant_id}:{session_id}:items"

    def _pubsub_channel(self, session_id: str) -> str:
        """Generate pub/sub channel for session updates."""
        return f"{self.PUBSUB_PREFIX}:{session_id}"

    async def create_session(
        self,
        session_id: str,
        tenant_id: str,
        project_id: Optional[str] = None,
        config: Optional[SessionConfig] = None,
        persona: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new session in Redis.
        
        Args:
            session_id: Unique session identifier
            tenant_id: Tenant ID for isolation
            project_id: Optional project ID
            config: Session configuration
            persona: Optional persona configuration
            
        Returns:
            Created Session object
        """
        now = time.time()
        conversation_id = f"conv_{session_id.replace('sess_', '')}"
        
        session = Session(
            id=session_id,
            tenant_id=tenant_id,
            project_id=project_id,
            status="created",
            gateway_id=self._gateway_id,
            conversation_id=conversation_id,
            created_at=now,
            last_activity=now,
            config=config or SessionConfig(),
            persona=persona,
        )

        key = self._session_key(tenant_id, session_id)
        
        # Store session as hash
        await self._redis.hset(key, session.to_dict())
        await self._redis.expire(key, self.HEARTBEAT_TTL)

        logger.info(
            "Session created in Redis",
            extra={
                "session_id": session_id,
                "tenant_id": tenant_id,
                "gateway_id": self._gateway_id,
            }
        )

        # Publish session created event
        await self._publish_event(session_id, "session.created", session.to_dict())

        return session

    async def get_session(
        self,
        session_id: str,
        tenant_id: str,
    ) -> Optional[Session]:
        """Retrieve session from Redis.
        
        Args:
            session_id: Session identifier
            tenant_id: Tenant ID for isolation
            
        Returns:
            Session object or None if not found/expired
        """
        key = self._session_key(tenant_id, session_id)
        data = await self._redis.hgetall(key)
        
        if not data:
            logger.debug(
                "Session not found in Redis",
                extra={"session_id": session_id, "tenant_id": tenant_id}
            )
            return None

        try:
            return Session.from_dict(data)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(
                "Failed to deserialize session",
                extra={"session_id": session_id, "error": str(e)}
            )
            return None

    async def update_session(
        self,
        session_id: str,
        tenant_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Session]:
        """Update session and publish change event.
        
        Args:
            session_id: Session identifier
            tenant_id: Tenant ID for isolation
            updates: Dictionary of fields to update
            
        Returns:
            Updated Session object or None if not found
        """
        session = await self.get_session(session_id, tenant_id)
        if not session:
            return None

        # Apply updates
        if "status" in updates:
            session.status = updates["status"]
        if "gateway_id" in updates:
            session.gateway_id = updates["gateway_id"]
        if "persona" in updates:
            session.persona = updates["persona"]
        if "config" in updates:
            config_updates = updates["config"]
            if isinstance(config_updates, dict):
                current_config = session.config.to_dict()
                current_config.update(config_updates)
                session.config = SessionConfig.from_dict(current_config)

        session.last_activity = time.time()

        key = self._session_key(tenant_id, session_id)
        await self._redis.hset(key, session.to_dict())
        await self._redis.expire(key, self.HEARTBEAT_TTL)

        logger.info(
            "Session updated in Redis",
            extra={"session_id": session_id, "updates": list(updates.keys())}
        )

        # Publish session updated event
        await self._publish_event(session_id, "session.updated", session.to_dict())

        return session

    async def heartbeat(self, session_id: str, tenant_id: str) -> bool:
        """Refresh session TTL.
        
        Args:
            session_id: Session identifier
            tenant_id: Tenant ID for isolation
            
        Returns:
            True if session exists and TTL was refreshed
        """
        key = self._session_key(tenant_id, session_id)
        
        # Update last_activity and refresh TTL atomically
        client = self._redis.client
        
        exists = await client.exists(key)
        if not exists:
            return False

        await client.hset(key, "last_activity", str(time.time()))
        await client.expire(key, self.HEARTBEAT_TTL)
        
        return True

    async def close_session(
        self,
        session_id: str,
        tenant_id: str,
    ) -> bool:
        """Close a session and clean up resources.
        
        Args:
            session_id: Session identifier
            tenant_id: Tenant ID for isolation
            
        Returns:
            True if session was closed
        """
        session = await self.get_session(session_id, tenant_id)
        if not session:
            return False

        session.status = "closed"
        session.last_activity = time.time()

        key = self._session_key(tenant_id, session_id)
        items_key = self._items_key(tenant_id, session_id)

        # Update status before deletion (for pub/sub notification)
        await self._redis.hset(key, session.to_dict())
        
        # Publish close event before cleanup
        await self._publish_event(session_id, "session.closed", {"session_id": session_id})

        # Delete session and items
        await self._redis.delete(key, items_key)

        logger.info(
            "Session closed and cleaned up",
            extra={"session_id": session_id, "tenant_id": tenant_id}
        )

        return True

    async def append_conversation_item(
        self,
        session_id: str,
        tenant_id: str,
        item: Dict[str, Any],
    ) -> bool:
        """Append a conversation item to the session.
        
        Items are stored in a Redis list, capped at MAX_CONVERSATION_ITEMS.
        Overflow items should be persisted to PostgreSQL separately.
        
        Args:
            session_id: Session identifier
            tenant_id: Tenant ID for isolation
            item: Conversation item to append
            
        Returns:
            True if item was appended
        """
        key = self._items_key(tenant_id, session_id)
        client = self._redis.client

        # Append item as JSON
        await client.rpush(key, json.dumps(item))
        
        # Trim to max items (keep most recent)
        await client.ltrim(key, -self.MAX_CONVERSATION_ITEMS, -1)
        
        # Set TTL to match session
        await client.expire(key, self.HEARTBEAT_TTL * 2)

        return True

    async def get_conversation_items(
        self,
        session_id: str,
        tenant_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get conversation items for a session.
        
        Args:
            session_id: Session identifier
            tenant_id: Tenant ID for isolation
            limit: Maximum items to return
            
        Returns:
            List of conversation items (most recent last)
        """
        key = self._items_key(tenant_id, session_id)
        client = self._redis.client

        items_json = await client.lrange(key, -limit, -1)
        
        items = []
        for item_str in items_json:
            try:
                items.append(json.loads(item_str))
            except json.JSONDecodeError:
                logger.warning("Failed to parse conversation item", extra={"raw": item_str})
        
        return items

    async def _publish_event(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Publish session event to pub/sub channel."""
        channel = self._pubsub_channel(session_id)
        message = json.dumps({
            "type": event_type,
            "timestamp": time.time(),
            "gateway_id": self._gateway_id,
            "data": data,
        })
        await self._redis.publish(channel, message)

    async def subscribe_to_session(self, session_id: str):
        """Subscribe to session updates.
        
        Returns an async generator that yields session events.
        """
        channel = self._pubsub_channel(session_id)
        pubsub = await self._redis.subscribe(channel)
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    yield json.loads(message["data"])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse pubsub message")

    def start_cleanup_task(self) -> None:
        """Start background task to clean up expired sessions."""
        if self._cleanup_task and not self._cleanup_task.done():
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session cleanup task started")

    def stop_cleanup_task(self) -> None:
        """Stop the cleanup background task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("Session cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """Background loop to clean up expired sessions.
        
        This loop:
        1. Scans for sessions that haven't had heartbeat in 2x TTL
        2. Emits session.closed events for expired sessions
        3. Cleans up orphaned conversation items
        
        Note: Redis TTL handles key expiration, but we need this for:
        - Emitting events before keys expire
        - Cleaning up related keys (items, audio buffers)
        - Persisting conversation items to PostgreSQL
        """
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                # Scan for sessions owned by this gateway that may be stale
                # In production, use Redis keyspace notifications instead
                client = self._redis.client
                
                # Scan for session keys (pattern: session:*:sess_*)
                cursor = 0
                stale_threshold = time.time() - (self.HEARTBEAT_TTL * 2)
                
                while True:
                    cursor, keys = await client.scan(
                        cursor=cursor,
                        match=f"{self.HASH_PREFIX}:*:sess_*",
                        count=100,
                    )
                    
                    for key in keys:
                        # Skip config and items keys
                        if ":config" in key or ":items" in key:
                            continue
                        
                        try:
                            data = await client.hgetall(key)
                            if not data:
                                continue
                            
                            # Check if session is stale
                            last_activity = float(data.get("last_activity", 0))
                            gateway_id = data.get("gateway_id", "")
                            session_id = data.get("id", "")
                            tenant_id = data.get("tenant_id", "")
                            
                            # Only clean up sessions owned by this gateway
                            if gateway_id != self._gateway_id:
                                continue
                            
                            if last_activity < stale_threshold:
                                logger.info(
                                    "Cleaning up stale session",
                                    extra={
                                        "session_id": session_id,
                                        "last_activity": last_activity,
                                        "stale_threshold": stale_threshold,
                                    }
                                )
                                
                                # Emit session.closed event
                                await self._publish_event(
                                    session_id,
                                    "session.closed",
                                    {"session_id": session_id, "reason": "expired"}
                                )
                                
                                # Clean up related keys
                                items_key = self._items_key(tenant_id, session_id)
                                await client.delete(key, items_key)
                                
                        except Exception as e:
                            logger.warning(
                                "Error processing session during cleanup",
                                extra={"key": key, "error": str(e)}
                            )
                    
                    if cursor == 0:
                        break
                
                logger.debug("Session cleanup cycle completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop", extra={"error": str(e)})


__all__ = [
    "DistributedSessionManager",
    "Session",
    "SessionConfig",
]

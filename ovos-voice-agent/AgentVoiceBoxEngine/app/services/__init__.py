"""AgentVoiceBox Engine services."""

from .redis_client import (
    RedisClient,
    get_redis_client,
    init_redis_client,
    close_redis_client,
)
from .distributed_session import (
    DistributedSessionManager,
    Session,
    SessionConfig,
)
from .distributed_rate_limiter import (
    DistributedRateLimiter,
    RateLimitConfig,
    RateLimitResult,
    count_tokens,
)
from .connection_manager import (
    ConnectionManager,
    ConnectionInfo,
    get_connection_manager,
    init_connection_manager,
    setup_signal_handlers,
)
from .redis_streams import (
    RedisStreamsClient,
    AudioSTTRequest,
    TTSRequest,
    TranscriptionResult,
    get_streams_client,
    init_streams_client,
)
from .session_service import SessionService
from .token_service import TokenService
from .kafka_client import KafkaFactory, kafka_producer
from .opa_client import OPAClient
from .api_key_service import (
    ARGON2_AVAILABLE,
    APIKeyInfo,
    APIKeyHasher,
    APIKeyService,
    generate_api_key,
)
from .ephemeral_token_service import (
    EphemeralToken,
    EphemeralTokenService,
)
from .tenant_context import (
    TenantContext,
    TenantIsolation,
    set_tenant_context,
    get_tenant_context,
    require_tenant_context,
    clear_tenant_context,
)
from .audit_service import (
    AuditAction,
    AuditEntry,
    AuditService,
    get_audit_service,
    init_audit_service,
    audit,
)

# Async database is optional (only for workers with asyncpg installed)
try:
    from .async_database import (
        ASYNCPG_AVAILABLE,
        AsyncDatabaseConfig,
        AsyncDatabaseClient,
        ConversationItemData,
        AsyncConversationRepository,
        ConversationOverflowHandler,
        get_async_database,
        close_async_database,
    )
except ImportError:
    ASYNCPG_AVAILABLE = False
    AsyncDatabaseConfig = None
    AsyncDatabaseClient = None
    ConversationItemData = None
    AsyncConversationRepository = None
    ConversationOverflowHandler = None
    get_async_database = None
    close_async_database = None

__all__ = [
    "RedisClient",
    "get_redis_client",
    "init_redis_client",
    "close_redis_client",
    "DistributedSessionManager",
    "Session",
    "SessionConfig",
    "DistributedRateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    "count_tokens",
    "ConnectionManager",
    "ConnectionInfo",
    "get_connection_manager",
    "init_connection_manager",
    "setup_signal_handlers",
    "RedisStreamsClient",
    "AudioSTTRequest",
    "TTSRequest",
    "TranscriptionResult",
    "get_streams_client",
    "init_streams_client",
    "SessionService",
    "TokenService",
    "KafkaFactory",
    "kafka_producer",
    "OPAClient",
    "ASYNCPG_AVAILABLE",
    "AsyncDatabaseConfig",
    "AsyncDatabaseClient",
    "ConversationItemData",
    "AsyncConversationRepository",
    "ConversationOverflowHandler",
    "get_async_database",
    "close_async_database",
    "ARGON2_AVAILABLE",
    "APIKeyInfo",
    "APIKeyHasher",
    "APIKeyService",
    "generate_api_key",
    "EphemeralToken",
    "EphemeralTokenService",
    "TenantContext",
    "TenantIsolation",
    "set_tenant_context",
    "get_tenant_context",
    "require_tenant_context",
    "clear_tenant_context",
    "AuditAction",
    "AuditEntry",
    "AuditService",
    "get_audit_service",
    "init_audit_service",
    "audit",
]

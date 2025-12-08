# AgentVoiceBox Production Architecture Design

## Overview

This document specifies the technical architecture for AgentVoiceBox, a distributed speech-to-speech platform designed to handle **1,000,000+ concurrent WebSocket connections** with sub-200ms end-to-end latency. The architecture follows a stateless gateway pattern with dedicated worker pools for CPU/GPU-intensive operations.

### Design Principles

1. **Stateless Gateways**: All session state lives in Redis, enabling horizontal scaling
2. **Separation of Concerns**: Gateway handles protocol, workers handle compute
3. **Backpressure Propagation**: Every component respects downstream capacity
4. **Graceful Degradation**: Partial failures don't cause total outages
5. **Observable by Default**: Every operation emits metrics and traces

---

## Architecture

### System Topology

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INTERNET                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         HAPROXY LOAD BALANCER CLUSTER                           │
│                                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                             │
│  │  HAProxy 1  │  │  HAProxy 2  │  │  HAProxy 3  │  (Active-Active with VRRP)  │
│  │  (Primary)  │  │  (Secondary)│  │  (Tertiary) │                             │
│  └─────────────┘  └─────────────┘  └─────────────┘                             │
│                                                                                 │
│  Features:                                                                      │
│  - WebSocket-aware routing (Upgrade header detection)                          │
│  - Consistent hashing on session_id for sticky sessions                        │
│  - Health checks every 2 seconds                                               │
│  - Connection draining on backend removal                                      │
│  - Rate limiting at edge (100 conn/sec per IP)                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   GATEWAY POD 1      │ │   GATEWAY POD 2      │ │   GATEWAY POD N      │
│                      │ │                      │ │                      │
│  ┌────────────────┐  │ │  ┌────────────────┐  │ │  ┌────────────────┐  │
│  │ FastAPI/ASGI   │  │ │  │ FastAPI/ASGI   │  │ │  │ FastAPI/ASGI   │  │
│  │ (uvicorn)      │  │ │  │ (uvicorn)      │  │ │  │ (uvicorn)      │  │
│  └────────────────┘  │ │  └────────────────┘  │ │  └────────────────┘  │
│                      │ │                      │ │                      │
│  Responsibilities:   │ │  Responsibilities:   │ │  Responsibilities:   │
│  - WebSocket mgmt    │ │  - WebSocket mgmt    │ │  - WebSocket mgmt    │
│  - Auth/AuthZ        │ │  - Auth/AuthZ        │ │  - Auth/AuthZ        │
│  - Protocol parsing  │ │  - Protocol parsing  │ │  - Protocol parsing  │
│  - Rate limiting     │ │  - Rate limiting     │ │  - Rate limiting     │
│  - Event routing     │ │  - Event routing     │ │  - Event routing     │
│                      │ │                      │ │                      │
│  Capacity: 50K conn  │ │  Capacity: 50K conn  │ │  Capacity: 50K conn  │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           REDIS CLUSTER (6 nodes)                               │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Sessions        │  │ Rate Limits     │  │ Pub/Sub         │                 │
│  │ (Hash slots     │  │ (Sorted sets    │  │ (Channels per   │                 │
│  │  0-5460)        │  │  with TTL)      │  │  session_id)    │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Audio Streams   │  │ Work Queues     │  │ Circuit State   │                 │
│  │ (Redis Streams  │  │ (Consumer       │  │ (Keys with      │                 │
│  │  per session)   │  │  groups)        │  │  TTL)           │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│                                                                                 │
│  Configuration:                                                                 │
│  - 3 masters + 3 replicas                                                       │
│  - maxmemory: 64GB per node                                                     │
│  - maxmemory-policy: volatile-lru                                               │
│  - cluster-require-full-coverage: no (partial availability)                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  STT WORKER     │     │  LLM WORKER     │     │  TTS WORKER     │
│  POOL           │     │  POOL           │     │  POOL           │
│                 │     │                 │     │                 │
│  ┌───────────┐  │     │  ┌───────────┐  │     │  ┌───────────┐  │
│  │ Worker 1  │  │     │  │ Worker 1  │  │     │  │ Worker 1  │  │
│  │ (GPU)     │  │     │  │ (CPU)     │  │     │  │ (GPU)     │  │
│  ├───────────┤  │     │  ├───────────┤  │     │  ├───────────┤  │
│  │ Worker 2  │  │     │  │ Worker 2  │  │     │  │ Worker 2  │  │
│  │ (GPU)     │  │     │  │ (CPU)     │  │     │  │ (GPU)     │  │
│  ├───────────┤  │     │  ├───────────┤  │     │  ├───────────┤  │
│  │ Worker N  │  │     │  │ Worker N  │  │     │  │ Worker N  │  │
│  │ (GPU)     │  │     │  │ (CPU)     │  │     │  │ (GPU)     │  │
│  └───────────┘  │     │  └───────────┘  │     │  └───────────┘  │
│                 │     │                 │     │                 │
│  Engine:        │     │  Providers:     │     │  Engine:        │
│  Faster-Whisper │     │  - OpenAI       │     │  Kokoro ONNX    │
│                 │     │  - Groq         │     │                 │
│  Batch: 10/GPU  │     │  - Ollama       │     │  Streaming: Yes │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         POSTGRESQL CLUSTER                                      │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Primary         │  │ Replica 1       │  │ Replica 2       │                 │
│  │ (Write)         │  │ (Read)          │  │ (Read)          │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│                                                                                 │
│  Tables (partitioned by tenant_id):                                             │
│  - sessions: Session metadata and config                                        │
│  - conversation_items: Message history (JSONB)                                  │
│  - audit_logs: Security and compliance events                                   │
│  - function_calls: Tool invocation records                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Components and Interfaces

### 1. Gateway Service

The gateway is the WebSocket termination point. It handles protocol translation, authentication, and event routing.

```python
# gateway/main.py - Core structure
class GatewayService:
    """
    Stateless WebSocket gateway.
    All session state is stored in Redis.
    """
    
    def __init__(self):
        self.redis: RedisCluster
        self.rate_limiter: DistributedRateLimiter
        self.event_router: EventRouter
        self.metrics: PrometheusMetrics
    
    async def handle_connection(self, websocket: WebSocket, session_id: str):
        """Handle a single WebSocket connection lifecycle."""
        pass
    
    async def route_event(self, session_id: str, event: dict):
        """Route incoming event to appropriate handler or worker."""
        pass
    
    async def publish_to_client(self, session_id: str, event: dict):
        """Send event to client via WebSocket."""
        pass
```

**Interfaces:**

| Interface | Protocol | Purpose |
|-----------|----------|---------|
| `/v1/realtime` | WebSocket | Client connections (OpenAI-compatible) |
| `/v1/realtime/sessions` | HTTP POST | Create session with config |
| `/v1/tts/voices` | HTTP GET | List available voices |
| `/health` | HTTP GET | Health check with dependency status |
| `/metrics` | HTTP GET | Prometheus metrics |

### 2. Session Manager

Manages distributed session state in Redis.

```python
# gateway/session_manager.py
class DistributedSessionManager:
    """
    Redis-backed session management.
    Supports cross-gateway session access.
    """
    
    HASH_PREFIX = "session:"
    HEARTBEAT_TTL = 30  # seconds
    
    async def create_session(self, session_id: str, config: SessionConfig) -> Session:
        """Create new session in Redis."""
        pass
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session from Redis."""
        pass
    
    async def update_session(self, session_id: str, updates: dict) -> Session:
        """Update session and publish change event."""
        pass
    
    async def heartbeat(self, session_id: str):
        """Refresh session TTL."""
        pass
    
    async def cleanup_expired(self):
        """Background task to clean up expired sessions."""
        pass
```

**Redis Data Structures:**

```
# Session Hash
session:{session_id} = {
    "id": "sess_abc123",
    "status": "connected",
    "gateway_id": "gateway-pod-1",
    "created_at": 1704067200,
    "config": "{...json...}",
    "conversation_id": "conv_xyz789"
}
TTL: 30 seconds (refreshed by heartbeat)

# Session Config (separate for atomic updates)
session:{session_id}:config = {
    "voice": "am_onyx",
    "speed": 1.1,
    "temperature": 0.8,
    "instructions": "You are a helpful assistant",
    "tools": "[...json...]"
}

# Conversation Items (List)
session:{session_id}:items = [
    "{item1_json}",
    "{item2_json}",
    ...
]
```

### 3. Distributed Rate Limiter

Token bucket algorithm implemented with Redis Lua scripts for atomic operations.

```python
# gateway/rate_limiter.py
class DistributedRateLimiter:
    """
    Redis-based sliding window rate limiter.
    Uses Lua scripts for atomic check-and-decrement.
    """
    
    async def check_and_consume(
        self, 
        key: str, 
        tokens: int = 1,
        requests: int = 1
    ) -> RateLimitResult:
        """
        Atomically check limits and consume quota.
        Returns remaining quota and reset time.
        """
        pass
    
    async def get_limits(self, key: str) -> RateLimitInfo:
        """Get current limit status without consuming."""
        pass
```

**Lua Script for Atomic Rate Limiting:**

```lua
-- rate_limit.lua
-- KEYS[1] = rate limit key
-- ARGV[1] = current timestamp (ms)
-- ARGV[2] = window size (ms)
-- ARGV[3] = max requests
-- ARGV[4] = max tokens
-- ARGV[5] = requests to consume
-- ARGV[6] = tokens to consume

local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local max_tokens = tonumber(ARGV[4])
local req_consume = tonumber(ARGV[5])
local tok_consume = tonumber(ARGV[6])

-- Remove expired entries
local window_start = now - window
redis.call('ZREMRANGEBYSCORE', key .. ':req', '-inf', window_start)
redis.call('ZREMRANGEBYSCORE', key .. ':tok', '-inf', window_start)

-- Count current usage
local req_count = redis.call('ZCARD', key .. ':req')
local tok_count = redis.call('ZCARD', key .. ':tok')

-- Check limits
if req_count + req_consume > max_requests then
    return {0, max_requests - req_count, max_tokens - tok_count, window}
end
if tok_count + tok_consume > max_tokens then
    return {0, max_requests - req_count, max_tokens - tok_count, window}
end

-- Consume quota
for i = 1, req_consume do
    redis.call('ZADD', key .. ':req', now, now .. ':' .. i)
end
for i = 1, tok_consume do
    redis.call('ZADD', key .. ':tok', now, now .. ':' .. i)
end

-- Set expiry
redis.call('EXPIRE', key .. ':req', math.ceil(window / 1000) + 1)
redis.call('EXPIRE', key .. ':tok', math.ceil(window / 1000) + 1)

return {1, max_requests - req_count - req_consume, max_tokens - tok_count - tok_consume, window}
```

### 4. STT Worker

Dedicated worker for speech-to-text processing using Faster-Whisper.

```python
# workers/stt_worker.py
class STTWorker:
    """
    GPU-accelerated STT worker.
    Consumes audio from Redis Streams, publishes transcriptions.
    """
    
    def __init__(self):
        self.model: WhisperModel
        self.consumer_group = "stt-workers"
        self.stream_name = "audio:stt"
    
    async def run(self):
        """Main worker loop."""
        while True:
            # Read from stream with consumer group
            messages = await self.redis.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.worker_id,
                streams={self.stream_name: ">"},
                count=10,
                block=1000
            )
            
            for message in messages:
                await self.process_audio(message)
                await self.redis.xack(self.stream_name, self.consumer_group, message.id)
    
    async def process_audio(self, message: StreamMessage):
        """Transcribe audio and publish result."""
        session_id = message.data["session_id"]
        audio_data = base64.b64decode(message.data["audio"])
        
        # Transcribe
        segments, info = self.model.transcribe(audio_data)
        text = " ".join(s.text for s in segments)
        
        # Publish result
        await self.redis.publish(f"transcription:{session_id}", json.dumps({
            "type": "transcription.completed",
            "session_id": session_id,
            "text": text,
            "confidence": info.language_probability,
            "language": info.language
        }))
```

### 5. TTS Worker

Dedicated worker for text-to-speech synthesis using Kokoro ONNX.

```python
# workers/tts_worker.py
class TTSWorker:
    """
    GPU-accelerated TTS worker with streaming output.
    Consumes synthesis requests, streams audio chunks.
    """
    
    def __init__(self):
        self.kokoro: KokoroEngine
        self.consumer_group = "tts-workers"
        self.stream_name = "tts:requests"
    
    async def process_synthesis(self, message: StreamMessage):
        """Synthesize text and stream audio chunks."""
        session_id = message.data["session_id"]
        text = message.data["text"]
        voice = message.data.get("voice", "am_onyx")
        speed = float(message.data.get("speed", 1.1))
        
        # Stream audio chunks as they're generated
        async for audio_chunk, sample_rate in self.kokoro.create_stream(
            text=text,
            voice=voice,
            speed=speed
        ):
            # Check for cancellation
            if await self.is_cancelled(session_id):
                break
            
            # Publish chunk to session's audio stream
            chunk_b64 = base64.b64encode(audio_chunk).decode()
            await self.redis.xadd(
                f"audio:out:{session_id}",
                {"chunk": chunk_b64, "sample_rate": sample_rate}
            )
        
        # Signal completion
        await self.redis.publish(f"tts:{session_id}", json.dumps({
            "type": "tts.completed",
            "session_id": session_id
        }))
```

### 6. LLM Worker

Handles LLM inference with provider abstraction and failover.

```python
# workers/llm_worker.py
class LLMWorker:
    """
    LLM inference worker with multi-provider support.
    Implements circuit breaker for provider failover.
    """
    
    def __init__(self):
        self.providers: List[LLMProvider] = [
            OpenAIProvider(),
            GroqProvider(),
            OllamaProvider()
        ]
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    async def generate(self, request: LLMRequest) -> AsyncIterator[str]:
        """Generate response with automatic failover."""
        for provider in self.providers:
            breaker = self.circuit_breakers[provider.name]
            
            if breaker.is_open:
                continue
            
            try:
                async for token in provider.generate_stream(request):
                    yield token
                breaker.record_success()
                return
            except Exception as e:
                breaker.record_failure()
                logger.warning(f"Provider {provider.name} failed: {e}")
        
        raise AllProvidersFailedError("All LLM providers unavailable")
```

---

## Data Models

### Session Model

```python
@dataclass
class Session:
    id: str                          # sess_{uuid}
    status: SessionStatus            # created, connected, disconnected
    gateway_id: Optional[str]        # Which gateway owns this connection
    created_at: datetime
    expires_at: Optional[datetime]
    
    # Configuration
    model: str = "ovos-voice-1"
    voice: str = "am_onyx"
    speed: float = 1.1
    temperature: float = 0.8
    instructions: str = "You are a helpful assistant."
    tools: List[Tool] = field(default_factory=list)
    
    # State
    conversation_id: str             # conv_{uuid}
    input_audio_buffer: bytes = b""
    is_speaking: bool = False
    current_response_id: Optional[str] = None

class SessionStatus(Enum):
    CREATED = "created"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    EXPIRED = "expired"
```

### Conversation Item Model

```python
@dataclass
class ConversationItem:
    id: str                          # item_{uuid}
    type: ItemType                   # message, function_call, function_call_output
    status: ItemStatus               # in_progress, completed, cancelled
    role: Role                       # user, assistant, system
    content: List[ContentPart]
    created_at: datetime
    
    # For function calls
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None

class ContentPart:
    type: ContentType                # input_text, input_audio, output_text, audio
    text: Optional[str] = None
    transcript: Optional[str] = None
    audio: Optional[str] = None      # base64 encoded
```

### Rate Limit Model

```python
@dataclass
class RateLimitConfig:
    requests_per_minute: int = 100
    tokens_per_minute: int = 100000
    
@dataclass
class RateLimitResult:
    allowed: bool
    requests_remaining: int
    tokens_remaining: int
    reset_seconds: float
```

---


## Error Handling

### Error Taxonomy

All errors follow the OpenAI error format for API compatibility:

```python
class ErrorType(Enum):
    INVALID_REQUEST_ERROR = "invalid_request_error"      # Malformed request
    AUTHENTICATION_ERROR = "authentication_error"        # Invalid/expired token
    PERMISSION_ERROR = "permission_error"                # Insufficient permissions
    NOT_FOUND_ERROR = "not_found_error"                  # Resource doesn't exist
    RATE_LIMIT_ERROR = "rate_limit_error"                # Quota exceeded
    API_ERROR = "api_error"                              # Internal server error
    OVERLOADED_ERROR = "overloaded_error"                # System at capacity
    TIMEOUT_ERROR = "timeout_error"                      # Operation timed out

@dataclass
class ErrorEvent:
    type: str = "error"
    error: ErrorDetail

@dataclass  
class ErrorDetail:
    type: ErrorType
    code: str
    message: str
    param: Optional[str] = None
    event_id: str = field(default_factory=lambda: f"event_{uuid.uuid4().hex[:16]}")
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    """
    Prevents cascade failures by failing fast when dependencies are unhealthy.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing fast, requests rejected immediately
    - HALF_OPEN: Testing if dependency recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3
    ):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        
    async def call(self, func: Callable, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise
```

### Graceful Degradation Modes

| Failure | Degraded Mode | User Impact |
|---------|---------------|-------------|
| LLM unavailable | Echo mode with apology | "I'm having trouble thinking right now. Please try again." |
| TTS unavailable | Text-only responses | Transcript sent without audio |
| STT unavailable | Manual text input only | User must type instead of speak |
| Redis unavailable | Reject new connections | Existing sessions continue in-memory |
| PostgreSQL unavailable | Skip persistence | Conversations not saved, core function works |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Session State Consistency
*For any* session, if a gateway writes state to Redis and another gateway reads it, the read SHALL return the most recent write within 100ms.
**Validates: Requirements 2.1, 2.2**

### Property 2: Rate Limit Accuracy
*For any* client making requests across multiple gateway instances, the total requests allowed SHALL NOT exceed the configured limit (100/min) by more than 5%.
**Validates: Requirements 3.1, 3.2, 3.3**

### Property 3: Audio Chunk Ordering
*For any* audio stream, chunks SHALL be delivered to the client in the exact order they were generated by the TTS worker, with no gaps or duplicates.
**Validates: Requirements 4.5, 6.3**

### Property 4: Transcription Delivery
*For any* completed transcription, the result SHALL be delivered to the originating session within 100ms of STT worker completion.
**Validates: Requirements 4.3, 5.3**

### Property 5: Cancel Propagation
*For any* cancel request, all in-flight work (TTS synthesis, LLM generation) SHALL stop within 50ms of the cancel being received.
**Validates: Requirements 6.5**

### Property 6: Connection Draining
*For any* gateway shutdown, all existing connections SHALL be gracefully closed with proper cleanup within 30 seconds.
**Validates: Requirements 1.2**

### Property 7: Heartbeat Liveness
*For any* active session, if heartbeats stop for 30 seconds, the session SHALL be marked expired and resources cleaned up.
**Validates: Requirements 2.3, 2.4**

### Property 8: Circuit Breaker Recovery
*For any* circuit breaker that opens due to failures, it SHALL attempt recovery after the configured timeout and close if the dependency is healthy.
**Validates: Requirements 12.3, 12.4, 12.5**

### Property 9: Message Persistence
*For any* conversation item created, it SHALL be persisted to PostgreSQL within 1 second and be queryable thereafter.
**Validates: Requirements 10.1, 10.3**

### Property 10: Authentication Enforcement
*For any* WebSocket connection attempt without a valid token, the connection SHALL be rejected with authentication_error before any session state is created.
**Validates: Requirements 11.1, 11.3**

---

## Testing Strategy

### Dual Testing Approach

The system requires both unit tests and property-based tests:

- **Unit tests**: Verify specific examples, edge cases, and integration points
- **Property-based tests**: Verify universal properties hold across all inputs using Hypothesis

### Property-Based Testing Framework

**Library**: Hypothesis (Python)
**Minimum iterations**: 100 per property test

### Test Categories

#### 1. Gateway Unit Tests
```python
# tests/gateway/test_session_manager.py
class TestSessionManager:
    async def test_create_session_stores_in_redis(self):
        """Verify session creation persists to Redis."""
        pass
    
    async def test_get_session_returns_none_for_missing(self):
        """Verify missing session returns None, not error."""
        pass
    
    async def test_heartbeat_refreshes_ttl(self):
        """Verify heartbeat extends session lifetime."""
        pass
```

#### 2. Rate Limiter Property Tests
```python
# tests/gateway/test_rate_limiter_properties.py
# **Feature: production-voice-agent, Property 2: Rate Limit Accuracy**
# **Validates: Requirements 3.1, 3.2, 3.3**

from hypothesis import given, strategies as st

class TestRateLimiterProperties:
    @given(
        requests=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=200),
        num_gateways=st.integers(min_value=1, max_value=5)
    )
    async def test_distributed_rate_limit_accuracy(self, requests, num_gateways):
        """
        Property: Total allowed requests across all gateways SHALL NOT 
        exceed limit by more than 5%.
        """
        pass
```

#### 3. Audio Pipeline Property Tests
```python
# tests/workers/test_audio_properties.py
# **Feature: production-voice-agent, Property 3: Audio Chunk Ordering**
# **Validates: Requirements 4.5, 6.3**

class TestAudioPipelineProperties:
    @given(
        text=st.text(min_size=1, max_size=1000),
        voice=st.sampled_from(["am_onyx", "af_bella", "am_adam"])
    )
    async def test_audio_chunks_ordered(self, text, voice):
        """
        Property: Audio chunks SHALL be delivered in generation order
        with no gaps or duplicates.
        """
        pass
```

#### 4. Session Consistency Property Tests
```python
# tests/gateway/test_session_properties.py
# **Feature: production-voice-agent, Property 1: Session State Consistency**
# **Validates: Requirements 2.1, 2.2**

class TestSessionConsistencyProperties:
    @given(
        updates=st.lists(st.fixed_dictionaries({
            "voice": st.sampled_from(["am_onyx", "af_bella"]),
            "speed": st.floats(min_value=0.5, max_value=2.0)
        }), min_size=1, max_size=50)
    )
    async def test_concurrent_updates_consistent(self, updates):
        """
        Property: Concurrent updates from multiple gateways SHALL
        result in consistent final state.
        """
        pass
```

#### 5. Integration Tests
```python
# tests/integration/test_full_pipeline.py
class TestFullPipeline:
    async def test_speech_to_speech_roundtrip(self):
        """End-to-end test: audio in -> transcription -> LLM -> TTS -> audio out."""
        pass
    
    async def test_cancel_stops_all_work(self):
        """Verify cancel propagates to all workers."""
        pass
    
    async def test_gateway_failover(self):
        """Verify session survives gateway restart."""
        pass
```

#### 6. Load Tests
```python
# tests/load/test_scale.py
class TestScale:
    async def test_1000_concurrent_connections(self):
        """Verify single gateway handles 1000 connections."""
        pass
    
    async def test_rate_limit_under_load(self):
        """Verify rate limiting works correctly under high load."""
        pass
```

### Test Infrastructure

| Component | Tool | Purpose |
|-----------|------|---------|
| Unit Tests | pytest + pytest-asyncio | Fast, isolated tests |
| Property Tests | Hypothesis | Randomized input testing |
| Integration Tests | pytest + testcontainers | Real Redis/Postgres |
| Load Tests | Locust | Concurrent connection testing |
| Chaos Tests | chaos-mesh | Failure injection |

---

## Deployment Architecture

### Kubernetes Resources

```yaml
# k8s/gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentvoicebox-gateway
spec:
  replicas: 20
  selector:
    matchLabels:
      app: gateway
  template:
    spec:
      containers:
      - name: gateway
        image: agentvoicebox/gateway:latest
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 2
```

### Docker Compose (Local Development - 15GB RAM Limit)

**IMPORTANT**: This configuration is optimized for local development with a 15GB RAM constraint.
Production deployment uses Kubernetes (see `k8s/` directory).

```yaml
# docker-compose.yml
# LOCAL DEVELOPMENT CONFIGURATION
# Total RAM Budget: 15GB
# 
# Service Allocation:
# - Gateway: 1GB
# - STT Worker: 4GB (Whisper model)
# - TTS Worker: 3GB (Kokoro model)
# - LLM Worker: 1GB (API calls only)
# - Redis: 2GB
# - PostgreSQL: 1GB
# - Prometheus: 512MB
# - Grafana: 512MB
# - Reserved/Buffer: 2GB
# -----------------------
# Total: ~15GB

version: "3.9"

services:
  gateway:
    build: 
      context: .
      dockerfile: docker/gateway/Dockerfile
    ports:
      - "8000:8000"
      - "8001:8001"  # Metrics
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://agentvoicebox:agentvoicebox@postgres:5432/agentvoicebox
      - LOG_LEVEL=INFO
      - WORKERS=2
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    volumes:
      - ./ovos-voice-agent:/app/src
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped
    
  stt-worker:
    build:
      context: .
      dockerfile: docker/stt-worker/Dockerfile
    environment:
      - REDIS_URL=redis://redis:6379
      - WHISPER_MODEL=base
      - WHISPER_DEVICE=cpu
      - WHISPER_COMPUTE_TYPE=int8
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - whisper-cache:/root/.cache/whisper
      - ./ovos-voice-agent:/app/src
    restart: unless-stopped
  
  tts-worker:
    build:
      context: .
      dockerfile: docker/tts-worker/Dockerfile
    environment:
      - REDIS_URL=redis://redis:6379
      - KOKORO_MODEL_DIR=/models/kokoro
      - TTS_ENGINE=kokoro
    deploy:
      resources:
        limits:
          memory: 3G
        reservations:
          memory: 1G
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - kokoro-models:/models/kokoro
      - tts-cache:/app/cache
      - ./ovos-voice-agent:/app/src
    restart: unless-stopped
  
  llm-worker:
    build:
      context: .
      dockerfile: docker/llm-worker/Dockerfile
    environment:
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_API_BASE=${OPENAI_API_BASE:-https://api.openai.com/v1}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 256M
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ./ovos-voice-agent:/app/src
    restart: unless-stopped
  
  redis:
    image: redis:7-alpine
    command: >
      redis-server 
      --appendonly yes 
      --maxmemory 2gb 
      --maxmemory-policy volatile-lru
      --save 60 1
      --loglevel warning
    ports:
      - "6379:6379"
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 3
    restart: unless-stopped
  
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=agentvoicebox
      - POSTGRES_PASSWORD=agentvoicebox
      - POSTGRES_DB=agentvoicebox
    command: >
      postgres 
      -c shared_buffers=256MB 
      -c max_connections=100
      -c work_mem=4MB
    ports:
      - "5432:5432"
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agentvoicebox -d agentvoicebox"]
      interval: 5s
      timeout: 3s
      retries: 3
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.52.0
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=7d"
      - "--storage.tsdb.retention.size=1GB"
    ports:
      - "9090:9090"
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
    volumes:
      - ./docker/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    depends_on:
      - gateway
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.4.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
    volumes:
      - grafana-data:/var/lib/grafana
      - ./docker/grafana/provisioning:/etc/grafana/provisioning:ro
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis-data:
    driver: local
  postgres-data:
    driver: local
  whisper-cache:
    driver: local
  kokoro-models:
    driver: local
  tts-cache:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local

networks:
  default:
    name: agentvoicebox-network
```

### Docker Compose Profiles (Optional Services)

```yaml
# docker-compose.override.yml (for GPU development)
version: "3.9"

services:
  stt-worker:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - WHISPER_DEVICE=cuda
      - WHISPER_COMPUTE_TYPE=float16

  tts-worker:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Local Development Commands

```bash
# Start all services (15GB RAM mode)
docker compose up -d

# Start with GPU support (if available)
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# View logs
docker compose logs -f gateway

# Scale workers (within RAM limits)
docker compose up -d --scale stt-worker=2 --scale tts-worker=2

# Stop all services
docker compose down

# Clean up volumes (reset state)
docker compose down -v
```

---

## Migration Path

### Phase 1: Extract Session State to Redis
1. Add Redis client to existing gateway
2. Implement DistributedSessionManager alongside in-memory
3. Feature flag to switch between implementations
4. Validate consistency, then remove in-memory

### Phase 2: Extract Workers
1. Create STT worker service
2. Route audio through Redis Streams
3. Validate latency meets requirements
4. Repeat for TTS and LLM workers

### Phase 3: Horizontal Scaling
1. Deploy multiple gateway instances
2. Configure HAProxy for WebSocket routing
3. Implement connection draining
4. Load test to validate scale targets

### Phase 4: Production Hardening
1. Add circuit breakers
2. Implement comprehensive metrics
3. Set up alerting
4. Chaos testing for failure scenarios

---

## Open Questions

1. **GPU Allocation**: Should STT and TTS share GPUs or have dedicated pools?
2. **Multi-Region**: Is multi-region deployment required for v1?
3. **Tenant Isolation**: How strict should tenant isolation be? (Separate Redis namespaces vs. separate clusters)
4. **Audio Storage**: Should raw audio be persisted for compliance, or only transcripts?

# Implementation Plan

## AgentVoiceBox Production Architecture

This implementation plan transforms the existing prototype into a production-ready distributed system based on the design document.

---

## Completed Work (Already Implemented)

- [x] Basic WebSocket transport (`AgentVoiceBoxEngine/app/transports/realtime_ws.py`)
- [x] Session management with PostgreSQL (`AgentVoiceBoxEngine/app/services/session_service.py`)
- [x] Token service for ephemeral secrets (`AgentVoiceBoxEngine/app/services/token_service.py`)
- [x] TTS provider abstraction with Kokoro/Piper/Espeak fallback (`AgentVoiceBoxEngine/app/tts/provider.py`)
- [x] Basic Prometheus metrics (`AgentVoiceBoxEngine/app/observability/metrics.py`)
- [x] OPA policy integration (`AgentVoiceBoxEngine/app/services/opa_client.py`)
- [x] SQLAlchemy models for sessions and conversation items (`AgentVoiceBoxEngine/app/models/`)
- [x] Docker Compose with PostgreSQL, Kafka, OPA, Prometheus (`AgentVoiceBoxEngine/docker-compose.yml`)
- [x] In-memory rate limiter (`ovos-voice-agent/rate_limiter.py`)
- [x] Speech pipeline with STT/TTS (`sprint2-speech/speech_pipeline.py`)
- [x] OpenAI-compatible WebSocket server (`sprint4-websocket/realtime_server.py`)
- [x] Basic authentication via bearer tokens (`enterprise/app/utils/auth.py`)

---

## Remaining Tasks

- [x] 1. Redis Integration & Distributed Session Management
  - [x] 1.1 Add Redis to docker-compose.yml
    - Add Redis 7 service with `appendonly yes` persistence
    - Configure `maxmemory 2gb` and `maxmemory-policy volatile-lru`
    - Add health check with `redis-cli ping`
    - _Requirements: 9.1, 9.2_

  - [x] 1.2 Create Redis client wrapper with connection pooling
    - Implement async Redis client using redis-py
    - Add connection pooling (max 50 connections)
    - Add automatic reconnection logic
    - _Requirements: 9.1_

  - [x] 1.3 Implement DistributedSessionManager
    - Create session in Redis hash with 30-second TTL
    - Implement `get_session` with fallback handling
    - Implement `update_session` with pub/sub notification
    - Implement heartbeat refresh mechanism
    - _Requirements: 9.2, 9.3, 9.4_

  - [ ]* 1.4 Write property test for session consistency
    - **Property 1: Session State Consistency**
    - **Validates: Requirements 9.2, 9.3**

  - [x] 1.5 Implement session cleanup background task
    - Scan for expired sessions every 30 seconds
    - Clean up associated resources (audio buffers, conversation items)
    - Emit session.closed events
    - _Requirements: 9.4_

- [x] 2. Distributed Rate Limiter
  - [x] 2.1 Implement Redis Lua script for atomic rate limiting
    - Create sliding window algorithm in Lua
    - Support both request and token limits
    - Return remaining quota and reset time
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 2.2 Create DistributedRateLimiter class
    - Load Lua script on initialization
    - Implement `check_and_consume` method
    - Implement `get_limits` method
    - Support per-tenant limit overrides
    - _Requirements: 6.1, 6.2, 6.4_

  - [ ]* 2.3 Write property test for rate limit accuracy
    - **Property 2: Rate Limit Accuracy**
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [x] 2.4 Integrate rate limiter into gateway
    - Replace in-memory rate limiter with Redis-based
    - Check limits on every incoming event
    - Send `rate_limits.updated` events
    - Return `rate_limit_error` when exceeded
    - _Requirements: 6.1, 6.2, 6.6_

- [x] 3. Checkpoint - Verify Redis integration
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Gateway Service Refactor
  - [-] 4.1 Refactor gateway to be stateless
    - Remove all in-memory session storage
    - Use DistributedSessionManager for all state
    - Use Redis pub/sub for cross-instance events
    - _Requirements: 7.1, 7.2_

  - [ ] 4.2 Implement connection lifecycle management
    - Handle WebSocket connect with auth validation
    - Implement graceful disconnect with cleanup
    - Add connection draining for shutdown (SIGTERM handling)
    - _Requirements: 7.1, 7.4_

  - [ ]* 4.3 Write property test for connection draining
    - **Property 6: Connection Draining**
    - **Validates: Requirements 7.4**

  - [ ] 4.4 Implement event routing to workers via Redis Streams
    - Route audio events to STT worker stream
    - Route synthesis requests to TTS worker stream
    - Route LLM requests to LLM worker stream
    - _Requirements: 7.3_

  - [ ] 4.5 Implement pub/sub listener for worker responses
    - Subscribe to transcription results
    - Subscribe to TTS audio chunks
    - Subscribe to LLM responses
    - Forward results to appropriate WebSocket
    - _Requirements: 7.3_

- [ ] 5. Checkpoint - Verify gateway refactor
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. STT Worker Service
  - [ ] 6.1 Create STT worker with Redis Streams consumer
    - Implement consumer group `stt-workers` for work distribution
    - Load Faster-Whisper model on startup
    - Configure for CPU or GPU based on environment
    - _Requirements: 10.1, 10.2_

  - [ ] 6.2 Implement audio transcription pipeline
    - Decode base64 audio from stream message
    - Run Whisper transcription
    - Publish results to Redis pub/sub channel `transcription:{session_id}`
    - Acknowledge processed messages
    - _Requirements: 10.2, 10.4_

  - [ ]* 6.3 Write property test for transcription delivery
    - **Property 4: Transcription Delivery**
    - **Validates: Requirements 10.4**

  - [ ] 6.4 Create STT worker Dockerfile
    - Multi-stage build with Faster-Whisper
    - Support CPU (int8) and GPU (float16) compute types
    - Configure 4GB memory limit
    - _Requirements: 10.1, 10.3_

- [ ] 7. TTS Worker Service
  - [ ] 7.1 Create TTS worker with Redis Streams consumer
    - Implement consumer group `tts-workers` for work distribution
    - Load Kokoro ONNX model on startup
    - Configure voice and speed defaults from environment
    - _Requirements: 11.1, 11.2_

  - [ ] 7.2 Implement streaming audio synthesis
    - Generate audio chunks as they're produced
    - Publish chunks to Redis Stream `audio:out:{session_id}` with ordering
    - Support cancellation mid-synthesis via `is_cancelled` check
    - _Requirements: 11.2, 11.3, 11.5_

  - [ ]* 7.3 Write property test for audio chunk ordering
    - **Property 3: Audio Chunk Ordering**
    - **Validates: Requirements 11.3**

  - [ ]* 7.4 Write property test for cancel propagation
    - **Property 5: Cancel Propagation**
    - **Validates: Requirements 11.5**

  - [ ] 7.5 Create TTS worker Dockerfile
    - Multi-stage build with Kokoro ONNX
    - Include model files in image
    - Configure 3GB memory limit
    - Fall back to Piper if Kokoro unavailable
    - _Requirements: 11.1, 11.6, 11.7_

- [ ] 8. Checkpoint - Verify worker services
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. LLM Worker Service
  - [ ] 9.1 Create LLM worker with multi-provider support
    - Implement provider abstraction (OpenAI, Groq, Ollama)
    - Configure providers from environment variables
    - Implement provider health checks
    - _Requirements: 12.1, 12.2, 12.4_

  - [ ] 9.2 Implement circuit breaker for provider failover
    - Track failures per provider (threshold: 5 consecutive failures)
    - Open circuit and fail fast when threshold exceeded
    - Attempt recovery after 30 seconds
    - _Requirements: 12.4, 16.4, 16.5_

  - [ ]* 9.3 Write property test for circuit breaker recovery
    - **Property 8: Circuit Breaker Recovery**
    - **Validates: Requirements 16.4, 16.5**

  - [ ] 9.4 Implement streaming token generation
    - Stream tokens as they arrive from LLM
    - Forward to TTS worker immediately (don't wait for completion)
    - Support function calling detection
    - _Requirements: 12.2, 12.3, 12.6_

- [ ] 10. PostgreSQL Schema & Migrations
  - [ ] 10.1 Create database schema with Alembic migrations
    - Create `sessions` table with partitioning by tenant_id
    - Create `conversation_items` table with JSONB content
    - Create `audit_logs` table
    - Add covering indexes for query optimization
    - _Requirements: 13.1, 13.2, 13.4_

  - [ ] 10.2 Implement async database client
    - Use asyncpg for async PostgreSQL access
    - Implement connection pooling
    - Add write-ahead buffering for non-blocking writes
    - _Requirements: 13.1, 13.3_

  - [ ]* 10.3 Write property test for message persistence
    - **Property 9: Message Persistence**
    - **Validates: Requirements 13.3, 13.5**

  - [ ] 10.4 Implement conversation history queries
    - Query last 100 items for a session within 100ms
    - Support pagination
    - Overflow from Redis to PostgreSQL automatically
    - _Requirements: 13.5, 9.5_

- [ ] 11. Checkpoint - Verify database integration
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Authentication & Tenant Isolation
  - [ ] 12.1 Enhance token validation middleware
    - Validate bearer tokens on WebSocket connect within 50ms
    - Support ephemeral session tokens with 10-minute TTL
    - Reject expired tokens with `authentication_error`
    - _Requirements: 3.1, 3.3, 3.5, 3.6_

  - [ ]* 12.2 Write property test for authentication enforcement
    - **Property 10: Authentication Enforcement**
    - **Validates: Requirements 3.1, 3.3**

  - [ ] 12.3 Implement tenant isolation
    - Namespace Redis keys by tenant_id
    - Filter database queries by tenant_id at database layer
    - Validate tenant context on every operation
    - _Requirements: 1.2, 1.3, 1.4_

  - [ ] 12.4 Add PII redaction to logging
    - Identify sensitive fields (audio, transcripts, user identifiers)
    - Redact before logging
    - Maintain correlation IDs (session_id, trace_id) for debugging
    - _Requirements: 15.3, 14.4_

- [ ] 13. Observability Stack
  - [ ] 13.1 Implement comprehensive Prometheus metrics
    - Add latency histograms: `websocket_message_processing`, `stt_transcription`, `tts_synthesis`, `llm_generation`
    - Add gauges: `active_connections`, `queue_depth`, `worker_utilization`
    - Add counters: `requests_total`, `errors_total`, `rate_limits_total`
    - _Requirements: 14.1, 14.2, 14.3_

  - [ ] 13.2 Configure Prometheus scraping
    - Create prometheus.yml with service discovery
    - Set scrape interval to 15 seconds
    - Configure 7-day retention for local development
    - _Requirements: 14.1_

  - [ ] 13.3 Create Grafana dashboards
    - System overview dashboard
    - Per-service dashboards (Gateway, STT, TTS, LLM)
    - Alert configuration for SLO breaches
    - _Requirements: 14.1, 14.7_

  - [ ] 13.4 Implement structured JSON logging
    - JSON format with fields: timestamp, level, service, tenant_id, session_id, correlation_id, message
    - Configure log levels per service
    - _Requirements: 14.4_

- [ ] 14. Error Handling & Fault Tolerance
  - [ ] 14.1 Implement error taxonomy per OpenAI spec
    - Define all error types: `invalid_request_error`, `authentication_error`, `rate_limit_error`, `api_error`, etc.
    - Create error event factory
    - Standardize error responses
    - _Requirements: 16.1_

  - [ ] 14.2 Implement graceful degradation modes
    - Text-only mode when TTS fails
    - Echo mode with apology when LLM fails
    - Degraded mode when Redis partially fails (reject new connections, continue existing)
    - _Requirements: 16.6_

  - [ ]* 14.3 Write property test for heartbeat liveness
    - **Property 7: Heartbeat Liveness**
    - **Validates: Requirements 9.3, 9.4**

- [ ] 15. Checkpoint - Verify observability and error handling
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Integration Testing
  - [ ] 16.1 Create end-to-end test suite
    - Test full speech-to-speech pipeline
    - Test cancel propagation across all workers
    - Test session persistence across gateway restart
    - _Requirements: 7.1, 7.3_

  - [ ] 16.2 Create load test suite with Locust
    - Test 100 concurrent connections (local dev limit)
    - Test rate limiting under load
    - Measure latency percentiles (p50, p95, p99)
    - _Requirements: 7.1, 6.1_

- [ ] 17. Docker Compose Production Configuration
  - [ ] 17.1 Update docker-compose.yml with all services
    - Gateway service (1GB memory limit)
    - STT worker (4GB memory limit)
    - TTS worker (3GB memory limit)
    - LLM worker (1GB memory limit)
    - Redis (2GB memory limit)
    - PostgreSQL (1GB memory limit)
    - Prometheus (512MB memory limit)
    - Grafana (512MB memory limit)
    - Total: ~15GB for local development
    - _Requirements: 17.6_

  - [ ] 17.2 Create health check endpoints
    - `/health` for liveness probe
    - `/ready` for readiness probe (includes dependency checks)
    - _Requirements: 7.7_

- [ ] 18. Documentation
  - [ ] 18.1 Create local development guide
    - Docker Compose setup instructions
    - Environment variable documentation
    - Troubleshooting guide
    - _Requirements: 18.1, 18.4_

  - [ ] 18.2 Create API documentation
    - OpenAPI 3.1 spec for REST endpoints
    - WebSocket event documentation (AsyncAPI)
    - Code examples
    - _Requirements: 18.1, 18.2_

- [ ] 19. Final Checkpoint - Production readiness verification
  - Ensure all tests pass, ask the user if questions arise.

# Implementation Plan

## AgentVoiceBox Production Architecture

This implementation plan transforms the existing prototype into a production-ready distributed SaaS platform.

---

## Phase 1: Core Infrastructure (COMPLETED)

- [x] 1. Redis Integration & Distributed Session Management
  - [x] 1.1 Add Redis 7 to docker-compose.yml with persistence and health checks
  - [x] 1.2 Create RedisClient with connection pooling and auto-reconnection
  - [x] 1.3 Implement DistributedSessionManager with tenant isolation
  - [x] 1.4 Implement session cleanup background task
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 2. Distributed Rate Limiter
  - [x] 2.1 Implement Redis Lua script for atomic sliding window rate limiting
  - [x] 2.2 Create DistributedRateLimiter with per-tenant overrides
  - [x] 2.3 Integrate rate limiter into WebSocket gateway
  - [x] 2.4 Wire up dependencies in dependencies.py
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6_

---

## Phase 2: Gateway Stateless Refactor

- [x] 3. Gateway Stateless Refactor
  - [x] 3.1 Replace in-memory session storage with DistributedSessionManager
    - Modified app/__init__.py to initialize Redis client on startup
    - Modified realtime_ws.py to create sessions in both Redis and PostgreSQL
    - Added heartbeat mechanism (15s interval) to keep Redis sessions alive
    - Added Redis session cleanup on WebSocket disconnect
    - Conversation items written to both Redis (fast) and PostgreSQL (durable)
    - _Requirements: 7.1, 7.2_

  - [x] 3.2 Implement SIGTERM graceful shutdown with connection draining
    - Created ConnectionManager class for tracking active connections
    - Added SIGTERM/SIGINT signal handlers
    - Implemented 30-second drain period before force close
    - New connections rejected during shutdown with `server_shutting_down` error
    - Connections registered/unregistered in WebSocket handler
    - _Requirements: 7.4_

  - [x] 3.3 Implement Redis Streams for worker communication
    - Created RedisStreamsClient class (`app/services/redis_streams.py`)
    - Streams: `audio:stt` (STT requests), `tts:requests` (TTS requests), `audio:out:{session_id}` (audio chunks)
    - Consumer groups: `stt-workers`, `tts-workers`
    - Pub/Sub channels: `transcription:{session_id}`, `tts:{session_id}`
    - Methods: publish_audio_for_stt, publish_tts_request, publish_audio_chunk, subscribe_to_transcriptions
    - Integrated into app initialization and dependencies
    - _Requirements: 7.3_

  - [x] 3.4 Implement pub/sub listener for worker responses
    - Added streams_client to RealtimeWebsocketConnection
    - Methods: _publish_audio_for_stt, _publish_tts_request, _cancel_tts_worker
    - Cleanup worker streams on session close
    - Cancel handler notifies TTS workers
    - RedisStreamsClient has subscribe_to_transcriptions and read_audio_chunks methods
    - _Requirements: 7.3_

- [x] 4. Checkpoint - Gateway Refactor
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 3: Worker Services

- [x] 5. STT Worker Service
  - [x] 5.1 Create STT worker with Redis Streams consumer group
    - Consumer group: `stt-workers`
    - Load Faster-Whisper model (CPU/GPU configurable)
    - _Requirements: 10.1, 10.2_

  - [x] 5.2 Implement transcription pipeline
    - Decode base64 audio, transcribe, publish to pub/sub
    - _Requirements: 10.2, 10.4_

  - [x] 5.3 Create STT worker Dockerfile (4GB memory limit)
    - _Requirements: 10.1, 10.3_

- [x] 6. TTS Worker Service
  - [x] 6.1 Create TTS worker with Redis Streams consumer group
    - Consumer group: `tts-workers`
    - Load Kokoro ONNX model
    - _Requirements: 11.1, 11.2_

  - [x] 6.2 Implement streaming synthesis with cancellation support
    - Publish chunks with sequence numbers
    - Check cancellation flag
    - _Requirements: 11.2, 11.3, 11.5_

  - [x] 6.3 Create TTS worker Dockerfile (3GB memory limit)
    - _Requirements: 11.1, 11.6, 11.7_

- [x] 7. LLM Worker Service
  - [x] 7.1 Create LLM worker with multi-provider support (OpenAI, Groq, Ollama)
    - _Requirements: 12.1, 12.2_

  - [x] 7.2 Implement circuit breaker for provider failover
    - 5 failures threshold, 30s recovery timeout
    - _Requirements: 12.4, 16.4, 16.5_

  - [x] 7.3 Implement streaming token generation to TTS
    - _Requirements: 12.2, 12.3_

- [x] 8. Checkpoint - Worker Services
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 4: Database & Persistence

- [x] 9. PostgreSQL Schema & Migrations
  - [x] 9.1 Create Alembic migrations for tenants, projects, api_keys, sessions, conversation_items, audit_logs
    - Partition by tenant_id
    - _Requirements: 13.1, 13.2, 13.4_

  - [x] 9.2 Implement async database client with asyncpg
    - _Requirements: 13.1, 13.3_

  - [x] 9.3 Implement Redis-to-PostgreSQL overflow for conversation items
    - _Requirements: 13.5, 9.5_

- [x] 10. Checkpoint - Database
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 5: Authentication & Multi-Tenancy

- [x] 11. Enhanced Authentication
  - [x] 11.1 Implement API key validation with Argon2id hashing
    - Redis cache (hot) with PostgreSQL fallback (cold)
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 11.2 Implement ephemeral session tokens (10-minute TTL)
    - _Requirements: 3.5, 3.6_

  - [x] 11.3 Implement tenant isolation at all layers
    - Redis key namespacing, DB query filtering
    - _Requirements: 1.2, 1.3, 1.4_

- [x] 12. PII Redaction & Audit Logging
  - [x] 12.1 Implement PII redaction in logs
    - _Requirements: 15.3, 14.4_

  - [x] 12.2 Implement audit logging for admin actions
    - _Requirements: 1.7, 15.5_

- [x] 13. Checkpoint - Auth & Multi-Tenancy
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 6: Observability

- [x] 14. Metrics & Monitoring
  - [x] 14.1 Add latency histograms (p50/p95/p99) for all operations
    - _Requirements: 14.1, 14.2_

  - [x] 14.2 Add gauges for active_connections, queue_depth, worker_utilization
    - _Requirements: 14.3_

  - [x] 14.3 Configure Prometheus scraping and Grafana dashboards
    - _Requirements: 14.1, 14.7_

- [x] 15. Structured Logging
  - [x] 15.1 Implement JSON logging with correlation IDs
    - _Requirements: 14.4_

- [x] 16. Checkpoint - Observability
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 7: Error Handling & Fault Tolerance

- [x] 17. Error Handling
  - [x] 17.1 Implement OpenAI-compatible error taxonomy
    - _Requirements: 16.1_

  - [x] 17.2 Implement graceful degradation modes
    - Text-only when TTS fails, echo mode when LLM fails
    - _Requirements: 16.6_

- [x] 18. Checkpoint - Error Handling
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 7.5: Real Infrastructure Integration Tests

- [x] 18.5. Real Infrastructure Tests (Docker Compose Stack)
  - [x] 18.5.1 Create docker-compose.test.yml with full stack (Redis, PostgreSQL, Gateway, Workers)
    - Real Redis 7 cluster mode simulation
    - Real PostgreSQL 16 with test database
    - Real Gateway service
    - Real STT/TTS/LLM workers (can use mock models for speed)
    - _Requirements: 17.6_

  - [x] 18.5.2 Real Redis Integration Tests
    - Test distributed session creation/retrieval across multiple gateway instances
    - Test rate limiter accuracy under concurrent load (100 requests)
    - Test Redis failover behavior (kill primary, verify replica promotion)
    - Test Redis Streams consumer group rebalancing
    - Test session TTL expiration and cleanup
    - _Requirements: 9.1, 9.2, 9.3, 9.6_

  - [x] 18.5.3 Real PostgreSQL Integration Tests
    - Test tenant isolation (create 2 tenants, verify no cross-access)
    - Test conversation item persistence and retrieval
    - Test audit log writes and queries
    - Test database connection pool under load
    - Test migration rollback/forward
    - _Requirements: 13.1, 13.2, 13.5_

  - [x] 18.5.4 Real WebSocket Gateway Tests
    - Test WebSocket connection lifecycle (connect, auth, messages, disconnect)
    - Test session reconnection to different gateway instance
    - Test graceful shutdown with active connections (SIGTERM)
    - Test rate limiting rejection with proper error codes
    - Test concurrent connections (50+ simultaneous)
    - _Requirements: 7.1, 7.2, 7.4, 7.6_

  - [x] 18.5.5 Real Worker Pipeline Tests
    - Test STT worker: send real audio, verify transcription returned
    - Test TTS worker: send text, verify audio chunks returned in order
    - Test LLM worker: send prompt, verify streaming response
    - Test worker failover (kill worker, verify work reassigned)
    - Test cancellation propagation (cancel mid-TTS, verify stops)
    - _Requirements: 10.1, 10.2, 11.1, 11.2, 12.1, 12.2_

  - [x] 18.5.6 Real End-to-End Speech Pipeline Test
    - Full flow: Audio → STT → LLM → TTS → Audio out
    - Measure actual latencies (STT p95 < 500ms, TTS TTFB p95 < 200ms)
    - Test with multiple concurrent sessions
    - _Requirements: 14.2_

  - [x] 18.5.7 Real Authentication & Multi-Tenancy Tests
    - Test API key validation against real PostgreSQL
    - Test ephemeral token flow (issue, validate, expire)
    - Test tenant isolation in Redis keys
    - Test cross-tenant access denial
    - _Requirements: 3.1, 3.2, 3.5, 1.2, 1.3_

- [-] 18.6. Checkpoint - Real Infrastructure Tests
  - All tests run against real Docker Compose stack
  - No mocks, no fakes, no stubs
  - Verify actual latencies meet SLA targets

---

## Phase 8: SaaS Platform - Identity (Keycloak)

- [ ] 19. Keycloak Integration
  - [ ] 19.1 Add Keycloak 24 to docker-compose.yml
    - _Requirements: 19.1_

  - [ ] 19.2 Create realm configuration with roles (tenant_admin, developer, viewer, billing_admin)
    - _Requirements: 19.2, 19.7_

  - [ ] 19.3 Implement KeycloakService for user/realm management
    - _Requirements: 19.2, 19.8_

  - [ ] 19.4 Integrate Keycloak JWT validation in gateway
    - _Requirements: 19.4, 19.7_

  - [ ] 19.5 Implement user deactivation flow
    - _Requirements: 19.9_

- [ ] 20. Checkpoint - Keycloak
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 9: SaaS Platform - Billing (Lago)

- [ ] 21. Lago Integration
  - [ ] 21.1 Add Lago to docker-compose.yml
    - _Requirements: 20.1_

  - [ ] 21.2 Configure billing plans (Free, Pro, Enterprise)
    - _Requirements: 20.3, 23.1_

  - [ ] 21.3 Configure billable metrics (api_requests, audio_minutes, llm_tokens)
    - _Requirements: 20.4_

  - [ ] 21.4 Implement LagoService for customer/subscription management
    - _Requirements: 20.1, 20.5_

  - [ ] 21.5 Implement async usage metering pipeline
    - _Requirements: 20.5_

- [ ] 22. Payment Processors
  - [ ] 22.1 Integrate Stripe with webhook handlers
    - _Requirements: 20.6, 22.1_

  - [ ] 22.2 Integrate PayPal with webhook handlers
    - _Requirements: 20.6, 22.2_

  - [ ] 22.3 Implement refund processing
    - _Requirements: 20.8, 22.7_

  - [ ] 22.4 Implement dunning and suspension flow
    - _Requirements: 20.9_

- [ ] 23. Checkpoint - Billing
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 10: Customer Portal Backend

- [ ] 24. Portal API
  - [ ] 24.1 Create FastAPI portal service with Keycloak auth middleware
    - _Requirements: 21.1, 21.2_

  - [ ] 24.2 Implement dashboard endpoints (usage, billing, health)
    - _Requirements: 21.3, 21.5_

  - [ ] 24.3 Implement API key management endpoints
    - _Requirements: 21.4_

  - [ ] 24.4 Implement billing endpoints (plan, invoices, upgrade/downgrade)
    - _Requirements: 21.6, 21.7_

  - [ ] 24.5 Implement payment method endpoints
    - _Requirements: 22.3, 22.4, 22.6_

  - [ ] 24.6 Implement team management endpoints
    - _Requirements: 21.9_

  - [ ] 24.7 Implement settings and webhook endpoints
    - _Requirements: 21.8_

- [ ] 25. Checkpoint - Portal Backend
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 11: Customer Portal Frontend

- [ ] 26. Portal UI
  - [ ] 26.1 Set up Next.js 14 with TypeScript and Tailwind
    - _Requirements: 21.1, 21.10_

  - [ ] 26.2 Implement dashboard page with usage charts
    - _Requirements: 21.3_

  - [ ] 26.3 Implement API keys management page
    - _Requirements: 21.4_

  - [ ] 26.4 Implement usage analytics page
    - _Requirements: 21.5_

  - [ ] 26.5 Implement billing page (plans, invoices, payment methods)
    - _Requirements: 21.6, 21.7_

  - [ ] 26.6 Implement team management page
    - _Requirements: 21.9_

  - [ ] 26.7 Implement settings page
    - _Requirements: 21.8_

  - [ ] 26.8 Implement responsive design and WCAG 2.1 AA accessibility
    - _Requirements: 21.10_

- [ ] 27. Checkpoint - Portal Frontend
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 12: Tenant Onboarding

- [ ] 28. Onboarding Flow
  - [ ] 28.1 Implement signup endpoint (creates Keycloak user, Lago customer, first API key)
    - _Requirements: 24.1, 24.2_

  - [ ] 28.2 Implement signup frontend with email verification
    - _Requirements: 24.2, 24.3_

  - [ ] 28.3 Implement interactive quickstart (test API call)
    - _Requirements: 24.4, 24.5_

  - [ ] 28.4 Implement welcome email template
    - _Requirements: 24.3_

  - [ ] 28.5 Implement onboarding milestone tracking
    - _Requirements: 24.7, 24.8_

- [ ] 29. Checkpoint - Onboarding
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 13: Final Integration & Documentation

- [ ] 30. Integration Testing
  - [ ] 30.1 End-to-end onboarding test
  - [ ] 30.2 Billing integration test (Stripe test mode)
  - [ ] 30.3 Full speech-to-speech pipeline test
  - [ ] 30.4 Load test with Locust (100 concurrent connections)

- [ ] 31. Documentation
  - [ ] 31.1 Local development guide (Docker Compose setup)
    - _Requirements: 18.1, 18.4_

  - [ ] 31.2 OpenAPI 3.1 spec for REST endpoints
    - _Requirements: 18.1_

  - [ ] 31.3 AsyncAPI spec for WebSocket events
    - _Requirements: 18.2_

- [ ] 32. Final Checkpoint - Production Ready
  - Ensure all tests pass, ask the user if questions arise.

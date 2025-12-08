# Requirements Document

## Introduction

This document specifies the requirements for transforming the OVOS Voice Agent from a working prototype (handling ~100 concurrent connections) into a production-grade, globally distributed speech-to-speech platform capable of handling **millions of concurrent connections** with sub-200ms latency. The system must maintain 100% OpenAI Realtime API compatibility while leveraging battle-tested open-source infrastructure.

## Glossary

- **Gateway**: Stateless WebSocket termination layer that handles protocol translation and routing
- **Worker**: Specialized microservice that performs CPU/GPU-intensive tasks (STT, TTS, LLM inference)
- **Session**: A single client connection with associated state (conversation history, audio buffers, config)
- **Turn**: A complete user utterance from speech start to speech end
- **Barge-in**: User interruption during assistant audio playback
- **VAD**: Voice Activity Detection - algorithm to detect speech presence in audio
- **STT**: Speech-to-Text transcription service
- **TTS**: Text-to-Speech synthesis service
- **Backpressure**: Flow control mechanism to prevent overwhelming downstream services
- **Circuit Breaker**: Pattern to prevent cascade failures when dependencies are unhealthy
- **CQRS**: Command Query Responsibility Segregation - separate read/write paths for scalability

---

## Requirements

### Requirement 1: Horizontal Gateway Scaling

**User Story:** As a platform operator, I want to add gateway instances without downtime, so that I can handle traffic spikes by scaling horizontally.

#### Acceptance Criteria

1. WHEN a new gateway instance starts THEN the system SHALL register it with the load balancer within 10 seconds and begin accepting connections
2. WHEN a gateway instance is terminated THEN the system SHALL drain existing connections gracefully over 30 seconds before shutdown
3. WHILE multiple gateway instances are running THEN the system SHALL distribute new connections evenly using consistent hashing on session_id
4. WHEN a client reconnects after gateway failure THEN the system SHALL restore session state from distributed storage within 500ms
5. THE system SHALL maintain session affinity so that all messages for a session route to the same gateway instance when possible

### Requirement 2: Distributed Session Management

**User Story:** As a platform operator, I want session state stored in distributed storage, so that any gateway instance can serve any session.

#### Acceptance Criteria

1. WHEN a session is created THEN the system SHALL persist session metadata to Redis Cluster within 50ms
2. WHEN session state is updated THEN the system SHALL propagate changes to all interested gateway instances via Redis Pub/Sub within 100ms
3. WHILE a session is active THEN the system SHALL maintain a heartbeat in Redis with 30-second TTL to detect abandoned sessions
4. WHEN a session heartbeat expires THEN the system SHALL clean up associated resources and emit session.closed event
5. IF Redis Cluster becomes unavailable THEN the system SHALL continue serving existing in-memory sessions in degraded mode and reject new connections
6. WHEN Redis Cluster recovers THEN the system SHALL resync in-memory state within 60 seconds

### Requirement 3: Distributed Rate Limiting

**User Story:** As a platform operator, I want rate limits enforced globally across all gateway instances, so that no single client can overwhelm the system.

#### Acceptance Criteria

1. WHEN a client exceeds 100 requests per minute THEN the system SHALL reject subsequent requests with rate_limit_error until the window resets
2. WHEN a client exceeds 100,000 tokens per minute THEN the system SHALL reject subsequent requests with rate_limit_error
3. WHILE rate limiting THEN the system SHALL use Redis-based sliding window algorithm for accurate cross-instance enforcement
4. WHEN rate limit state is checked THEN the system SHALL complete the check within 5ms using Redis Lua scripts
5. THE system SHALL support per-tenant rate limit overrides stored in configuration

### Requirement 4: Audio Processing Pipeline

**User Story:** As a user, I want my speech processed with minimal latency, so that conversations feel natural and responsive.

#### Acceptance Criteria

1. WHEN audio chunks arrive at the gateway THEN the system SHALL forward them to STT workers via Redis Streams within 10ms
2. WHEN STT workers receive audio THEN the system SHALL begin transcription within 50ms of receiving the first chunk
3. WHEN transcription completes THEN the system SHALL deliver results to the gateway within 100ms via Redis Pub/Sub
4. WHEN TTS synthesis is requested THEN the system SHALL begin streaming audio chunks within 200ms (time-to-first-byte)
5. WHILE audio is streaming THEN the system SHALL maintain consistent 20ms chunk intervals for smooth playback
6. IF a worker becomes unhealthy THEN the system SHALL route work to healthy workers within 100ms

### Requirement 5: STT Worker Pool

**User Story:** As a platform operator, I want dedicated STT workers that can scale independently, so that transcription capacity matches demand.

#### Acceptance Criteria

1. WHEN an STT worker starts THEN the system SHALL register with the worker pool and begin consuming from Redis Streams
2. WHILE processing audio THEN the STT worker SHALL use Faster-Whisper with CUDA acceleration when available
3. WHEN transcription completes THEN the STT worker SHALL publish results to Redis Pub/Sub with session_id routing key
4. THE STT worker SHALL support batch processing of up to 10 concurrent transcriptions per GPU
5. WHEN worker load exceeds 80% THEN the system SHALL signal for horizontal scaling via metrics
6. IF transcription fails THEN the STT worker SHALL retry once and then emit transcription.failed event

### Requirement 6: TTS Worker Pool

**User Story:** As a platform operator, I want dedicated TTS workers that stream audio efficiently, so that response latency is minimized.

#### Acceptance Criteria

1. WHEN a TTS worker starts THEN the system SHALL load Kokoro ONNX model and register with the worker pool
2. WHEN synthesis is requested THEN the TTS worker SHALL stream audio chunks as they are generated (not wait for completion)
3. WHILE streaming THEN the TTS worker SHALL publish chunks to Redis Streams with ordering guarantees
4. THE TTS worker SHALL support voice selection and speed adjustment per request
5. WHEN a cancel signal is received THEN the TTS worker SHALL stop synthesis within 50ms
6. IF Kokoro is unavailable THEN the TTS worker SHALL fall back to Piper TTS with degraded quality notification

### Requirement 7: LLM Integration

**User Story:** As a user, I want intelligent responses generated quickly, so that conversations are helpful and responsive.

#### Acceptance Criteria

1. WHEN generating a response THEN the system SHALL call the configured LLM API with conversation context
2. WHILE waiting for LLM response THEN the system SHALL enforce a 30-second timeout
3. WHEN LLM response arrives THEN the system SHALL stream tokens to TTS immediately (not wait for completion)
4. IF the primary LLM is unavailable THEN the system SHALL fail over to backup LLM within 5 seconds
5. THE system SHALL support multiple LLM providers (OpenAI, Groq, Ollama) via configuration
6. WHEN function calling is detected THEN the system SHALL execute the function and include results in context

### Requirement 8: Connection Management

**User Story:** As a platform operator, I want efficient connection handling, so that the system can maintain millions of concurrent WebSocket connections.

#### Acceptance Criteria

1. THE gateway SHALL handle 50,000 concurrent WebSocket connections per instance using async I/O
2. WHEN a connection is idle for 5 minutes THEN the system SHALL send a ping frame to verify liveness
3. WHEN a connection fails ping/pong THEN the system SHALL close the connection and clean up resources
4. WHILE accepting connections THEN the system SHALL enforce a maximum of 100 new connections per second per instance to prevent thundering herd
5. THE system SHALL use connection pooling for all downstream services (Redis, Postgres, HTTP)

### Requirement 9: Observability

**User Story:** As a platform operator, I want comprehensive metrics and logging, so that I can monitor system health and debug issues.

#### Acceptance Criteria

1. THE system SHALL expose Prometheus metrics at /metrics endpoint on each service
2. THE system SHALL track latency histograms for: WebSocket message processing, STT transcription, TTS synthesis, LLM generation
3. THE system SHALL track gauges for: active connections, queue depths, worker utilization
4. WHEN an error occurs THEN the system SHALL log structured JSON with correlation_id, session_id, and error details
5. THE system SHALL support distributed tracing via OpenTelemetry when enabled
6. WHEN latency exceeds SLO thresholds THEN the system SHALL emit alerts via configured channels

### Requirement 10: Data Persistence

**User Story:** As a platform operator, I want conversation data persisted reliably, so that I can support audit requirements and conversation continuity.

#### Acceptance Criteria

1. WHEN a conversation item is created THEN the system SHALL persist it to PostgreSQL within 1 second
2. THE system SHALL use async writes with write-ahead buffering to avoid blocking the hot path
3. WHEN querying conversation history THEN the system SHALL return results within 100ms for the last 100 items
4. THE system SHALL partition conversation data by tenant_id for query isolation
5. THE system SHALL retain conversation data for 90 days by default with configurable retention

### Requirement 11: Security

**User Story:** As a platform operator, I want secure authentication and authorization, so that only authorized clients can access the system.

#### Acceptance Criteria

1. WHEN a WebSocket connection is initiated THEN the system SHALL validate the bearer token before accepting
2. THE system SHALL support ephemeral client secrets with configurable TTL (default 10 minutes)
3. WHEN a token expires THEN the system SHALL reject the connection with authentication_error
4. THE system SHALL encrypt all data in transit using TLS 1.3
5. THE system SHALL support tenant isolation so that one tenant cannot access another's sessions
6. WHEN sensitive data is logged THEN the system SHALL redact PII fields

### Requirement 12: Fault Tolerance

**User Story:** As a platform operator, I want the system to handle failures gracefully, so that partial outages don't cause complete service disruption.

#### Acceptance Criteria

1. WHEN a Redis node fails THEN the system SHALL failover to replica within 5 seconds using Redis Sentinel/Cluster
2. WHEN a worker crashes THEN the system SHALL reassign pending work to healthy workers within 10 seconds
3. WHEN a downstream service is unhealthy THEN the system SHALL open circuit breaker after 5 consecutive failures
4. WHILE circuit breaker is open THEN the system SHALL return degraded responses and retry every 30 seconds
5. WHEN circuit breaker closes THEN the system SHALL resume normal operation and log recovery event
6. THE system SHALL support graceful degradation: audio-only mode if LLM fails, text-only mode if TTS fails

---

## Infrastructure Components

### Required Open-Source Infrastructure

| Component | Technology | Purpose | Why This Choice |
|-----------|------------|---------|-----------------|
| Load Balancer | HAProxy 2.8+ | WebSocket routing, health checks | Battle-tested, WebSocket-native, 1M+ conn/instance |
| Session Store | Redis Cluster 7+ | Distributed state, pub/sub, streams | Sub-ms latency, proven at Twitter/Discord scale |
| Message Queue | Redis Streams | Audio chunk routing, work distribution | Simpler than Kafka, sufficient for 1M msg/sec |
| Database | PostgreSQL 16+ | Conversation persistence, audit logs | JSONB for flexible schema, partitioning support |
| Metrics | Prometheus + Grafana | Observability stack | Industry standard, extensive ecosystem |
| Tracing | Jaeger | Distributed tracing | OpenTelemetry compatible, low overhead |
| Container Orchestration | Kubernetes | Auto-scaling, deployment | Required for 1M+ connections across regions |

### Why NOT These Technologies

| Technology | Reason to Skip |
|------------|----------------|
| Kafka | Overkill for <1M msg/sec, adds operational complexity. Redis Streams sufficient. |
| Milvus | Vector DB not needed unless doing semantic search/RAG. Not in current requirements. |
| MongoDB | PostgreSQL JSONB provides same flexibility with better consistency guarantees. |
| RabbitMQ | Redis Streams provides same functionality with fewer moving parts. |
| Consul | Kubernetes provides service discovery natively. |

---

## Capacity Planning

### Target Scale

| Metric | Target | Infrastructure Required |
|--------|--------|------------------------|
| Concurrent Connections | 1,000,000 | 20 gateway instances (50K each) |
| Messages/Second | 500,000 | Redis Cluster 6 nodes |
| STT Requests/Second | 10,000 | 50 STT workers (4 GPU each) |
| TTS Requests/Second | 10,000 | 50 TTS workers (4 GPU each) |
| Storage (90 days) | 10 TB | PostgreSQL with partitioning |

### Resource Estimates Per Component

| Component | CPU | Memory | GPU | Instances |
|-----------|-----|--------|-----|-----------|
| Gateway | 4 cores | 8 GB | - | 20 |
| STT Worker | 4 cores | 16 GB | 1x T4/A10 | 50 |
| TTS Worker | 4 cores | 16 GB | 1x T4/A10 | 50 |
| Redis Cluster | 8 cores | 64 GB | - | 6 |
| PostgreSQL | 16 cores | 128 GB | - | 3 (primary + 2 replicas) |
| HAProxy | 8 cores | 16 GB | - | 3 |


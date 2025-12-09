# Requirements Document

## Introduction

This document specifies the complete requirements for **OVOS Voice Agent SaaS Platform** - a production-grade, multi-tenant Speech-to-Speech platform delivered as a service. The platform provides OpenAI Realtime API-compatible voice processing capabilities to thousands of tenants, handling **millions of concurrent connections** with enterprise-grade security, isolation, and observability.

The system is designed as a **true SaaS offering** with:
- Multi-tenant architecture with complete data isolation
- Self-service tenant onboarding and API key management
- Usage-based metering and billing integration
- White-label capabilities for enterprise customers
- Compliance-ready infrastructure (SOC2, GDPR, HIPAA-ready)

## Glossary

- **Tenant**: A paying customer organization with isolated resources, API keys, and usage quotas
- **Project**: A logical grouping within a tenant (e.g., production, staging, dev environments)
- **API Key**: Authentication credential scoped to a project with configurable permissions
- **Session**: A single WebSocket connection representing one voice conversation
- **Gateway**: Stateless WebSocket termination service that routes requests to workers
- **Worker**: Specialized microservice for CPU/GPU-intensive tasks (STT, TTS, LLM)
- **Control Plane**: Management APIs for tenant/project/key administration
- **Data Plane**: Real-time voice processing APIs (WebSocket, REST)
- **Metering**: Usage tracking for billing (connections, audio minutes, tokens)
- **Quota**: Configurable limits per tenant/project (connections, requests, tokens)
- **Circuit Breaker**: Fault tolerance pattern preventing cascade failures
- **Backpressure**: Flow control preventing system overload
- **PII**: Personally Identifiable Information requiring special handling
- **Vault**: HashiCorp Vault - secrets management system
- **NATS**: High-performance messaging system for distributed communication

---

## Requirements

---

### Requirement 1: Multi-Tenant Architecture

**User Story:** As a SaaS operator, I want complete tenant isolation, so that one tenant's data and traffic cannot affect another tenant.

#### Acceptance Criteria

1. WHEN a tenant is created THEN the system SHALL provision isolated namespaces for sessions, conversations, and audit logs
2. WHILE processing requests THEN the system SHALL enforce tenant context on every operation using tenant_id extracted from API key
3. WHEN querying data THEN the system SHALL apply tenant_id filter at the database layer preventing cross-tenant data access
4. IF a request lacks valid tenant context THEN the system SHALL reject it with authentication_error before any processing
5. WHEN a tenant exceeds resource quotas THEN the system SHALL throttle only that tenant without affecting others
6. THE system SHALL support logical isolation (shared infrastructure) and physical isolation (dedicated resources) per tenant tier
7. WHEN audit logs are written THEN the system SHALL tag every entry with tenant_id for compliance reporting

---

### Requirement 2: Tenant Management (Control Plane)

**User Story:** As a SaaS operator, I want a control plane API to manage tenants, projects, and API keys, so that I can onboard customers and manage their access.

#### Acceptance Criteria

1. WHEN a new tenant is onboarded THEN the system SHALL create tenant record with unique tenant_id, billing_id, and tier assignment
2. WHEN a project is created THEN the system SHALL associate it with exactly one tenant and generate project_id
3. WHEN an API key is generated THEN the system SHALL hash it using Argon2id before storage and return plaintext only once
4. THE system SHALL support API key scopes: `realtime:connect`, `realtime:admin`, `billing:read`, `tenant:admin`
5. WHEN an API key is rotated THEN the system SHALL allow grace period (configurable, default 24 hours) where both old and new keys work
6. WHEN a tenant is suspended THEN the system SHALL immediately reject all new connections and gracefully close existing sessions within 60 seconds
7. THE system SHALL expose REST API at `/v1/admin/tenants`, `/v1/admin/projects`, `/v1/admin/keys` with OpenAPI documentation

---

### Requirement 3: API Key Authentication

**User Story:** As a developer, I want to authenticate using API keys, so that I can integrate the voice API into my application.

#### Acceptance Criteria

1. WHEN a WebSocket connection is initiated THEN the system SHALL require Bearer token in Authorization header or `access_token` query parameter
2. WHEN validating API key THEN the system SHALL verify against hashed keys in Redis cache (hot) with PostgreSQL fallback (cold)
3. IF API key is invalid or expired THEN the system SHALL reject connection with `authentication_error` within 50ms
4. WHEN API key is validated THEN the system SHALL extract tenant_id, project_id, scopes, and rate_limit_tier
5. THE system SHALL support ephemeral session tokens (short-lived, single-use) for browser clients generated via REST endpoint
6. WHEN generating ephemeral token THEN the system SHALL bind it to specific session_id and expire within 10 minutes
7. THE system SHALL log all authentication attempts (success/failure) with source IP, key_id (not full key), and timestamp

---

### Requirement 4: Secrets Management

**User Story:** As a SaaS operator, I want secrets managed securely, so that API keys, database credentials, and encryption keys are never exposed.

#### Acceptance Criteria

1. THE system SHALL use HashiCorp Vault for all secret storage and retrieval in production
2. WHEN a service starts THEN the system SHALL authenticate to Vault using Kubernetes service account (k8s auth method)
3. THE system SHALL store database credentials, API keys for external services, and encryption keys in Vault
4. WHEN secrets are rotated in Vault THEN the system SHALL detect changes within 60 seconds and reload without restart
5. THE system SHALL encrypt sensitive fields in database (conversation transcripts, PII) using keys from Vault
6. IF Vault is unavailable THEN the system SHALL use cached credentials for up to 1 hour and alert operators
7. THE system SHALL never log secrets; all secret values SHALL be redacted in logs and error messages

---

### Requirement 5: Usage Metering & Billing

**User Story:** As a SaaS operator, I want accurate usage metering, so that I can bill tenants based on consumption.

#### Acceptance Criteria

1. THE system SHALL meter the following dimensions: connection_minutes, audio_input_seconds, audio_output_seconds, llm_input_tokens, llm_output_tokens, api_requests
2. WHEN a metered event occurs THEN the system SHALL write to metering pipeline within 100ms (async, non-blocking)
3. THE system SHALL aggregate usage per tenant/project at 1-minute granularity and store in TimescaleDB
4. WHEN usage is queried THEN the system SHALL return accurate totals within 5-minute lag for real-time dashboards
5. THE system SHALL export usage to billing system (Stripe, custom) via webhook or batch export daily
6. WHEN a tenant approaches quota (80%) THEN the system SHALL emit warning event to notification system
7. THE system SHALL support usage-based pricing tiers: Free (limited), Pro (metered), Enterprise (committed)

---

### Requirement 6: Rate Limiting & Quotas

**User Story:** As a SaaS operator, I want configurable rate limits and quotas, so that I can protect the platform and enforce fair usage.

#### Acceptance Criteria

1. THE system SHALL enforce rate limits at three levels: global (platform protection), tenant (fair usage), project (granular control)
2. WHEN rate limit is exceeded THEN the system SHALL return `rate_limit_error` with `retry_after` header indicating reset time
3. THE system SHALL use Redis-based sliding window algorithm with Lua scripts for atomic operations (<5ms latency)
4. THE system SHALL support configurable limits: requests_per_minute, tokens_per_minute, concurrent_connections, audio_minutes_per_day
5. WHEN a tenant upgrades tier THEN the system SHALL apply new limits within 60 seconds without connection disruption
6. THE system SHALL emit `rate_limits.updated` event to connected clients when limits change
7. IF Redis is unavailable THEN the system SHALL fail-open with local rate limiting and alert operators

---

### Requirement 7: Gateway Service (Data Plane Entry)

**User Story:** As a platform operator, I want stateless gateway instances, so that I can scale horizontally to handle millions of connections.

#### Acceptance Criteria

1. THE gateway SHALL handle 50,000 concurrent WebSocket connections per instance using async I/O (uvicorn + uvloop)
2. WHEN a connection is accepted THEN the gateway SHALL authenticate, extract tenant context, and register session in Redis within 100ms
3. WHILE processing messages THEN the gateway SHALL route audio to workers via NATS JetStream and await responses via NATS subscription
4. WHEN a gateway instance is terminated THEN the system SHALL drain connections gracefully over 30 seconds (SIGTERM handling)
5. THE gateway SHALL maintain session affinity using consistent hashing on session_id for optimal cache utilization
6. WHEN a client reconnects THEN the gateway SHALL restore session state from Redis within 500ms
7. THE gateway SHALL expose health endpoints: `/health` (liveness), `/ready` (readiness including dependency checks)

---

### Requirement 8: Message Bus (NATS JetStream)

**User Story:** As a platform architect, I want reliable message delivery between services, so that audio processing is consistent and fault-tolerant.

#### Acceptance Criteria

1. THE system SHALL use NATS JetStream for all inter-service communication with at-least-once delivery guarantees
2. THE system SHALL define streams: `VOICE.audio.inbound`, `VOICE.audio.transcribed`, `VOICE.tts.requests`, `VOICE.tts.chunks`, `VOICE.events`
3. WHEN publishing messages THEN the system SHALL include tenant_id, session_id, timestamp, and correlation_id headers
4. THE system SHALL use consumer groups for worker pools enabling horizontal scaling and automatic rebalancing
5. WHEN a message fails processing THEN the system SHALL retry 3 times with exponential backoff before dead-lettering
6. THE system SHALL retain messages for 24 hours enabling replay for debugging and recovery
7. WHEN NATS cluster loses quorum THEN the system SHALL queue messages locally (up to 1000) and retry connection

---

### Requirement 9: Session State Management (Redis Cluster)

**User Story:** As a platform operator, I want distributed session state, so that any gateway can serve any session.

#### Acceptance Criteria

1. THE system SHALL use Redis Cluster (6+ nodes) for session state with automatic sharding by session_id
2. WHEN a session is created THEN the system SHALL store: session_id, tenant_id, project_id, config, created_at, last_activity with 1-hour TTL
3. WHILE a session is active THEN the system SHALL update last_activity heartbeat every 30 seconds extending TTL
4. WHEN session state is updated THEN the system SHALL publish to Redis Pub/Sub channel `session:{session_id}` for real-time sync
5. THE system SHALL store conversation items in Redis List with automatic trimming to last 100 items (overflow to PostgreSQL)
6. WHEN Redis Cluster fails over THEN the system SHALL reconnect within 5 seconds using Sentinel/Cluster topology refresh
7. THE system SHALL use Redis Streams for audio chunk buffering with consumer groups for worker distribution

---

### Requirement 10: STT Worker Service

**User Story:** As a platform operator, I want dedicated STT workers, so that transcription scales independently of gateway capacity.

#### Acceptance Criteria

1. THE STT worker SHALL consume from NATS stream `VOICE.audio.inbound` using durable consumer group `stt-workers`
2. WHEN audio is received THEN the STT worker SHALL transcribe using Faster-Whisper with CUDA acceleration (if available)
3. THE STT worker SHALL support batch processing: up to 8 concurrent transcriptions per GPU, 4 per CPU-only instance
4. WHEN transcription completes THEN the STT worker SHALL publish to `VOICE.audio.transcribed` with session_id routing
5. THE STT worker SHALL emit metrics: transcription_latency_ms, transcription_success_total, transcription_error_total, gpu_utilization
6. WHEN worker load exceeds 80% THEN the system SHALL signal HPA for horizontal scaling via Prometheus metrics
7. IF transcription fails after 3 retries THEN the STT worker SHALL publish `transcription.failed` event and dead-letter the message

---

### Requirement 11: TTS Worker Service

**User Story:** As a platform operator, I want dedicated TTS workers, so that speech synthesis scales independently and streams efficiently.

#### Acceptance Criteria

1. THE TTS worker SHALL consume from NATS stream `VOICE.tts.requests` using durable consumer group `tts-workers`
2. WHEN synthesis is requested THEN the TTS worker SHALL load Kokoro ONNX model and stream audio chunks as generated
3. THE TTS worker SHALL publish audio chunks to `VOICE.tts.chunks` with sequence numbers for ordered reassembly
4. THE TTS worker SHALL support voice selection from available Kokoro voices and speed adjustment (0.5x - 2.0x)
5. WHEN cancel signal is received THEN the TTS worker SHALL stop synthesis within 50ms and publish `tts.cancelled` event
6. THE TTS worker SHALL cache loaded models in memory and share across requests (model pool pattern)
7. IF Kokoro is unavailable THEN the TTS worker SHALL fall back to Piper TTS and emit `tts.degraded` event

---

### Requirement 12: LLM Integration Service

**User Story:** As a platform operator, I want flexible LLM integration, so that I can use multiple providers and handle failures gracefully.

#### Acceptance Criteria

1. THE system SHALL support LLM providers: OpenAI, Groq, Anthropic, Ollama (self-hosted) via unified interface
2. WHEN generating response THEN the system SHALL use tenant-configured provider or fall back to platform default
3. THE system SHALL stream LLM tokens to TTS immediately (not wait for completion) for minimum latency
4. WHEN primary LLM provider fails THEN the system SHALL failover to backup provider within 5 seconds
5. THE system SHALL enforce per-tenant LLM quotas (tokens/day) and reject requests when exceeded
6. WHEN function calling is detected THEN the system SHALL execute registered functions and include results in context
7. THE system SHALL support tenant-provided API keys (BYOK - Bring Your Own Key) stored encrypted in Vault

---

### Requirement 13: Database Layer (PostgreSQL)

**User Story:** As a platform operator, I want reliable data persistence, so that conversation history and audit logs are durable.

#### Acceptance Criteria

1. THE system SHALL use PostgreSQL 16+ with logical replication for read replicas
2. THE system SHALL partition tables by tenant_id using PostgreSQL native partitioning for query isolation
3. WHEN conversation items overflow Redis THEN the system SHALL persist to PostgreSQL asynchronously within 5 seconds
4. THE system SHALL store: tenants, projects, api_keys, sessions, conversation_items, audit_logs, usage_metrics
5. WHEN querying conversation history THEN the system SHALL return last 100 items within 100ms using covering indexes
6. THE system SHALL retain data per tenant configuration: 30/90/365 days with automated partition pruning
7. THE system SHALL encrypt PII columns (transcripts, user content) using AES-256-GCM with keys from Vault

---

### Requirement 14: Observability Stack

**User Story:** As a platform operator, I want comprehensive observability, so that I can monitor health, debug issues, and meet SLAs.

#### Acceptance Criteria

1. THE system SHALL expose Prometheus metrics at `/metrics` on every service with standard naming convention
2. THE system SHALL track latency histograms (p50, p95, p99) for: websocket_message_processing, stt_transcription, tts_synthesis, llm_generation
3. THE system SHALL track gauges for: active_connections (by tenant), queue_depth, worker_utilization, error_rate
4. THE system SHALL use structured JSON logging with fields: timestamp, level, service, tenant_id, session_id, correlation_id, message
5. THE system SHALL support distributed tracing via OpenTelemetry with Jaeger backend
6. WHEN SLO threshold is breached THEN the system SHALL emit alerts via PagerDuty/Slack/webhook within 60 seconds
7. THE system SHALL provide Grafana dashboards for: platform overview, per-tenant usage, worker health, error analysis

---

### Requirement 15: Security & Compliance

**User Story:** As a SaaS operator, I want enterprise-grade security, so that I can serve regulated industries and pass audits.

#### Acceptance Criteria

1. THE system SHALL encrypt all data in transit using TLS 1.3 with modern cipher suites
2. THE system SHALL encrypt all data at rest using AES-256 (database, object storage, backups)
3. WHEN PII is logged THEN the system SHALL redact or hash sensitive fields (transcripts, user identifiers)
4. THE system SHALL support tenant data residency requirements (EU, US, APAC) via regional deployments
5. THE system SHALL maintain audit logs for all administrative actions with 7-year retention
6. THE system SHALL support SSO integration (SAML 2.0, OIDC) for enterprise tenant authentication
7. THE system SHALL pass security scanning (SAST, DAST, dependency scanning) in CI/CD pipeline

---

### Requirement 16: Fault Tolerance & Disaster Recovery

**User Story:** As a platform operator, I want the system to handle failures gracefully, so that partial outages don't cause complete service disruption.

#### Acceptance Criteria

1. WHEN a Redis node fails THEN the system SHALL failover to replica within 5 seconds using Redis Cluster automatic failover
2. WHEN a NATS node fails THEN the system SHALL reconnect to healthy nodes within 3 seconds using cluster-aware client
3. WHEN a worker crashes THEN the system SHALL reassign pending work via NATS consumer rebalancing within 10 seconds
4. THE system SHALL implement circuit breakers for all external dependencies (LLM APIs, Vault, external services)
5. WHILE circuit breaker is open THEN the system SHALL return degraded responses and retry every 30 seconds
6. THE system SHALL support multi-region active-passive deployment with RPO < 1 minute, RTO < 5 minutes
7. THE system SHALL perform automated backups: PostgreSQL (hourly), Redis (hourly snapshots), Vault (daily)

---

### Requirement 17: Deployment & Infrastructure

**User Story:** As a platform operator, I want automated deployment, so that I can release updates safely and scale on demand.

#### Acceptance Criteria

1. THE system SHALL deploy on Kubernetes using Helm charts with configurable values per environment
2. THE system SHALL use Horizontal Pod Autoscaler (HPA) based on CPU, memory, and custom metrics (connections, queue depth)
3. THE system SHALL support rolling deployments with zero-downtime using readiness probes and PodDisruptionBudgets
4. THE system SHALL use GitOps (ArgoCD/Flux) for declarative infrastructure management
5. THE system SHALL run in containerized form with multi-stage Dockerfile optimized for size and security
6. THE system SHALL support local development using Docker Compose with all dependencies
7. THE system SHALL define infrastructure as code using Terraform for cloud resources (managed databases, load balancers)

---

### Requirement 18: Developer Experience

**User Story:** As a developer integrating the API, I want excellent documentation and SDKs, so that I can build quickly and correctly.

#### Acceptance Criteria

1. THE system SHALL provide OpenAPI 3.1 specification for all REST endpoints with examples
2. THE system SHALL provide AsyncAPI specification for WebSocket protocol with all events documented
3. THE system SHALL provide SDKs for: Python, JavaScript/TypeScript, Go with idiomatic patterns
4. THE system SHALL provide interactive API explorer (Swagger UI) at `/docs` endpoint
5. THE system SHALL provide webhook endpoint for receiving events (session.created, transcription.completed, etc.)
6. THE system SHALL provide sample applications demonstrating common integration patterns
7. THE system SHALL provide status page showing real-time platform health and incident history

---

## Infrastructure Components

### Production Stack (Open Source)

| Layer | Component | Technology | Why This Choice |
|-------|-----------|------------|-----------------|
| **Load Balancer** | L4/L7 LB | HAProxy 2.9 | WebSocket-native, 2M+ conn/instance, battle-tested |
| **API Gateway** | Rate limiting, Auth | Kong / Envoy | Plugin ecosystem, observability, proven at scale |
| **Identity/IAM** | Users, Roles, SSO | Keycloak 24 | Enterprise SSO, SAML/OIDC, RBAC, 10M+ deployments |
| **Billing** | Usage-based billing | Lago | Metering, Stripe/PayPal, invoices, refunds, AGPL-3.0 |
| **Message Bus** | Async messaging | NATS JetStream | 10M+ msg/sec, simpler than Kafka, built-in persistence |
| **Session Store** | Distributed state | Redis Cluster 7.2 | Sub-ms latency, Pub/Sub, Streams, proven at scale |
| **Primary Database** | Relational data | PostgreSQL 16 | JSONB, partitioning, logical replication |
| **Time-Series DB** | Metrics/Metering | TimescaleDB | PostgreSQL-compatible, compression, retention policies |
| **Secrets** | Secret management | HashiCorp Vault | Dynamic secrets, encryption as service, audit logging |
| **Observability** | Metrics | Prometheus + Thanos | Federation, long-term storage, proven ecosystem |
| **Observability** | Logging | Loki + Grafana | Log aggregation, label-based queries, cost-effective |
| **Observability** | Tracing | Jaeger | OpenTelemetry native, distributed tracing |
| **Orchestration** | Container platform | Kubernetes | Auto-scaling, self-healing, declarative |
| **GitOps** | Deployment | ArgoCD | Declarative, audit trail, rollback |
| **CI/CD** | Pipeline | GitHub Actions | Native integration, marketplace actions |
| **IaC** | Infrastructure | Terraform | Multi-cloud, state management, modules |

### Why NOT These Technologies

| Technology | Reason to Skip |
|------------|----------------|
| **Kafka** | Overkill for <10M msg/sec. NATS JetStream provides persistence with simpler operations. |
| **MongoDB** | PostgreSQL JSONB provides flexibility with ACID guarantees. No need for separate document store. |
| **Consul** | Kubernetes provides service discovery. Vault handles secrets. No need for separate tool. |
| **RabbitMQ** | NATS is faster, simpler, and handles our patterns (pub/sub, queues, streams) in one system. |
| **Elasticsearch** | Loki is sufficient for logs. PostgreSQL full-text search handles application search. |
| **Milvus** | Vector DB not needed unless adding semantic search/RAG. Not in current requirements. |

---

## Capacity Planning

### Target Scale (Year 1)

| Metric | Target | Infrastructure Required |
|--------|--------|------------------------|
| Tenants | 10,000 | Control plane: 3 instances |
| Concurrent Connections | 1,000,000 | 20 gateway instances (50K each) |
| Peak Messages/Second | 500,000 | NATS cluster: 3 nodes |
| STT Requests/Second | 10,000 | 50 STT workers (GPU) |
| TTS Requests/Second | 10,000 | 50 TTS workers (GPU) |
| LLM Requests/Second | 5,000 | External APIs + 10 Ollama instances |
| Storage (1 year) | 50 TB | PostgreSQL cluster + object storage |

### Resource Estimates Per Component

| Component | CPU | Memory | GPU | Storage | Instances |
|-----------|-----|--------|-----|---------|-----------|
| Gateway | 4 cores | 8 GB | - | - | 20 |
| STT Worker | 4 cores | 16 GB | 1x T4 | - | 50 |
| TTS Worker | 4 cores | 16 GB | 1x T4 | 10 GB (models) | 50 |
| LLM Worker (Ollama) | 8 cores | 32 GB | 1x A10 | 50 GB (models) | 10 |
| Control Plane | 2 cores | 4 GB | - | - | 3 |
| Redis Cluster | 8 cores | 64 GB | - | 100 GB | 6 |
| PostgreSQL | 16 cores | 128 GB | - | 2 TB | 3 |
| TimescaleDB | 8 cores | 64 GB | - | 1 TB | 3 |
| NATS Cluster | 4 cores | 16 GB | - | 100 GB | 3 |
| Vault | 2 cores | 4 GB | - | 10 GB | 3 |
| HAProxy | 8 cores | 16 GB | - | - | 3 |

---

## Data Models (High-Level)

### Core Entities

```
Tenant
├── tenant_id (UUID)
├── name
├── tier (free/pro/enterprise)
├── billing_id (Stripe customer ID)
├── settings (JSONB)
├── created_at
└── status (active/suspended/deleted)

Project
├── project_id (UUID)
├── tenant_id (FK)
├── name
├── environment (production/staging/development)
├── settings (JSONB)
└── created_at

APIKey
├── key_id (UUID)
├── project_id (FK)
├── key_hash (Argon2id)
├── key_prefix (first 8 chars for identification)
├── scopes (array)
├── rate_limit_tier
├── expires_at
└── created_at

Session
├── session_id (UUID)
├── project_id (FK)
├── tenant_id (denormalized for partitioning)
├── config (JSONB)
├── status (active/closed)
├── created_at
├── closed_at
└── metadata (JSONB)

ConversationItem
├── item_id (UUID)
├── session_id (FK)
├── tenant_id (partition key)
├── role (user/assistant/system/function)
├── content (JSONB, encrypted)
├── created_at
└── metadata (JSONB)

UsageRecord
├── record_id (UUID)
├── tenant_id (FK)
├── project_id (FK)
├── dimension (connection_minutes/audio_seconds/tokens)
├── quantity
├── timestamp
└── metadata (JSONB)
```

---

## API Surface

### Control Plane APIs (REST)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/admin/tenants` | POST | Create tenant |
| `/v1/admin/tenants/{id}` | GET/PATCH/DELETE | Manage tenant |
| `/v1/admin/projects` | POST | Create project |
| `/v1/admin/projects/{id}` | GET/PATCH/DELETE | Manage project |
| `/v1/admin/keys` | POST | Generate API key |
| `/v1/admin/keys/{id}` | GET/DELETE | Manage API key |
| `/v1/admin/keys/{id}/rotate` | POST | Rotate API key |
| `/v1/admin/usage` | GET | Query usage metrics |

### Data Plane APIs (REST + WebSocket)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/realtime` | WebSocket | Real-time voice session |
| `/v1/realtime/sessions` | POST | Create session (get ephemeral token) |
| `/v1/realtime/sessions/{id}` | GET/DELETE | Manage session |
| `/v1/audio/transcriptions` | POST | One-shot STT |
| `/v1/audio/speech` | POST | One-shot TTS |
| `/v1/tts/voices` | GET | List available voices |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |

---

## SLA Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Availability | 99.9% | Monthly uptime |
| WebSocket Latency (p99) | < 50ms | Message round-trip |
| STT Latency (p99) | < 500ms | Audio to transcript |
| TTS Time-to-First-Byte (p99) | < 200ms | Request to first audio chunk |
| API Latency (p99) | < 100ms | REST endpoints |
| Error Rate | < 0.1% | 5xx responses |

---

## SaaS Platform Requirements

### Requirement 19: Identity & Access Management (Keycloak)

**User Story:** As a SaaS operator, I want enterprise-grade identity management, so that tenants can manage users, roles, and SSO integration.

#### Acceptance Criteria

1. THE system SHALL use Keycloak as the identity provider for all user authentication and authorization
2. WHEN a tenant is created THEN the system SHALL provision a Keycloak realm with default roles (admin, developer, viewer)
3. THE system SHALL support authentication methods: username/password, social login (Google, GitHub), SAML 2.0, OIDC
4. WHEN a user logs in THEN the system SHALL issue JWT tokens with tenant_id, user_id, roles, and permissions claims
5. THE system SHALL provide self-service user management: registration, password reset, profile updates, MFA enrollment
6. WHEN an enterprise tenant requires SSO THEN the system SHALL support SAML 2.0 and OIDC identity provider federation
7. THE system SHALL enforce role-based access control (RBAC) with permissions: `api:read`, `api:write`, `billing:read`, `billing:write`, `admin:*`
8. THE system SHALL provide admin console for tenant administrators to manage users, roles, and groups
9. WHEN a user is deactivated THEN the system SHALL revoke all active sessions and API keys within 60 seconds

---

### Requirement 20: Billing & Subscription Management (Lago)

**User Story:** As a SaaS operator, I want comprehensive billing management, so that I can charge tenants based on usage with multiple payment methods.

#### Acceptance Criteria

1. THE system SHALL use Lago as the billing engine for usage-based pricing and subscription management
2. WHEN a tenant signs up THEN the system SHALL create a Lago customer record linked to tenant_id
3. THE system SHALL support pricing models: usage-based (per API call, per minute), subscription tiers (Free, Pro, Enterprise), hybrid (base + overage)
4. THE system SHALL meter usage dimensions: `api_requests`, `audio_minutes_input`, `audio_minutes_output`, `llm_tokens`, `concurrent_connections`
5. WHEN usage events occur THEN the system SHALL send metering data to Lago within 60 seconds (async, non-blocking)
6. THE system SHALL integrate payment processors: Stripe (credit cards, ACH), PayPal (PayPal balance, cards)
7. WHEN a billing period ends THEN the system SHALL generate invoices automatically and send to customer email
8. THE system SHALL support billing operations: refunds (full/partial), credits, proration, dunning (failed payment retry)
9. WHEN payment fails THEN the system SHALL retry 3 times over 7 days, then suspend tenant with 48-hour grace period
10. THE system SHALL provide webhook events: `invoice.created`, `invoice.paid`, `invoice.failed`, `subscription.updated`, `payment.refunded`

---

### Requirement 21: Customer Self-Service Portal

**User Story:** As a tenant administrator, I want a self-service portal, so that I can manage my account, view usage, and handle billing without contacting support.

#### Acceptance Criteria

1. THE system SHALL provide a web-based customer portal accessible at `portal.agentvoicebox.com`
2. WHEN a user logs in THEN the portal SHALL authenticate via Keycloak and display tenant-specific data
3. THE portal SHALL display dashboard with: current usage, billing summary, API health status, recent activity
4. THE portal SHALL provide API key management: create, rotate, revoke, set permissions, view usage per key
5. THE portal SHALL display usage analytics: charts for API calls, audio minutes, tokens over time (hourly/daily/monthly)
6. THE portal SHALL provide billing section: current plan, usage breakdown, invoice history, payment methods, upgrade/downgrade
7. WHEN viewing invoices THEN the portal SHALL allow download as PDF and display line-item breakdown
8. THE portal SHALL provide settings: organization profile, notification preferences, webhook configuration, team members
9. THE portal SHALL support team management: invite users, assign roles, remove users, transfer ownership
10. THE portal SHALL be responsive (mobile-friendly) and accessible (WCAG 2.1 AA compliant)

---

### Requirement 22: Payment Processing Integration

**User Story:** As a tenant, I want to pay using my preferred payment method, so that I can easily manage my subscription.

#### Acceptance Criteria

1. THE system SHALL integrate Stripe for credit/debit card payments with PCI DSS compliance (via Stripe Elements)
2. THE system SHALL integrate PayPal for PayPal balance and alternative card payments
3. WHEN a customer adds a payment method THEN the system SHALL tokenize and store securely (no raw card data stored)
4. THE system SHALL support automatic recurring payments on billing cycle (monthly/annual)
5. WHEN a payment is processed THEN the system SHALL send confirmation email with receipt
6. THE system SHALL support manual payments: pay now, pay invoice, add credits
7. WHEN a refund is requested THEN the system SHALL process via original payment method within 5-10 business days
8. THE system SHALL display payment history with status: pending, completed, failed, refunded
9. THE system SHALL support multiple currencies: USD, EUR, GBP with automatic conversion
10. WHEN tax is applicable THEN the system SHALL calculate and display tax based on customer location (VAT, sales tax)

---

### Requirement 23: Subscription & Plan Management

**User Story:** As a SaaS operator, I want flexible subscription plans, so that I can offer different tiers to different customer segments.

#### Acceptance Criteria

1. THE system SHALL support subscription tiers:
   - **Free**: 100 API calls/month, 10 audio minutes, community support
   - **Pro**: 10,000 API calls/month, 1,000 audio minutes, email support, $49/month
   - **Enterprise**: Unlimited API calls, unlimited audio, dedicated support, custom pricing
2. WHEN a tenant upgrades THEN the system SHALL apply new limits immediately and prorate billing
3. WHEN a tenant downgrades THEN the system SHALL apply at end of current billing period
4. THE system SHALL support add-ons: additional API calls, additional audio minutes, priority support
5. WHEN usage exceeds plan limits THEN the system SHALL either throttle (Free) or charge overage (Pro/Enterprise)
6. THE system SHALL support annual billing with discount (2 months free)
7. WHEN a tenant cancels THEN the system SHALL retain access until end of paid period and offer win-back
8. THE system SHALL support trial periods: 14-day free trial of Pro features for new signups
9. WHEN trial expires THEN the system SHALL downgrade to Free tier unless payment method added

---

### Requirement 24: Tenant Onboarding Flow

**User Story:** As a new customer, I want a smooth onboarding experience, so that I can start using the API quickly.

#### Acceptance Criteria

1. WHEN a user signs up THEN the system SHALL create: Keycloak user, Keycloak realm (tenant), Lago customer, default project, first API key
2. THE onboarding flow SHALL collect: email, password, organization name, use case (optional)
3. WHEN signup completes THEN the system SHALL send welcome email with: API key, quickstart guide link, support contact
4. THE system SHALL provide interactive quickstart: test API call, view response, see usage update
5. WHEN first API call succeeds THEN the system SHALL display success celebration and next steps
6. THE system SHALL offer optional guided tour of portal features
7. WHEN a tenant is inactive for 7 days THEN the system SHALL send engagement email with tips
8. THE system SHALL track onboarding completion: signup → first API call → first successful response → payment method added


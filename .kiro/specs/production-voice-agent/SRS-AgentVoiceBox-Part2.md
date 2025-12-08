# AgentVoiceBox SRS - Part 2: Extended Requirements

## 13. File Handling Requirements

### 13.1 Audio File Management

**AVB-FH-001:** THE AgentVoiceBox system SHALL store temporary audio files in memory-mapped tmpfs volumes.

**AVB-FH-002:** THE AgentVoiceBox system SHALL purge temporary audio files older than 5 minutes.

**AVB-FH-003:** WHEN audio exceeds 10MB per session THEN AgentVoiceBox SHALL stream to MinIO/S3.

**AVB-FH-004:** THE AgentVoiceBox system SHALL use content-addressable storage for audio deduplication.

**AVB-FH-005:** THE AgentVoiceBox system SHALL validate audio integrity using SHA-256 checksums.

### 13.2 Model File Management

**AVB-FH-010:** THE AgentVoiceBox model files SHALL be stored on ReadOnlyMany persistent volumes.

**AVB-FH-011:** THE AgentVoiceBox system SHALL verify model integrity at startup using SHA-256.

**AVB-FH-012:** WHEN model checksum fails THEN AgentVoiceBox SHALL refuse to start and emit alert.

### 13.3 Log File Management

**AVB-FH-020:** THE AgentVoiceBox system SHALL write logs to stdout/stderr for container collection.

**AVB-FH-021:** THE AgentVoiceBox system SHALL compress rotated logs using zstd.

---

## 14. Compression Requirements

### 14.1 Audio Compression

**AVB-CMP-001:** THE AgentVoiceBox system SHALL support Opus codec for WebSocket transmission.

**AVB-CMP-002:** THE AgentVoiceBox system SHALL support FLAC for audio archival.

**AVB-CMP-003:** THE AgentVoiceBox audio compression SHALL achieve minimum 10:1 ratio for voice.

### 14.2 Message Compression

**AVB-CMP-010:** THE AgentVoiceBox WebSocket SHALL support per-message deflate (RFC 7692).

**AVB-CMP-011:** THE AgentVoiceBox system SHALL use zstd level 3 for optimal speed/ratio.

### 14.3 Data Compression

**AVB-CMP-020:** THE AgentVoiceBox PostgreSQL SHALL use TOAST compression for JSONB.

**AVB-CMP-021:** THE AgentVoiceBox Redis SHALL use LZF compression for RDB snapshots.

**AVB-CMP-022:** THE AgentVoiceBox backups SHALL use zstd level 9.

---

## 15. Failsafe and Recovery Requirements

### 15.1 Circuit Breaker Pattern


**AVB-FS-001:** THE AgentVoiceBox system SHALL implement circuit breakers using Tenacity library.

**AVB-FS-002:** THE AgentVoiceBox circuit breaker SHALL open after 5 consecutive failures.

**AVB-FS-003:** WHILE circuit breaker open THEN AgentVoiceBox SHALL return degraded response.

**AVB-FS-004:** THE AgentVoiceBox circuit breaker SHALL retry every 30 seconds.

### 15.2 Retry Mechanisms

**AVB-FS-010:** THE AgentVoiceBox system SHALL use exponential backoff: base=1s, max=60s.

**AVB-FS-011:** THE AgentVoiceBox system SHALL limit retries to 5 attempts maximum.

**AVB-FS-012:** THE AgentVoiceBox system SHALL NOT retry auth errors or rate limits.

### 15.3 Timeout Management

| Operation | Timeout | Action on Timeout |
|-----------|---------|-------------------|
| Redis read | 100ms | Return cached/error |
| Redis write | 500ms | Queue for retry |
| PostgreSQL read | 1s | Return cached/error |
| PostgreSQL write | 5s | Queue for retry |
| LLM API | 30s | Return fallback |
| STT processing | 10s | Return partial/error |
| TTS processing | 10s | Return text-only |

### 15.4 Dead Letter Queue

**AVB-FS-040:** THE AgentVoiceBox system SHALL route failed messages to DLQ after max retries.

**AVB-FS-041:** THE AgentVoiceBox DLQ SHALL retain messages for 7 days.

**AVB-FS-042:** THE AgentVoiceBox system SHALL alert when DLQ depth exceeds 1000.

### 15.5 Graceful Degradation Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| Text-Only | TTS unavailable | Send transcript without audio |
| Audio-Only | LLM unavailable | Echo without AI response |
| Read-Only | PostgreSQL down | Serve from Redis, queue writes |
| Offline | Redis down | Reject new, serve existing |

---

## 16. Backup and Disaster Recovery

### 16.1 PostgreSQL Backup (pgBackRest)

**AVB-BK-001:** THE AgentVoiceBox system SHALL use pgBackRest for PostgreSQL backup.

**AVB-BK-002:** Full backup: weekly (Sunday 02:00 UTC).

**AVB-BK-003:** Incremental backup: every 6 hours.

**AVB-BK-004:** Continuous WAL archiving for point-in-time recovery.

**AVB-BK-005:** Backups stored in separate availability zone.

| Type | Retention |
|------|-----------|
| Hourly incremental | 24 hours |
| Daily incremental | 7 days |
| Weekly full | 4 weeks |
| Monthly full | 12 months |

**AVB-BK-007:** Backups encrypted using AES-256.

### 16.2 Redis Backup

**AVB-BK-010:** RDB snapshots every 15 minutes.

**AVB-BK-011:** AOF persistence with fsync every second.

**AVB-BK-012:** Replicate to object storage hourly.

### 16.3 Disaster Recovery

**AVB-BK-020:** RPO: 1 hour.

**AVB-BK-021:** RTO: 4 hours.

**AVB-BK-022:** Test DR procedures quarterly.

---

## 17. Telemetry and Observability

### 17.1 Metrics (Prometheus)

**AVB-TEL-001:** ALL services SHALL expose `/metrics` endpoint.

**AVB-TEL-002:** Naming: `agentvoicebox_<component>_<metric>_<unit>`.

#### Gateway Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `agentvoicebox_gateway_connections_total` | Counter | Total connections |
| `agentvoicebox_gateway_connections_active` | Gauge | Active connections |
| `agentvoicebox_gateway_messages_total` | Counter | Messages processed |
| `agentvoicebox_gateway_message_duration_seconds` | Histogram | Processing latency |

#### STT Worker Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `agentvoicebox_stt_requests_total` | Counter | STT requests |
| `agentvoicebox_stt_duration_seconds` | Histogram | Transcription latency |
| `agentvoicebox_stt_queue_depth` | Gauge | Pending requests |
| `agentvoicebox_stt_gpu_utilization` | Gauge | GPU usage % |

#### TTS Worker Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `agentvoicebox_tts_requests_total` | Counter | TTS requests |
| `agentvoicebox_tts_duration_seconds` | Histogram | Synthesis time |
| `agentvoicebox_tts_first_chunk_seconds` | Histogram | TTFB |
| `agentvoicebox_tts_queue_depth` | Gauge | Pending requests |

#### LLM Worker Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `agentvoicebox_llm_requests_total` | Counter | LLM requests |
| `agentvoicebox_llm_duration_seconds` | Histogram | Inference time |
| `agentvoicebox_llm_tokens_total` | Counter | Tokens processed |
| `agentvoicebox_llm_circuit_breaker_state` | Gauge | 0=closed, 1=open |

#### Infrastructure Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `agentvoicebox_redis_commands_total` | Counter | Redis commands |
| `agentvoicebox_redis_duration_seconds` | Histogram | Redis latency |
| `agentvoicebox_postgres_queries_total` | Counter | PostgreSQL queries |
| `agentvoicebox_postgres_duration_seconds` | Histogram | PostgreSQL latency |

### 17.2 Distributed Tracing (Jaeger/OpenTelemetry)

**AVB-TEL-020:** Implement OpenTelemetry tracing for all services.

**AVB-TEL-021:** Propagate trace context via W3C headers.

**AVB-TEL-022:** Export traces to Jaeger collector.

**AVB-TEL-023:** Sample at 1% in production (100% for errors).

### 17.3 Structured Logging (Loki)

**AVB-TEL-030:** Output logs in JSON format.

**AVB-TEL-031:** Standard fields: timestamp, level, service, trace_id, session_id, tenant_id.

**AVB-TEL-032:** Ship logs to Loki via Promtail.

**AVB-TEL-033:** Retain logs for 30 days.

### 17.4 Alerting (Alertmanager)

| Alert | Condition | Severity |
|-------|-----------|----------|
| HighErrorRate | error_rate > 1% for 5m | Critical |
| HighLatency | p99 > 2s for 5m | Critical |
| RedisDown | redis_up == 0 for 1m | Critical |
| PostgresDown | postgres_up == 0 for 1m | Critical |
| QueueBacklog | queue_depth > 10000 for 5m | Warning |
| HighMemory | memory > 90% for 10m | Warning |

### 17.5 Dashboards (Grafana)

| Dashboard | Purpose |
|-----------|---------|
| Overview | System health, SLO status |
| Gateway | Connections, throughput, latency |
| Workers | STT/TTS/LLM performance |
| Infrastructure | Redis, PostgreSQL metrics |
| Tenant | Per-tenant usage |
| Alerts | Active and historical alerts |

---

## 18. Network Security

### 18.1 TLS Configuration

**AVB-NET-001:** Use TLS 1.3 for all external connections.

**AVB-NET-002:** Cipher suites: TLS_AES_256_GCM_SHA384, TLS_CHACHA20_POLY1305_SHA256.

**AVB-NET-003:** Disable SSLv3, TLS 1.0, TLS 1.1.

**AVB-NET-004:** Enable HSTS max-age=31536000.

### 18.2 Certificate Management (cert-manager)

**AVB-NET-010:** Use cert-manager for automatic provisioning.

**AVB-NET-011:** Support Let's Encrypt for public certs.

**AVB-NET-012:** Use internal CA for service-to-service mTLS.

### 18.3 Network Policies

| Source | Destination | Port |
|--------|-------------|------|
| Load Balancer | Gateway | 8000 |
| Gateway | Redis | 6379 |
| Gateway | PostgreSQL | 5432 |
| Workers | Redis | 6379 |
| Prometheus | All | 9090 |

**AVB-NET-023:** Deny all other traffic by default.

---

## 19. Open Source Technology Stack

### Core Infrastructure

| Component | Technology | License |
|-----------|------------|---------|
| Load Balancer | HAProxy 2.8+ | GPL-2.0 |
| API Gateway | FastAPI 0.109+ | MIT |
| Async Runtime | uvloop 0.19+ | MIT |

### Data Storage

| Component | Technology | License |
|-----------|------------|---------|
| Cache/State | Redis Cluster 7.2+ | BSD-3 |
| Database | PostgreSQL 16+ | PostgreSQL |
| Object Storage | MinIO | AGPL-3.0 |
| Connection Pool | PgBouncer 1.21+ | ISC |

### AI/ML

| Component | Technology | License |
|-----------|------------|---------|
| STT Engine | Faster-Whisper 1.0+ | MIT |
| TTS Engine | Kokoro-ONNX 1.0+ | Apache-2.0 |
| TTS Fallback | Piper 1.2+ | MIT |
| ML Runtime | ONNX Runtime 1.16+ | MIT |

### Observability

| Component | Technology | License |
|-----------|------------|---------|
| Metrics | Prometheus 2.48+ | Apache-2.0 |
| Dashboards | Grafana 10+ | AGPL-3.0 |
| Tracing | Jaeger 1.52+ | Apache-2.0 |
| Logging | Loki 2.9+ | AGPL-3.0 |
| Alerting | Alertmanager 0.26+ | Apache-2.0 |

### Security

| Component | Technology | License |
|-----------|------------|---------|
| Secrets | HashiCorp Vault 1.15+ | BUSL-1.1 |
| Certificates | cert-manager 1.13+ | Apache-2.0 |
| Scanning | Trivy 0.48+ | Apache-2.0 |

### Deployment

| Component | Technology | License |
|-----------|------------|---------|
| Containers | Docker 24+ | Apache-2.0 |
| Orchestration | Kubernetes 1.28+ | Apache-2.0 |
| Package Manager | Helm 3.13+ | Apache-2.0 |
| GitOps | ArgoCD 2.9+ | Apache-2.0 |
| Backup | pgBackRest 2.49+ | MIT |
| Compression | zstd 1.5+ | BSD/GPL |

---

**Document:** AgentVoiceBox SRS Part 2  
**Total Requirements:** 200+  
**Compliance:** ISO/IEC/IEEE 29148:2018

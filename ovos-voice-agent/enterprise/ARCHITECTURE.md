# Enterprise Architecture Plan

This document describes the enterprise-grade topology for the OVOS voice agent. It enumerates data
stores, message flows, observability surfaces, and deployment considerations.

## Data Persistence (Postgres)

Tables | Purpose
---|---
`sessions` | Track realtime session lifecycle, persona configuration, and timestamps.
`conversation_items` | Conversation history, structured as JSONB for assistant/user/tool turns.
`tool_invocations` *(future)* | Audit trail of tool requests/responses executed during a session.
`policy_audit` *(future)* | Persist OPA decisions and metadata for compliance reviews.

Migrations will be managed via Alembic. All tables should include tenant identifiers when multi-tenant
support is enabled.

## Event Streaming (Kafka)

Topic | Key | Description
---|---|---
`voice.sessions` | `session_id` | Session lifecycle events (created/updated/closed) for downstream consumers.
`voice.audio.inbound` | `session_id` | PCM16 audio chunks streamed from gateway to speech workers.
`voice.audio.transcribed` | `session_id` | Transcription results emitted by the speech pipeline.
`voice.responses` | `session_id` | Assistant responses (text + audio references) destined for clients.
`voice.audit` | `session_id` | Structured audit records, including policy denials and tool interactions.

Each topic will use Avro/JSON Schema registry enforcement. Partitioning will follow `session_id` so
all events for a conversation remain ordered.

## Policy Enforcement (OPA)

Policies are housed under `voice/` package with the following entrypoints:

- `voice/allow` – global request guard (session bootstrap, client secret creation).
- `voice/tool` – authorize tool usage with actor/tenant metadata.
- `voice/audio` – optional content moderation pipeline for outbound audio.

OPA will run as a sidecar. Flask requests include contextual headers
(`X-Actor`, `X-Tenant`, `X-Scopes`) and JSON payloads for evaluation. Kafka consumers may also
consult OPA to validate tool invocations before execution.

## Observability

- **Metrics**: Prometheus endpoint `/metrics` publishes counters/gauges/histograms seeded in
  `app/observability/metrics.py`. Additional metrics (Kafka lag, TTS latency) will be added in their
  respective workers using the shared registry.
- **Logging**: JSON-formatted logs with correlation fields (`session_id`, `trace_id`). Sentry support
  is available via DSN configuration.
- **Tracing** *(future)*: When `OBSERVABILITY__ENABLE_TRACING=true`, OpenTelemetry exporters will be
  wired for distributed tracing across gateway and workers.

## Deployment Notes

- Gateway served via Gunicorn or uWSGI with `wsgi.py`. For WebSocket/WebRTC, prefer Hypercorn
  (ASGI) or Flask-Sock, keeping compatibility with existing infrastructure.
- Kafka/Postgres/OPA endpoints are read from environment variables; a `settings.example.env` helper is
  provided.
- Secrets should be injected via secret management (Vault, AWS SM) rather than `.env` in production.

## Alignment with Somabrain / SomagentHub

While no direct hooks exist in the repository, the modular design keeps integration points at the
Kafka layer and REST gateway. Somabrain and SomagentHub services can subscribe to the Kafka topics or
call the gateway APIs to orchestrate downstream workflows without modifying core voice processing.

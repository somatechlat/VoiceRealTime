# VoiceEngine Parity Roadmap

This roadmap hardens the VoiceEngine control plane so it can act as a drop-in replacement for the OpenAI
Realtime API while remaining fully aligned with OVOS architecture and enterprise deployment needs. Capacity
is unconstrained, so Sprints A–C execute in parallel with shared acceptance checkpoints.

## Guiding Principles

- **Contract fidelity**: JSON payloads, status codes, headers, and websocket events must match OpenAI
  expectations byte-for-byte so existing SDKs work without modification.
- **Policy-first security**: Every entry point (REST, WebSocket, WebRTC) enforces OPA policies, project-scoped
  credentials, and auditable request IDs.
- **Observability everywhere**: Metrics, structured logs, and traces expose per-call health, latency, and
  rate-limit decisions.
- **OVOS cooperation**: Listener, speech pipeline, skills bus, and transformers integrate seamlessly without
  forked logic.

## Sprint A – Gateway Contract (REST)

> Owner: API squad · Dependencies: database migrations, OPA policies · Exit: Contract tests against OpenAI SDK

- Introduce Pydantic schemas that mirror OpenAI `POST /v1/realtime/client_secrets` and `/v1/realtime/sessions`.
- Persist full session state (`model`, `output_modalities`, `audio` configs, expiry) in `SessionModel`.
- Enforce Bearer token authentication with project scoping, emitting rate-limit and request-id headers.
- Normalize error envelopes (`type`, `code`, `message`, `param`) and increment Prometheus counters for denials.
- Deliver OpenAPI spec (`enterprise/openapi/realtime.yaml`) plus pytest contract suite.
- Containerize the gateway with a reproducible Dockerfile and hook it into Compose for contract tests.

## Sprint B – Transport & Calls (Realtime)

> Owner: Realtime squad · Dependencies: Sprint A session schema · Exit: WebRTC & WebSocket harness passing

- Implement `/v1/realtime/calls*` REST handlers for WebRTC/SIP, emitting SDP answers and `Location` headers.
- Require REST-issued ephemeral keys on all websocket handshakes; hydrate connection state from persistence. ✅
- Align outbound events with spec (`response.audio.delta`, `rate_limits.updated`, cancellations, etc.).
- Build automated conformance harness using OpenAI JS SDK and pytest-asyncio smoke flows.
- Provide configurable audio transcoding hooks to bridge OVOS listener and realtime streaming.
- Extend the Docker Compose stack with realtime transports (Redis/Kafka bridges, TURN/WebRTC sidecars) for
  end-to-end rehearsal in CI.

## Sprint C – OVOS Integration & QA

> Owner: Platform squad · Dependencies: Sprint A+B interfaces · Exit: End-to-end voice conversation demo

- Plug OVOS listener/speech pipeline into realtime events; route assistant responses through skills bus.
- Extend metrics dashboards (Prometheus/Grafana) with policy denials, latency buckets, and call success ratios.
- Author resilience tests for persona swaps, multilingual wake words, and continuous mode per OVOS manual.
- Publish runbooks for deployment, incident response, and troubleshooting; wire CI smoke tests into GitHub Actions.
- Harden container images with health probes, resource limits, and optional autoscaling manifests; promote a
  "single-command" Compose demo for field enablement.

## Cross-Cutting Workstreams

- **Security**: Rotate keys automatically, store secrets in Vault-compatible backend, add audit logging for PII.
- **Data layer**: Add Alembic migrations, connection pooling, and backfill scripts for legacy session data.
- **Docs & Enablement**: Maintain living architecture diagrams, API reference, and onboarding flows in `/docs`.
- **Deployment**: Produce Docker/Compose definitions (`VoiceEngine-*` services), enforce image scans, generate
  Makefile automation for lint/test/package, and provide Terraform/Kubernetes manifests as stretch goals.

## Milestones & Checkpoints

| Week | Target | Acceptance |
| --- | --- | --- |
| Week 1 | Sprint A contract tests green | All REST endpoints validated by Postman + pytest schema harness |
| Week 2 | Sprint B realtime harness passes | WebRTC demo call bridges OVOS audio without schema drift |
| Week 3 | Sprint C E2E demo & dashboards | Recorded OVOS conversation, dashboards populated, runbooks merged |
| Week 4 | Deployment track GA | Hardened container images, Compose stack, and Makefile automation published |

## Immediate Next Actions

1. Scaffold OpenAPI contract and supporting Pydantic models (Sprint A).
2. Draft database migration for expanded session schema.
3. Stand up realtime conformance harness skeleton to unblock Sprint B test development. ✅
4. Begin OVOS listener integration spike to identify adaptors needed for Sprint C.
5. Stand up canonical Dockerfile + Compose baseline to unblock shared test environments (Deployment track).

## Status Tracking

Progress uses the shared TODO list in `VOICE_AGENT_ROADMAP.md` and issue tracker labels `Sprint-A`, `Sprint-B`,
`Sprint-C`. Update this document when milestones shift or new cross-cutting concerns emerge.

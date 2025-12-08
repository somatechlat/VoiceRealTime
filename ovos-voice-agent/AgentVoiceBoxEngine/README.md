# Enterprise OVOS Voice Agent Gateway

This module scaffolds an enterprise-ready control plane for the OVOS voice agent. It mirrors the
OpenAI Realtime API surface while keeping all inference and policy execution on infrastructure you
control.

## Components

- **Flask API Gateway** (`enterprise/app`): issues client secrets, manages session metadata, and
  exposes REST endpoints for health and bootstrap flows.
- **Observability** (`observability/`): structured JSON logging, Prometheus metrics, optional Sentry
  integration.
- **Persistence** (`models/`, `utils/database.py`): SQLAlchemy models for sessions and conversation
  items, ready for Postgres.
- **Policy Enforcement** (`services/opa_client.py`): lightweight OPA client suitable for embedding in
  request handlers before generating realtime tokens.
- **Event Streaming Hooks** (`services/kafka_client.py`): Kafka factory helpers for wiring producers
  and consumers that will back speech-to-speech workers.
- **Realtime WebSocket Transport** (`app/transports/`): Flask-Sock powered `/v1/realtime` endpoint that
  validates ephemeral secrets, enforces OPA policy, and emits OpenAI-compatible events.

## Getting Started

1. Create a virtual environment and install `requirements-enterprise.txt`.
2. Copy `settings.example.env` to `.env` and fill in Postgres, Kafka, and OPA endpoints.
3. Run `gunicorn "app:create_app()"` or `flask --app app:create_app run` for development.
4. Metrics are exposed at `/metrics`, health at `/health`, and realtime bootstrap endpoints live
   under `/v1/realtime/*`.

## Testing

Integration tests ship with the realtime gateway to verify the client-secret and session bootstrap
flow. After installing dependencies, execute from the `enterprise/` directory (or the repository
root):

```bash
pytest enterprise/tests/test_realtime_routes.py
pytest enterprise/tests/test_realtime_websocket.py
```

The suite provisions an ephemeral SQLite database and stubs the OPA client, so no additional
infrastructure is required to exercise the REST contract locally.

## Realtime WebSocket Transport

1. Request an ephemeral client secret via `POST /v1/realtime/client_secrets`.
2. Hydrate the session by calling `POST /v1/realtime/sessions` with the returned secret.
3. Connect to the gateway using a WebSocket client and supply the secret in the
  `Authorization: Bearer <client_secret>` header when negotiating `ws://<host>/v1/realtime`.
4. The server emits `session.created` + `rate_limits.updated` events immediately and now supports
  `session.update`, `conversation.item.create`, `response.create`, and audio buffer events.

Session and conversation updates are persisted to the same SQL backend used by the REST bootstrap
flow, ensuring parity between REST and realtime transports. Automated coverage lives in
`enterprise/tests/test_realtime_websocket.py` and exercises authentication, persistence, and
response generation stubs.

## Containerized Deployment

Ship and run the entire stack (gateway, Postgres, Kafka, OPA, Prometheus) with Docker:

```bash
cd enterprise
cp settings.docker.env .env # customise secrets as needed
make docker-build
make docker-up
```

The API will be reachable at <http://localhost:8000> and exposes metrics at <http://localhost:9090>.
Stop and clean up with `make docker-down`. The Compose stack uses Apache Kafka in KRaft mode—no
ZooKeeper required—and ships with health checks plus persistent volumes for Postgres and Kafka data.

### Makefile Automation

A convenience `Makefile` unlocks repeatable workflows:

- `make install-dev` – install runtime + dev dependencies (black, ruff, pytest).
- `make format` / `make lint` / `make check` – enforce code style with Black and Ruff.
- `make pytest-contracts` – run the realtime integration harness.
- `make docker-build` / `make docker-up` / `make docker-down` – manage container lifecycle.
- `make docker-logs` – tail container logs for quick troubleshooting.

## Next Steps

- Implement Kafka-backed workers that consume audio and publish responses.
- Integrate WebSocket/WebRTC handling via Flask-Sock or Hypercorn ASGI for low-latency streaming.
- Expand OPA policies to cover tool invocation, persona changes, and rate limits.
- Add migrations (Alembic) and CI checks to enforce schema compatibility.

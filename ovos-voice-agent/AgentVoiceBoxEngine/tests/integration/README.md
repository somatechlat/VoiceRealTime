# Integration Tests

Comprehensive end-to-end tests against real infrastructure. NO MOCKS.

## Test Categories

| File | Coverage | Requirements |
|------|----------|--------------|
| `test_user_flows.py` | User signup, voice conversations, settings | 7, 10, 11, 12, 24 |
| `test_postgres.py` | Database operations, tenant isolation | 13.1, 13.2, 13.5 |
| `test_redis.py` | Session management, rate limiting | 9.1, 9.2, 6.1, 6.2 |
| `test_websocket_gateway.py` | WebSocket lifecycle, connections | 7.1, 7.2, 7.4, 7.6 |
| `test_worker_pipeline.py` | STT/TTS/LLM workers | 10, 11, 12 |
| `test_e2e_speech_pipeline.py` | Full audio pipeline latency | 14.2 |
| `test_e2e_billing.py` | Billing, subscriptions, payments | 20, 22 |
| `test_e2e_onboarding.py` | Signup, onboarding flow | 24 |
| `test_auth_multitenancy.py` | Auth, tenant isolation | 1, 3, 19 |

## Running Tests

### 1. Start Infrastructure

```bash
docker compose -f docker-compose.test.yml up -d
```

### 2. Wait for Services

```bash
# Check all services are healthy
docker compose -f docker-compose.test.yml ps
```

### 3. Run Tests

```bash
# All integration tests
pytest tests/integration/ -v

# Specific test file
pytest tests/integration/test_user_flows.py -v

# Specific test class
pytest tests/integration/test_user_flows.py::TestVoiceConversationFlow -v
```

### 4. Cleanup

```bash
docker compose -f docker-compose.test.yml down -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_URL` | `http://localhost:25000` | Gateway HTTP URL |
| `GATEWAY_WS_URL` | `ws://localhost:25000` | Gateway WebSocket URL |
| `PORTAL_URL` | `http://localhost:28000` | Portal API URL |
| `REDIS_URL` | `redis://localhost:16379/0` | Redis URL |
| `DATABASE_URL` | `postgresql://...@localhost:15432/...` | PostgreSQL URL |
| `KEYCLOAK_URL` | `http://localhost:18080` | Keycloak URL |

## Test Flows Covered

### User Flows
- Complete signup and onboarding
- Voice conversation (audio → STT → LLM → TTS → audio)
- Change voice mid-session
- Change instructions mid-session
- Cancel response mid-generation
- Session reconnection

### Admin Flows
- Create/update/suspend tenant
- Create/rotate/revoke API keys
- Invite/manage team members
- Change user roles

### Billing Flows
- View subscription and usage
- Upgrade/downgrade plans
- Add payment methods
- View invoices

### Server Flows
- Rate limiting enforcement
- Session state persistence
- Graceful degradation
- Health endpoints

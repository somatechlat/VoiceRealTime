# Integration Tests

Real infrastructure integration tests for AgentVoiceBox.

**NO MOCKS, NO FAKES, NO STUBS** - All tests run against real services.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ with pytest, pytest-asyncio
- At least 10GB RAM available for the test stack

## Quick Start

```bash
# 1. Initialize volume directories
./scripts/init-volumes.sh

# 2. Start the test infrastructure
docker compose -f docker-compose.test.yml up -d

# 3. Wait for services to be healthy
docker compose -f docker-compose.test.yml ps

# 4. Run all integration tests
pytest tests/integration/ -v

# 5. Stop the infrastructure
docker compose -f docker-compose.test.yml down
```

## Test Suites

### Redis Integration Tests (`test_redis_real.py`)
- Distributed session management across gateways
- Rate limiter accuracy under concurrent load
- Redis Streams consumer groups
- Session TTL expiration and cleanup
- Tenant isolation

### PostgreSQL Integration Tests (`test_postgres_real.py`)
- Database connectivity and health checks
- Tenant isolation in queries
- Conversation item persistence
- Audit log writes and queries
- Transaction rollback behavior

### WebSocket Gateway Tests (`test_websocket_gateway.py`)
- Connection lifecycle (connect, auth, messages, disconnect)
- Session reconnection to different gateway
- Concurrent connections (50+)
- Error handling and format

### Worker Pipeline Tests (`test_worker_pipeline.py`)
- STT worker audio processing
- TTS worker streaming synthesis
- LLM worker request handling
- Consumer group behavior

### End-to-End Speech Pipeline (`test_e2e_speech_pipeline.py`)
- STT latency (p95 < 500ms)
- TTS TTFB (p95 < 200ms)
- Concurrent session handling

### Authentication & Multi-Tenancy (`test_auth_multitenancy.py`)
- API key validation
- Ephemeral token flow
- Cross-tenant access denial

## RAM Budget (10GB Total)

| Service | Memory Limit |
|---------|-------------|
| Redis | 512MB |
| PostgreSQL | 512MB |
| Gateway x2 | 512MB each |
| STT Worker | 2.5GB |
| TTS Worker | 2GB |
| LLM Worker | 512MB |
| Buffer | ~3GB |

## Environment Variables

```bash
# Non-conflicting ports (16xxx, 15xxx, 18xxx)
REDIS_URL=redis://localhost:16379/0
DATABASE_URI=postgresql://agentvoicebox:agentvoicebox_secure_pwd_2024@localhost:15432/agentvoicebox
GATEWAY_1_URL=ws://localhost:18000
GATEWAY_2_URL=ws://localhost:18001
GATEWAY_1_HTTP=http://localhost:18000
GATEWAY_2_HTTP=http://localhost:18001
TEST_API_KEY=test-api-key-for-integration
```

## Running Individual Test Suites

```bash
# Redis tests only
pytest tests/integration/test_redis_real.py -v

# PostgreSQL tests only
pytest tests/integration/test_postgres_real.py -v

# WebSocket tests only
pytest tests/integration/test_websocket_gateway.py -v

# Worker tests only
pytest tests/integration/test_worker_pipeline.py -v

# E2E tests only
pytest tests/integration/test_e2e_speech_pipeline.py -v

# Auth tests only
pytest tests/integration/test_auth_multitenancy.py -v
```

## Troubleshooting

### Services not starting
```bash
# Check logs
docker compose -f docker-compose.test.yml logs redis
docker compose -f docker-compose.test.yml logs postgres

# Verify health
docker compose -f docker-compose.test.yml ps
```

### Tests timing out
- Ensure workers are running and healthy
- Check Redis connectivity
- Verify PostgreSQL is accepting connections

### Memory issues
- Reduce concurrent test count
- Use `--workers 1` for pytest-xdist
- Check Docker memory limits

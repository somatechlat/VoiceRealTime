# AgentVoiceBox Local Development Guide

This guide covers setting up AgentVoiceBox for local development using Docker Compose.

## Prerequisites

- Docker Desktop 4.x+ with Docker Compose v2
- 10GB+ available RAM for Docker
- Git
- Python 3.11+ (for running tests locally)
- Node.js 20+ (for portal frontend development)

## Quick Start

### 1. Clone and Setup

```bash
cd ovos-voice-agent/AgentVoiceBoxEngine
```

### 2. Environment Configuration

Copy the example environment file:

```bash
cp settings.docker.env .env
```

Edit `.env` to configure:

```bash
# LLM Provider (required for voice responses)
LLM_DEFAULT_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here

# Or use OpenAI
# LLM_DEFAULT_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the Stack

```bash
docker compose -p agentvoicebox up -d
```

This starts all services:
- PostgreSQL (port 25002)
- Redis (port 25003)
- Keycloak (port 25004)
- Lago Billing (port 25005)
- Gateway API (port 25000)
- Portal API (port 25001)
- Portal Frontend (port 25007)
- STT Worker
- TTS Worker
- LLM Worker
- Prometheus (port 25008)
- Grafana (port 25009)

### 4. Verify Services

Check all services are healthy:

```bash
docker compose -p agentvoicebox ps
```

Test the gateway:

```bash
curl http://localhost:25000/health
```

## Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| Gateway API | http://localhost:25000 | WebSocket/REST API |
| Portal API | http://localhost:25001 | Customer portal backend |
| PostgreSQL | localhost:25002 | Database |
| Redis | localhost:25003 | Session store |
| Keycloak | http://localhost:25004 | Identity provider |
| Lago | http://localhost:25005 | Billing engine |
| Portal Frontend | http://localhost:25007 | Customer portal UI |
| Prometheus | http://localhost:25008 | Metrics |
| Grafana | http://localhost:25009 | Dashboards |
| Admin UI | http://localhost:25011 | Test interfaces |

## Default Credentials

### Keycloak Admin
- URL: http://localhost:25004/admin
- Username: `admin`
- Password: `admin`

### Grafana
- URL: http://localhost:25009
- Username: `admin`
- Password: `admin`

### PostgreSQL
- Host: localhost:25002
- Database: `agentvoicebox`
- Username: `agentvoicebox`
- Password: `agentvoicebox_secure_2024`

## Development Workflow

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run unit tests
pytest tests/ -v --ignore=tests/integration

# Run integration tests (requires Docker stack)
pytest tests/integration/ -v
```

### Viewing Logs

```bash
# All services
docker compose -p agentvoicebox logs -f

# Specific service
docker compose -p agentvoicebox logs -f gateway

# Worker logs
docker compose -p agentvoicebox logs -f stt-worker tts-worker llm-worker
```

### Rebuilding Services

```bash
# Rebuild all
docker compose -p agentvoicebox up -d --build

# Rebuild specific service
docker compose -p agentvoicebox up -d --build gateway
```

### Database Migrations

```bash
# Run migrations
docker compose -p agentvoicebox exec gateway alembic upgrade head

# Create new migration
docker compose -p agentvoicebox exec gateway alembic revision --autogenerate -m "description"
```

## Testing the API

### Create a Session

```bash
curl -X POST http://localhost:25000/v1/realtime/sessions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "model": "ovos-voice-1",
    "voice": "am_onyx",
    "instructions": "You are a helpful assistant."
  }'
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:25000/v1/realtime?access_token=your_api_key');

ws.onopen = () => {
  console.log('Connected');
  ws.send(JSON.stringify({
    type: 'session.update',
    session: { voice: 'af_bella' }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### List TTS Voices

```bash
curl http://localhost:25000/v1/tts/voices
```

## Troubleshooting

### Services Not Starting

Check Docker resources:
```bash
docker system df
docker system prune -a  # Clean up if needed
```

Ensure 10GB+ RAM is allocated to Docker.

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker compose -p agentvoicebox exec postgres pg_isready

# Connect to database
docker compose -p agentvoicebox exec postgres psql -U agentvoicebox
```

### Redis Connection Issues

```bash
# Check Redis is running
docker compose -p agentvoicebox exec redis redis-cli ping
```

### Worker Issues

```bash
# Check worker logs
docker compose -p agentvoicebox logs stt-worker
docker compose -p agentvoicebox logs tts-worker
docker compose -p agentvoicebox logs llm-worker

# Restart workers
docker compose -p agentvoicebox restart stt-worker tts-worker llm-worker
```

### Port Conflicts

If ports are in use, modify `docker-compose.yml` port mappings or stop conflicting services.

## Stopping the Stack

```bash
# Stop all services
docker compose -p agentvoicebox down

# Stop and remove volumes (WARNING: deletes data)
docker compose -p agentvoicebox down -v
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
│                    (HAProxy - prod)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Gateway Service                          │
│              (FastAPI + WebSocket)                          │
│                   Port: 25000                               │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  STT Worker   │    │  LLM Worker   │    │  TTS Worker   │
│ (Whisper)     │    │ (OpenAI/Groq) │    │ (Kokoro)      │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Redis Cluster                            │
│           (Sessions, Rate Limits, Streams)                  │
│                   Port: 25003                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PostgreSQL                               │
│        (Tenants, Sessions, Conversations)                   │
│                   Port: 25002                               │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. Review the [API Documentation](./API.md)
2. Check the [WebSocket Protocol](./WEBSOCKET_PROTOCOL.md)
3. Explore [Grafana Dashboards](http://localhost:25009)
4. Run the [Test Suite](../tests/README.md)

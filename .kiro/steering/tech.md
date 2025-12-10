# Technology Stack

## Blog (Root Project)
- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript (strict mode)
- **Styling**: Tailwind CSS 3.x with dark mode support
- **Content**: Markdown with gray-matter, remark, rehype-highlight
- **React**: 19.0.0-rc

## AgentVoiceBox Engine (Python Backend)
- **Framework**: Flask with Gunicorn/Gevent
- **Database**: PostgreSQL 16 with SQLAlchemy (async support)
- **Cache/Sessions**: Redis 7
- **Auth**: Keycloak 24, JWT validation
- **Billing**: Lago
- **Policy**: OPA (Open Policy Agent)
- **Linting**: Ruff, Black (line-length: 100)
- **Testing**: pytest

## Portal Frontend (Next.js)
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS with Radix UI primitives
- **State**: TanStack React Query
- **Forms**: react-hook-form + zod validation
- **Charts**: Recharts
- **Testing**: Vitest + Testing Library + fast-check (property tests)

## Workers
- **STT**: Faster-Whisper (tiny model, CPU/int8)
- **TTS**: Kokoro models
- **LLM**: Groq/OpenAI API integration

## Infrastructure
- **Containers**: Docker Compose
- **Monitoring**: Prometheus + Grafana
- **Port Range**: 25000-25099

---

## Common Commands

### Blog (Root)
```bash
npm run dev          # Start dev server with Turbopack
npm run build        # Production build
npm run start        # Start production server
```

### AgentVoiceBox Backend
```bash
make install-dev     # Install dependencies
make format          # Format with Black
make lint            # Lint with Ruff
make check           # Verify formatting + linting
make test            # Run pytest
make docker-up       # Start full stack
make docker-down     # Stop and cleanup
make docker-logs     # Tail container logs
```

### Portal Frontend
```bash
npm run dev          # Start dev server (port 3000)
npm run build        # Production build
npm run lint         # ESLint
npm run type-check   # TypeScript check
npm run test         # Vitest (single run)
npm run test:watch   # Vitest watch mode
npm run test:coverage # Coverage report
```

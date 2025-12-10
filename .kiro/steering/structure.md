# Project Structure

```
/
├── _posts/                          # Blog markdown posts
├── src/                             # Blog Next.js app
│   ├── app/                         # App Router pages
│   │   ├── _components/             # Shared UI components
│   │   ├── posts/[slug]/            # Dynamic post pages
│   │   ├── archive/                 # Post archive
│   │   └── chat/                    # Chat feature
│   ├── interfaces/                  # TypeScript types
│   └── lib/                         # Utilities (api, markdown)
├── public/                          # Static assets
│
├── ovos-voice-agent/                # Voice agent project
│   ├── AgentVoiceBoxEngine/         # Enterprise platform
│   │   ├── app/                     # Flask application
│   │   │   ├── models/              # SQLAlchemy models
│   │   │   ├── routes/              # API endpoints
│   │   │   ├── services/            # Business logic
│   │   │   ├── transports/          # WebSocket handlers
│   │   │   ├── observability/       # Logging, metrics
│   │   │   └── tts/                 # TTS provider
│   │   ├── portal/                  # Portal backend API
│   │   ├── portal-frontend/         # SaaS Portal (Next.js)
│   │   │   ├── src/app/             # App Router pages
│   │   │   │   ├── admin/           # Admin dashboard
│   │   │   │   ├── dashboard/       # Customer dashboard
│   │   │   │   ├── api-keys/        # API key management
│   │   │   │   ├── billing/         # Billing pages
│   │   │   │   └── team/            # Team management
│   │   │   ├── src/components/      # UI components
│   │   │   │   ├── ui/              # Radix-based primitives
│   │   │   │   ├── layout/          # Layout components
│   │   │   │   └── auth/            # Auth components
│   │   │   ├── src/services/        # API clients, utilities
│   │   │   ├── src/contexts/        # React contexts
│   │   │   └── src/__tests__/       # Test suites
│   │   ├── workers/                 # STT/TTS/LLM workers
│   │   ├── tests/                   # Python tests
│   │   ├── migrations/              # Alembic migrations
│   │   ├── keycloak/                # Realm config
│   │   ├── lago/                    # Billing seed data
│   │   ├── observability/           # Prometheus/Grafana
│   │   └── policies/                # OPA Rego policies
│   └── sprint*/                     # Development sprints
│
├── docs/                            # Documentation
└── .kiro/                           # Kiro configuration
    ├── specs/                       # Feature specifications
    └── steering/                    # AI guidance rules
```

## Key Conventions

- **Blog posts**: Add `.md` files to `_posts/` with YAML front matter
- **Portal pages**: Use Next.js App Router conventions in `portal-frontend/src/app/`
- **API services**: Place in `portal-frontend/src/services/`
- **UI components**: Radix primitives in `components/ui/`, layouts in `components/layout/`
- **Python code**: Follow `app/` structure with models, services, routes separation
- **Tests**: Python in `tests/`, Frontend in `src/__tests__/` (unit + property tests)

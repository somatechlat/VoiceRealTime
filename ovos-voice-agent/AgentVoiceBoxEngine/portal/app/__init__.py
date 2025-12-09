"""Customer Portal API Service.

FastAPI-based customer self-service portal providing:
- Dashboard with usage and billing summary
- API key management
- Billing and subscription management
- Team management
- Settings and webhooks

Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8, 21.9
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import api_keys, billing, dashboard, payments, settings, team

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Portal API starting up...")
    
    # Initialize services
    from ..app.services.lago_service import init_lago_service
    from ..app.services.payment_service import init_payment_service
    from ..app.services.keycloak_service import init_keycloak_service
    
    try:
        await init_lago_service()
        await init_payment_service()
        await init_keycloak_service()
        logger.info("Portal services initialized")
    except Exception as e:
        logger.warning(f"Some services failed to initialize: {e}")
    
    yield
    
    # Cleanup
    logger.info("Portal API shutting down...")
    from ..app.services.lago_service import close_lago_service
    from ..app.services.payment_service import close_payment_service
    from ..app.services.keycloak_service import close_keycloak_service
    
    await close_lago_service()
    await close_payment_service()
    await close_keycloak_service()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AgentVoiceBox Customer Portal API",
        description="Self-service portal for managing your AgentVoiceBox account",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware for browser clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:8081",
            "https://portal.agentvoicebox.com",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard"])
    app.include_router(api_keys.router, prefix="/api/v1", tags=["API Keys"])
    app.include_router(billing.router, prefix="/api/v1", tags=["Billing"])
    app.include_router(payments.router, prefix="/api/v1", tags=["Payments"])
    app.include_router(team.router, prefix="/api/v1", tags=["Team"])
    app.include_router(settings.router, prefix="/api/v1", tags=["Settings"])
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "portal-api"}
    
    return app


app = create_app()

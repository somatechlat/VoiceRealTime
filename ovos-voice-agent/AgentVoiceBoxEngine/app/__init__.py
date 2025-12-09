"""Enterprise-grade Flask application factory for the OVOS voice agent."""

from __future__ import annotations

import asyncio
import atexit
import logging
import os

# Third‑party imports – alphabetical order
from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.proxy_fix import ProxyFix

# Local imports – alphabetical order
from .config import AppConfig, configure_app_from_env
from .observability.logging import configure_logging
from .observability.metrics import init_metrics
from .routes.health import health_blueprint
from .routes.realtime import realtime_blueprint
from .routes.tts import tts_blueprint
from .services.redis_client import RedisClient
from .services.distributed_session import DistributedSessionManager
from .services.connection_manager import init_connection_manager, setup_signal_handlers
from .services.redis_streams import init_streams_client
from .transports import register_transports

logger = logging.getLogger(__name__)

# Flask-CORS is optional; fallback to a no‑op implementation if the package is not installed.
try:
    from flask_cors import CORS
except Exception:  # pragma: no cover – package may be absent in minimal environments
    def CORS(app, *_, **__):  # type: ignore[override]
        """Fallback stub for ``flask_cors.CORS`` when the library is unavailable.

        The real CORS extension adds ``Access-Control-Allow-Origin`` headers to
        responses.  For testing or environments where the dependency is omitted,
        we provide a no‑op function that simply returns the app unchanged.
        """
        return app

def _run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _init_redis(app: Flask, cfg: AppConfig) -> None:
    """Initialize Redis client and distributed session manager.
    
    Redis is used for:
    - Distributed session state (cross-gateway access)
    - Rate limiting (already implemented)
    - Pub/sub for real-time events
    
    PostgreSQL remains the source of truth for persistence.
    """
    try:
        redis_client = RedisClient(cfg.redis)
        _run_async(redis_client.connect())
        
        gateway_id = os.getenv("GATEWAY_ID", f"gateway-{os.getpid()}")
        session_manager = DistributedSessionManager(redis_client, gateway_id)
        session_manager.start_cleanup_task()
        
        # Initialize Redis Streams client for worker communication
        streams_client = init_streams_client(redis_client)
        
        app.extensions["redis_client"] = redis_client
        app.extensions["distributed_session_manager"] = session_manager
        app.extensions["redis_streams_client"] = streams_client
        
        logger.info(
            "Redis initialized",
            extra={"gateway_id": gateway_id, "redis_url": cfg.redis.url}
        )
        
        # Register cleanup on app shutdown
        def cleanup_redis():
            try:
                session_manager.stop_cleanup_task()
                _run_async(redis_client.disconnect())
                logger.info("Redis connection closed on shutdown")
            except Exception as e:
                logger.warning(f"Error closing Redis: {e}")
        
        atexit.register(cleanup_redis)
        
    except Exception as e:
        logger.warning(
            f"Redis initialization failed, falling back to PostgreSQL-only mode: {e}"
        )
        app.extensions["redis_client"] = None
        app.extensions["distributed_session_manager"] = None


def create_app(config: AppConfig | None = None) -> Flask:
    """Application factory that wires configuration, logging, metrics, and blueprints."""

    cfg = config or configure_app_from_env()
    configure_logging(cfg)

    app = Flask(__name__)
    app.config.from_mapping(cfg.to_flask_config())
    app.extensions["app_config"] = cfg

    # Initialize Redis for distributed state (graceful fallback if unavailable)
    _init_redis(app, cfg)

    # Initialize connection manager for graceful shutdown (30s drain per design)
    connection_manager = init_connection_manager(drain_timeout_seconds=30)
    app.extensions["connection_manager"] = connection_manager
    
    # Set up SIGTERM/SIGINT handlers for Kubernetes graceful shutdown
    # Skip in testing mode to avoid interfering with test runners
    if cfg.flask_env != "testing":
        setup_signal_handlers()

    # Attach blueprints
    app.register_blueprint(health_blueprint)
    app.register_blueprint(realtime_blueprint, url_prefix="/v1")
    app.register_blueprint(tts_blueprint, url_prefix="/v1")
    register_transports(app)

    # Enable Cross‑Origin Resource Sharing – allow any origin for all endpoints
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Expose Prometheus metrics endpoint at /metrics
    registry, metrics_app = init_metrics(cfg)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # type: ignore[assignment]
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": metrics_app})  # type: ignore[assignment]

    # Store registry to allow other modules to create custom metrics later
    app.extensions["metrics_registry"] = registry

    return app


__all__ = ["create_app", "AppConfig"]

"""Enterprise-grade Flask application factory for the OVOS voice agent."""

from __future__ import annotations

from flask_cors import CORS
from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import AppConfig, configure_app_from_env
from .observability.logging import configure_logging
from .observability.metrics import init_metrics
from .routes.health import health_blueprint
from .routes.realtime import realtime_blueprint
from .transports import register_transports

def create_app(config: AppConfig | None = None) -> Flask:
    """Application factory that wires configuration, logging, metrics, and blueprints."""

    cfg = config or configure_app_from_env()
    configure_logging(cfg)

    app = Flask(__name__)
    app.config.from_mapping(cfg.to_flask_config())
    app.extensions["app_config"] = cfg

    # Attach blueprints
    app.register_blueprint(health_blueprint)
    app.register_blueprint(realtime_blueprint, url_prefix="/v1")
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

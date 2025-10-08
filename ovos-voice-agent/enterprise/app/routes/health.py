"""Health and readiness endpoints."""

from __future__ import annotations

import datetime as dt

from flask import Blueprint, current_app, jsonify

health_blueprint = Blueprint("health", __name__)


@health_blueprint.get("/health")
def healthcheck():
    registry = current_app.extensions.get("metrics_registry")
    return jsonify(
        {
            "status": "healthy",
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            "metrics_registered": bool(registry),
        }
    )

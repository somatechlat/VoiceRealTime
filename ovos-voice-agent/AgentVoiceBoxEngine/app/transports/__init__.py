"""Transport registration for realtime protocols."""

from __future__ import annotations

from flask import Flask

from .realtime_ws import register_realtime_websocket


def register_transports(app: Flask) -> None:
    """Register all realtime transports with the Flask application."""

    register_realtime_websocket(app)


__all__ = ["register_transports"]

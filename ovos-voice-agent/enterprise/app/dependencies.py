"""Application-level dependency helpers for the enterprise gateway."""

from __future__ import annotations

from flask import current_app

from .config import AppConfig, configure_app_from_env
from .services.opa_client import OPAClient
from .services.session_service import SessionService
from .services.token_service import TokenService
from .utils.database import create_session_factory


def get_app_config() -> AppConfig:
    cfg = current_app.extensions.get("app_config")
    if cfg is None:
        cfg = configure_app_from_env()
        current_app.extensions["app_config"] = cfg
    return cfg


def get_session_factory(config: AppConfig):
    session_factory = current_app.extensions.get("session_factory")
    if session_factory is None:
        session_factory = create_session_factory(config)
        current_app.extensions["session_factory"] = session_factory
    return session_factory


def get_session_service(config: AppConfig) -> SessionService:
    service = current_app.extensions.get("session_service")
    if service is None:
        session_factory = get_session_factory(config)
        service = SessionService(session_factory)
        current_app.extensions["session_service"] = service
    return service


def get_opa_client(config: AppConfig) -> OPAClient:
    opa_client = current_app.extensions.get("opa_client")
    if opa_client is None:
        opa_client = OPAClient(config)
        current_app.extensions["opa_client"] = opa_client
    return opa_client


def get_token_service() -> TokenService:
    token_service = current_app.extensions.get("token_service")
    if token_service is None:
        token_service = TokenService()
        current_app.extensions["token_service"] = token_service
    return token_service


__all__ = [
    "get_app_config",
    "get_session_factory",
    "get_session_service",
    "get_opa_client",
    "get_token_service",
]

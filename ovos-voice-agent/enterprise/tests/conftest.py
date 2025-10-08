"""Shared pytest fixtures for the enterprise realtime gateway tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure the enterprise application package is importable when running from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # type: ignore  # noqa: E402
from app.config import (  # noqa: E402
    AppConfig,
    DatabaseSettings,
    KafkaSettings,
    ObservabilitySettings,
    OPASettings,
    RateLimitSettings,
    SecuritySettings,
)
from app.models.base import Base  # noqa: E402
from app.services.session_service import SessionService  # noqa: E402
from app.services.token_service import TokenService  # noqa: E402


class AllowAllOPA:
    """Test helper that unconditionally permits policy requests."""

    def __init__(self) -> None:
        self.requests: list[Dict[str, Any]] = []

    async def allow(self, payload: Dict[str, Any]) -> bool:
        self.requests.append(payload)
        return True


@pytest.fixture()
def app(tmp_path):
    db_uri = f"sqlite+pysqlite:///{tmp_path / 'test.db'}"

    config = AppConfig(
        flask_env="testing",
        secret_key="test-secret-key",
        kafka=KafkaSettings(bootstrap_servers="localhost:9092"),
        database=DatabaseSettings(uri=db_uri, pool_size=5, max_overflow=10, echo=False),
        opa=OPASettings(
            url="http://localhost:8181", decision_path="/v1/data/voice/allow", timeout_seconds=3
        ),
        security=SecuritySettings(
            project_api_keys={"demo-project": "test-token"},
            default_secret_ttl_seconds=120,
            rate_limits=RateLimitSettings(requests_per_minute=100, tokens_per_minute=200000),
        ),
        observability=ObservabilitySettings(service_name="ovos-test-service", log_level="DEBUG"),
    )

    flask_app = create_app(config)
    engine = create_engine(db_uri, future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    # Preload extensions used by the routes so tests rely on deterministic collaborators.
    flask_app.extensions["app_config"] = config
    flask_app.extensions["session_factory"] = session_factory
    flask_app.extensions["session_service"] = SessionService(session_factory)
    flask_app.extensions["token_service"] = TokenService()
    flask_app.extensions["opa_client"] = AllowAllOPA()

    with flask_app.app_context():
        yield flask_app

    engine.dispose()

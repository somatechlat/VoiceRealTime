"""Integration tests for realtime REST endpoints."""

from __future__ import annotations

from app.models.session import SessionModel  # noqa: E402
from app.schemas.realtime import default_session_config  # noqa: E402
from app.services.token_service import TokenService  # noqa: E402


def test_client_secret_endpoint_returns_defaults(app):
    client = app.test_client()
    response = client.post(
        "/v1/realtime/client_secrets",
        headers={
            "Authorization": "Bearer test-token",
            "X-Request-Id": "req_test_client_secret",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload["value"].startswith("ek_")
    assert payload["session"]["model"] == default_session_config().model
    assert response.headers["x-request-id"] == "req_test_client_secret"
    assert response.headers["openai-project"] == "demo-project"

    token_service: TokenService = app.extensions["token_service"]
    stored = token_service.get(payload["value"])
    assert stored is not None
    assert stored.session_id == payload["session"]["id"]


def test_create_session_persists_payload_and_persona(app):
    client = app.test_client()
    secret_response = client.post(
        "/v1/realtime/client_secrets",
        headers={
            "Authorization": "Bearer test-token",
            "X-Request-Id": "req_test_session_secret",
        },
    )
    secret_payload = secret_response.get_json()
    secret_value = secret_payload["value"]

    session_request = {
        "client_secret": secret_value,
        "session": {
            "instructions": "You are a flight concierge.",
            "output_modalities": ["text", "audio"],
            "tools": [{"type": "action", "name": "book_flight"}],
        },
        "persona": {"name": "Ava", "variant": "concierge"},
    }

    response = client.post(
        "/v1/realtime/sessions",
        json=session_request,
        headers={
            "Authorization": "Bearer test-token",
            "X-Request-Id": "req_test_session_create",
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload is not None
    session_data = payload["session"]
    assert session_data["persona"] == session_request["persona"]
    assert session_data["instructions"] == session_request["session"]["instructions"]
    assert set(session_data["output_modalities"]) == {"text", "audio"}

    session_factory = app.extensions["session_factory"]
    with session_factory() as db:
        record = db.get(SessionModel, session_data["id"])
        assert record is not None
        assert record.persona == session_request["persona"]
        assert record.session_config["output_modalities"] == ["text", "audio"]
        assert record.session_config["tools"][0]["name"] == "book_flight"


def test_create_session_with_invalid_secret_returns_401(app):
    client = app.test_client()
    response = client.post(
        "/v1/realtime/sessions",
        json={"client_secret": "ek_invalid_secret"},
        headers={
            "Authorization": "Bearer test-token",
            "X-Request-Id": "req_test_invalid_secret",
        },
    )

    assert response.status_code == 401
    payload = response.get_json()
    assert payload is not None
    assert payload["error"]["code"] == "invalid_client_secret"
    assert payload["error"]["type"] == "authentication_error"

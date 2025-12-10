"""Tests for the realtime WebSocket transport."""

from __future__ import annotations

import json
import uuid

from app.models.session import ConversationItem, SessionModel  # noqa: E402
from simple_websocket import ConnectionClosed


class DummyWebSocket:
    def __init__(self, messages: list[str] | None = None):
        self._messages = list(messages) if messages else []
        self.sent: list[str] = []
        self.closed = False

    def send(self, data: str) -> None:
        self.sent.append(data)

    def receive(self, timeout: float = None):
        if self._messages:
            return self._messages.pop(0)
        raise ConnectionClosed(1000, "closed")

    def close(self, *args, **kwargs) -> None:  # noqa: ANN002
        self.closed = True


def _bootstrap_session(client):
    secret_response = client.post(
        "/v1/realtime/client_secrets",
        headers={
            "Authorization": "Bearer test-token",
            "X-Request-Id": f"req_{uuid.uuid4().hex}",
        },
    )
    secret_payload = secret_response.get_json()
    secret_value = secret_payload["value"]

    session_response = client.post(
        "/v1/realtime/sessions",
        json={"client_secret": secret_value},
        headers={
            "Authorization": "Bearer test-token",
            "X-Request-Id": f"req_{uuid.uuid4().hex}",
        },
    )
    session_payload = session_response.get_json()
    session_id = session_payload["session"]["id"]
    return secret_value, session_id


def _invoke_socket(app, ws, headers=None):
    with app.test_request_context("/v1/realtime", headers=headers or {}):
        handler = app.view_functions["realtime_socket"]
        raw_handler = getattr(handler, "__wrapped__", handler)
        raw_handler(ws)


def test_websocket_rejects_missing_client_secret(app):
    ws = DummyWebSocket()
    _invoke_socket(app, ws)

    assert ws.closed is True
    assert len(ws.sent) == 1
    error_event = json.loads(ws.sent[0])
    assert error_event["type"] == "error"
    assert error_event["error"]["code"] == "missing_client_secret"


def test_websocket_rejects_invalid_client_secret(app):
    ws = DummyWebSocket()
    _invoke_socket(app, ws, headers={"Authorization": "Bearer ek_invalid_secret"})

    assert ws.closed is True
    error_event = json.loads(ws.sent[0])
    assert error_event["type"] == "error"
    assert error_event["error"]["code"] == "invalid_client_secret"


def test_websocket_session_update_persists_changes(app):
    client = app.test_client()
    secret, session_id = _bootstrap_session(client)

    ws = DummyWebSocket(
        [json.dumps({"type": "session.update", "session": {"instructions": "Be precise."}})]
    )
    _invoke_socket(app, ws, headers={"Authorization": f"Bearer {secret}"})

    events = [json.loads(event) for event in ws.sent]
    assert events[0]["type"] == "session.created"
    assert events[1]["type"] == "rate_limits.updated"
    session_updates = [event for event in events if event["type"] == "session.updated"]
    assert session_updates
    assert session_updates[0]["session"]["instructions"] == "Be precise."

    session_factory = app.extensions["session_factory"]
    with session_factory() as db:
        record = db.get(SessionModel, session_id)
        assert record is not None
        assert record.instructions == "Be precise."


def test_websocket_conversation_and_response_flow(app):
    client = app.test_client()
    secret, session_id = _bootstrap_session(client)

    ws = DummyWebSocket(
        [
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Hello there",
                            }
                        ],
                    },
                }
            ),
            json.dumps({"type": "response.create"}),
        ]
    )
    _invoke_socket(app, ws, headers={"Authorization": f"Bearer {secret}"})

    events = [json.loads(event) for event in ws.sent]
    assert events[0]["type"] == "session.created"
    assert events[1]["type"] == "rate_limits.updated"

    conversation_events = [
        event for event in events if event["type"] == "conversation.item.created"
    ]
    assert conversation_events, "Conversation events were not emitted"
    user_item = conversation_events[0]["item"]
    assert user_item["role"] == "user"
    assert user_item["content"][0]["text"] == "Hello there"

    response_types = [event["type"] for event in events if event["type"].startswith("response.")]
    assert "response.created" in response_types
    assert "response.audio.delta" in response_types
    assert "response.done" in response_types

    assistant_items = [
        event for event in conversation_events if event["item"]["role"] == "assistant"
    ]
    assert assistant_items
    assert "I heard you say" in assistant_items[0]["item"]["content"][0].get("text", "")

    session_factory = app.extensions["session_factory"]
    with session_factory() as db:
        items = (
            db.query(ConversationItem)
            .filter(ConversationItem.session_id == session_id)
            .order_by(ConversationItem.created_at.asc())
            .all()
        )
        roles = [item.role for item in items]
        assert "user" in roles
        assert "assistant" in roles

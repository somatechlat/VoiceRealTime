"""WebSocket transport mirroring the OpenAI Realtime API contract."""

from __future__ import annotations

# Standard library imports – sorted alphabetically
import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

# Optional heavy deps used when TTS engines are enabled
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional
    np = None  # type: ignore
try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - optional
    sf = None  # type: ignore

from flask import g, request
from flask_sock import Sock
from simple_websocket import ConnectionClosed

# Local imports – sorted alphabetically for lint compliance
from ..dependencies import (
    get_app_config,
    get_opa_client,
    get_session_service,
    get_token_service,
)
from ..models.session import SessionModel
from ..observability.metrics import policy_denials
from ..schemas.realtime import RealtimeSessionResource
from ..services.session_service import SessionService
from ..services.token_service import ClientSecretRecord
from ..tts.provider import get_provider  # TTS provider abstraction
from ..utils.auth import ensure_request_id

logger = logging.getLogger(__name__)

sock = Sock()


def register_realtime_websocket(app) -> None:
    """Attach the realtime WebSocket endpoint to the Flask application."""

    sock.init_app(app)

    @sock.route("/v1/realtime")
    def realtime_socket(ws):  # pragma: no cover - exercised via tests
        config = get_app_config()
        ensure_request_id(request)
        token_service = get_token_service()
        session_service = get_session_service(config)
        opa_client = get_opa_client(config)

        try:
            # First try the standard Authorization header. If not present, fall back to a
            # query‑string token (e.g. ws://host:20000/v1/realtime?access_token=XYZ).
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                token_qs = request.args.get("access_token")
                if token_qs:
                    auth_header = f"Bearer {token_qs}"
            secret = _extract_bearer(auth_header)
        except MissingBearerTokenError as exc:
            _send_error(ws, "authentication_error", "missing_client_secret", str(exc))
            ws.close()
            return
        token_record = token_service.get(secret)
        if token_record is None:
            _send_error(ws, "authentication_error", "invalid_client_secret", "Client secret invalid or expired")
            ws.close()
            return

        # Ephemeral secrets are one-time use.
        token_service.revoke(secret)

        if token_record.project_id:
            g.project_id = token_record.project_id

        policy_payload = {
            "path": request.path,
            "method": "WEBSOCKET",
            "project_id": token_record.project_id,
            "session_id": token_record.session_id,
            "query_string": request.query_string.decode("utf-8"),
            "headers": {key: value for key, value in request.headers.items()},
        }
        if not _authorize(opa_client, policy_payload):
            _send_error(ws, "policy_error", "policy_denied", "Request denied by policy engine")
            ws.close()
            return

        session_model = session_service.get_session(token_record.session_id)
        if session_model is None:
            session_model = session_service.create_session(
                token_record.session_id,
                token_record.project_id,
                token_record.session_config,
                token_record.expires_at,
                persona=None,
            )

        connection = RealtimeWebsocketConnection(
            ws=ws,
            config=config,
            session=session_model,
            session_service=session_service,
            token_record=token_record,
        )

        try:
            connection.run()
        finally:
            logger.info("Realtime connection closed", extra={"session_id": session_model.id})


def _extract_bearer(header_value: Optional[str]) -> str:
    if not header_value or not header_value.startswith("Bearer "):
        raise MissingBearerTokenError("Missing bearer token")
    token = header_value[7:].strip()
    if not token:
        raise MissingBearerTokenError("Empty bearer token")
    return token


class MissingBearerTokenError(RuntimeError):
    """Raised when the Authorization header is missing or malformed."""


def _authorize(opa_client, payload: Dict[str, Any]) -> bool:
    allowed = opa_client.allow(payload)
    if not allowed:
        policy_denials.inc()
    return allowed


def _send_error(ws, error_type: str, code: str, message: str, param: str | None = None) -> None:
    payload = {
        "type": "error",
        "error": {
            "type": error_type,
            "code": code,
            "message": message,
        },
    }
    if param:
        payload["error"]["param"] = param
    try:
        ws.send(json.dumps(payload))
    except ConnectionClosed:
        logger.debug("Connection already closed while sending error", exc_info=True)


def _normalize_max_output(value: Optional[str]) -> Optional[int | str]:
    if value is None:
        return None
    lowered = value.lower() if isinstance(value, str) else value
    if lowered == "inf":
        return "inf"
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _session_resource(model: SessionModel) -> Dict[str, Any]:
    session_cfg = model.session_config or {}
    resource = RealtimeSessionResource(
        id=model.id,
        status=model.status,
        created_at=int(model.created_at.timestamp() if model.created_at else time.time()),
        expires_at=int(model.expires_at.timestamp()) if model.expires_at else None,
        model=model.model,
        instructions=model.instructions,
        output_modalities=session_cfg.get("output_modalities"),
        tools=session_cfg.get("tools"),
        tool_choice=session_cfg.get("tool_choice"),
        audio=session_cfg.get("audio"),
        max_output_tokens=_normalize_max_output(model.max_output_tokens),
        persona=model.persona or None,
    )
    return resource.model_dump(exclude_none=True)


def _rate_limits_payload(rate_limits) -> Dict[str, Any]:
    return {
        "type": "rate_limits.updated",
        "rate_limits": {
            "requests": {
                "limit": rate_limits.requests_per_minute,
                "remaining": rate_limits.requests_per_minute,
                "reset_seconds": 60,
            },
            "tokens": {
                "limit": rate_limits.tokens_per_minute,
                "remaining": rate_limits.tokens_per_minute,
                "reset_seconds": 60,
            },
        },
    }


class RealtimeWebsocketConnection:
    def __init__(
        self,
        *,
        ws,
        config,
        session: SessionModel,
        session_service: SessionService,
        token_record: ClientSecretRecord,
    ) -> None:
        self._ws = ws
        self._config = config
        self._session_service = session_service
        self._session = session
        self._token = token_record
        self._conversation_order: List[str] = []
        self._is_speaking = False
        self._audio_buffer = bytearray()
        self._last_user_transcript: Optional[str] = None
        self._cancel_current: bool = False

    def run(self) -> None:
        logger.info(
            "Realtime connection established",
            extra={"session_id": self._session.id, "project_id": self._token.project_id},
        )
        self._send_event({"type": "session.created", "session": _session_resource(self._session)})
        self._send_event(_rate_limits_payload(self._config.security.rate_limits))

        while True:
            try:
                message = self._ws.receive()
            except ConnectionClosed:
                break

            if message is None:
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError as exc:
                _send_error(self._ws, "validation_error", "invalid_json", f"Invalid JSON: {exc}")
                continue

            event_type = payload.get("type")
            if not event_type:
                _send_error(self._ws, "validation_error", "missing_type", "Event type is required")
                continue

            handler_name = f"handle_{event_type.replace('.', '_')}"
            handler = getattr(self, handler_name, None)
            if not handler:
                _send_error(
                    self._ws,
                    "validation_error",
                    "unsupported_event_type",
                    f"Unsupported event type: {event_type}",
                )
                continue

            try:
                handler(payload)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to process websocket event", exc_info=exc)
                _send_error(
                    self._ws,
                    "processing_error",
                    "internal_error",
                    "Internal error processing event",
                )

    # Event handlers -----------------------------------------------------------------

    def handle_session_update(self, payload: Dict[str, Any]) -> None:
        updates = payload.get("session", {})
        persona = payload.get("persona")
        updated = self._session_service.update_session(
            self._session.id,
            session_updates=updates,
            persona=persona,
        )
        if updated is None:
            _send_error(
                self._ws,
                "not_found_error",
                "session_not_found",
                "Session no longer exists",
            )
            return

        self._session = updated
        self._send_event({"type": "session.updated", "session": _session_resource(updated)})

    def handle_input_audio_buffer_append(self, payload: Dict[str, Any]) -> None:
        audio_b64 = payload.get("audio")
        if not audio_b64:
            return
        try:
            chunk = base64.b64decode(audio_b64)
        except (ValueError, TypeError):
            _send_error(
                self._ws,
                "validation_error",
                "invalid_audio_chunk",
                "Audio chunk must be base64 encoded",
            )
            return

        self._audio_buffer.extend(chunk)
        if not self._is_speaking:
            self._is_speaking = True
            self._send_event(
                {
                    "type": "input_audio_buffer.speech_started",
                    "audio_start_ms": int(time.time() * 1000),
                    "item_id": _generate_item_id(),
                }
            )

    def handle_input_audio_buffer_commit(self, payload: Dict[str, Any]) -> None:
        audio_item_id = _generate_item_id()
        if self._is_speaking:
            self._send_event(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": int(time.time() * 1000),
                    "item_id": audio_item_id,
                }
            )
        self._is_speaking = False

        transcript = payload.get("transcript")
        if not transcript:
            transcript = f"Captured audio payload ({len(self._audio_buffer)} bytes)."

        content = [
            {
                "type": "input_audio",
                "transcript": transcript,
            }
        ]
        self._emit_conversation_item("user", content)
        self._audio_buffer.clear()

        self._send_event(
            {
                "type": "input_audio_buffer.committed",
                "previous_item_id": None if not self._conversation_order else self._conversation_order[-1],
                "item_id": audio_item_id,
            }
        )
        # Compatibility event name expected by some clients
        self._send_event(
            {
                "type": "input_audio_buffer.commit",
                "previous_item_id": None if not self._conversation_order else self._conversation_order[-1],
                "item_id": audio_item_id,
            }
        )

    def handle_conversation_item_create(self, payload: Dict[str, Any]) -> None:
        item = payload.get("item", {})
        role = item.get("role", "user")
        content = item.get("content") or []
        metadata = {k: v for k, v in item.items() if k not in {"role", "content", "id", "object", "type", "status"}}

        self._emit_conversation_item(role, content, metadata=metadata)

    def handle_response_create(self, payload: Dict[str, Any]) -> None:
        # Reset cancel flag for this response
        self._cancel_current = False
        response_id = _generate_response_id()
        item_id = _generate_item_id()
        self._send_event(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                },
            }
        )

        user_text = self._last_user_transcript or "Hello"
        # Try real LLM response via Groq; fall back to a simple echo if it fails
        assistant_text = None
        try:
            assistant_text = _call_llm(self._session.id, user_text)
        except Exception as exc:
            logger.warning("LLM generation failed; using fallback", exc_info=exc)
        if not assistant_text:
            assistant_text = f"I heard you say: {user_text}. How can I help you?"

        response_item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": assistant_text,
                }
            ],
        }

        self._send_event(
            {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": response_item,
            }
        )

        self._emit_conversation_item("assistant", response_item["content"], metadata={"id": item_id})

        self._send_event(
            {
                "type": "response.audio_transcript.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": assistant_text,
            }
        )

        # -----------------------------------------------------------------
        # TTS – use the new provider abstraction (Kokoro > Piper > Espeak).
        # The provider itself handles engine selection, model loading, and
        # fallback logic.  We only need to stream the base64 chunks to the
        # client and respect the cancel flag.
        # -----------------------------------------------------------------
        # Gather session‑level TTS overrides (voice, speed) – the provider will
        # fall back to environment defaults if these are ``None``.
        sess_cfg = (
            self._session.session_config or {}
        ).get("tts", {}) if hasattr(self._session, "session_config") else {}
        sess_voice = sess_cfg.get("voice") if isinstance(sess_cfg, dict) else None
        sess_speed = sess_cfg.get("speed") if isinstance(sess_cfg, dict) else None

        provider = get_provider()
        audio_streamed = False

        def _run_provider():
            """Execute the async TTS generator synchronously.

            The ``provider.synthesize`` method returns an async generator that
            yields base64‑encoded audio chunks.  Because ``handle_response_create``
            is a regular (non‑async) method, we need to bridge the async call.
            ``_run_async`` creates a temporary event loop and runs the coroutine
            to completion, allowing us to stream the chunks and respect the
            cancel flag.
            """

            async def _inner():
                nonlocal audio_streamed
                async for chunk_b64 in provider.synthesize(
                    assistant_text,
                    voice=sess_voice,
                    speed=sess_speed,
                    cancel_flag=self,
                ):
                    # Send each chunk as a delta event
                    self._send_event(
                        {
                            "type": "response.audio.delta",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": chunk_b64,
                        }
                    )
                    audio_streamed = True

            # Run the async inner coroutine synchronously.
            _run_async(_inner())

        try:
            _run_provider()
        except Exception as exc:  # pragma: no cover – defensive
            logger.warning(
                "TTS provider failed, falling back to silence", exc_info=exc
            )

        # If the provider yielded nothing (e.g., all engines unavailable),
        # send a tiny silence buffer so the client still receives a response.
        if not audio_streamed:
            silence_b64 = base64.b64encode(b"\x00" * 1024).decode("utf-8")
            self._send_event(
                {
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": silence_b64,
                }
            )

    def handle_response_cancel(self, payload: Dict[str, Any]) -> None:
        # Mark cancel and notify client
        self._cancel_current = True
        response_id = payload.get("response_id") or _generate_response_id()
        self._send_event({"type": "response.cancelled", "response_id": response_id})

    # Internal helpers ----------------------------------------------------------------

    def _emit_conversation_item(
        self,
        role: str,
        content: List[Dict[str, Any]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        item_id = metadata.get("id") if metadata and "id" in metadata else _generate_item_id()
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": role,
            "content": content,
        }
        if metadata:
            item.update({k: v for k, v in metadata.items() if k != "id"})

        self._session_service.append_conversation_item(self._session.id, item)

        previous_item_id = self._conversation_order[-1] if self._conversation_order else None
        self._conversation_order.append(item_id)

        transcript_text = _extract_transcript(content)
        if role == "user" and transcript_text:
            self._last_user_transcript = transcript_text

        self._send_event(
            {
                "type": "conversation.item.created",
                "previous_item_id": previous_item_id,
                "item": item,
            }
        )
        return item

    def _send_event(self, payload: Dict[str, Any]) -> None:
        try:
            self._ws.send(json.dumps(payload))
        except ConnectionClosed:
            logger.debug("Connection already closed while sending event", exc_info=True)


def _extract_transcript(content: List[Dict[str, Any]]) -> Optional[str]:
    for part in content:
        if part.get("type") in {"input_text", "output_text", "input_audio"}:
            text = part.get("text") or part.get("transcript")
            if text:
                return text
    return None


def _generate_item_id() -> str:
    return f"item_{uuid.uuid4().hex[:16]}"


def _generate_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:16]}"


def _run_async(coro):
    """Run an async coroutine from sync context with a fresh event loop if needed."""
    try:
        import asyncio
        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to execute async coroutine", exc_info=exc)
        return None


def _call_llm(session_id: str, user_text: str) -> Optional[str]:
    """Call the project LLM integration dynamically without brittle relative imports."""
    try:
        # Lazy import to avoid package path issues when running inside Docker/tests
        import importlib
        import os
        import sys
        # Add repo root to path: this file is enterprise/app/transports/realtime_ws.py
        here = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(here, "../../..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        llm_mod = importlib.import_module("llm_integration")
        generate = getattr(llm_mod, "generate_ai_response", None)
        if not generate:
            return None
        return _run_async(generate(session_id, user_text))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LLM import/call failed", exc_info=exc)
        return None


__all__ = ["register_realtime_websocket", "sock"]

"""Realtime API blueprints aligned with the OpenAI REST contract."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict

from flask import Blueprint, jsonify, request
from pydantic import ValidationError
from werkzeug.exceptions import Unauthorized

from ..config import AppConfig
from ..dependencies import (
    get_app_config,
    get_opa_client,
    get_session_service,
    get_token_service,
)
from ..observability.metrics import policy_denials
from ..schemas.realtime import (
    ClientSecretRequest,
    ClientSecretResponse,
    ErrorEnvelope,
    ErrorResponse,
    RealtimeSessionConfig,
    RealtimeSessionRequest,
    RealtimeSessionResource,
    RealtimeSessionResponse,
    default_session_config,
)
from ..services.opa_client import OPAClient
from ..utils.auth import apply_standard_headers, authenticate_request, ensure_request_id

realtime_blueprint = Blueprint("realtime", __name__)


def _authorize(opa: OPAClient, payload: Dict[str, Any]) -> bool:
    """Synchronously check OPA policy decision.

    Returns ``True`` if allowed, otherwise records a denial metric and returns ``False``.
    """
    allowed = opa.allow(payload)
    if not allowed:
        policy_denials.inc()
    return allowed


def _to_epoch(dt_value: dt.datetime | None) -> int | None:
    if dt_value is None:
        return None
    return int(dt_value.timestamp())


def _normalize_max_output(value: str | None) -> int | str | None:
    if value is None:
        return None
    lowered = value.lower()
    if lowered == "inf":
        return "inf"
    try:
        return int(value)
    except ValueError:
        return value


def _make_error_response(
    config: AppConfig,
    status_code: int,
    *,
    error_type: str,
    code: str,
    message: str,
    param: str | None = None,
):
    envelope = ErrorResponse(
        error=ErrorEnvelope(
            type=error_type,
            code=code,
            message=message,
            param=param,
            request_id=ensure_request_id(request),
        )
    ).model_dump()
    response = jsonify(envelope)
    response.status_code = status_code
    apply_standard_headers(response, config)
    return response


def _make_success_response(config: AppConfig, payload: Dict[str, Any], status_code: int = 200):
    ensure_request_id(request)
    response = jsonify(payload)
    response.status_code = status_code
    apply_standard_headers(response, config)
    return response


@realtime_blueprint.post("/realtime/client_secrets")
def create_client_secret():
    config = get_app_config()
    ensure_request_id(request)
    try:
        project_id = authenticate_request(request, config)
    except Unauthorized as exc:
        return _make_error_response(
            config,
            401,
            error_type="authentication_error",
            code="invalid_api_key",
            message=str(exc.description),
        )

    opa_client = get_opa_client(config)
    raw_body = request.get_json(silent=True) or {}

    try:
        parsed_body = ClientSecretRequest(**raw_body)
    except ValidationError as exc:
        return _make_error_response(
            config,
            422,
            error_type="validation_error",
            code="invalid_request_error",
            message=exc.json(),
        )

    policy_payload = {
        "path": request.path,
        "method": request.method,
        "project_id": project_id,
        "metadata": raw_body,
    }
    # Synchronously check policy
    allowed = _authorize(opa_client, policy_payload)
    if not allowed:
        return _make_error_response(
            config,
            403,
            error_type="policy_error",
            code="policy_denied",
            message="Request denied by policy engine",
        )

    session_cfg = parsed_body.session or default_session_config()
    ttl_seconds = (
        parsed_body.expires_after.seconds
        if parsed_body.expires_after
        else config.security.default_secret_ttl_seconds
    )

    token_service = get_token_service()
    session_id = session_cfg.model_dump().get("id") or f"sess_{uuid.uuid4().hex}"
    session_payload = session_cfg.model_dump(exclude_none=True)
    record = token_service.issue(project_id, session_id, session_payload, ttl_seconds)

    session_resource = RealtimeSessionResource(
        id=record.session_id,
        status="active",
        created_at=int(dt.datetime.utcnow().timestamp()),
        expires_at=_to_epoch(record.expires_at),
        model=session_cfg.model,
        instructions=session_cfg.instructions,
        output_modalities=session_cfg.output_modalities,
        tools=session_cfg.tools,
        tool_choice=session_cfg.tool_choice,
        audio=session_payload.get("audio"),
        max_output_tokens=session_cfg.max_output_tokens,
    )

    response_body = ClientSecretResponse(
        value=record.secret,
        expires_at=_to_epoch(record.expires_at),
        session=session_resource.model_dump(exclude_none=True),
    ).model_dump()

    return _make_success_response(config, response_body, status_code=200)


@realtime_blueprint.post("/realtime/sessions")
def create_realtime_session():
    config = get_app_config()
    ensure_request_id(request)
    try:
        project_id = authenticate_request(request, config)
    except Unauthorized as exc:
        return _make_error_response(
            config,
            401,
            error_type="authentication_error",
            code="invalid_api_key",
            message=str(exc.description),
        )

    raw_body = request.get_json(silent=True) or {}
    try:
        parsed_body = RealtimeSessionRequest(**raw_body)
    except ValidationError as exc:
        return _make_error_response(
            config,
            422,
            error_type="validation_error",
            code="invalid_request_error",
            message=exc.json(),
        )

    opa_client = get_opa_client(config)
    policy_payload = {
        "path": request.path,
        "method": request.method,
        "project_id": project_id,
        "metadata": raw_body,
    }
    # Synchronously check policy
    allowed = _authorize(opa_client, policy_payload)
    if not allowed:
        return _make_error_response(
            config,
            403,
            error_type="policy_error",
            code="policy_denied",
            message="Request denied by policy engine",
        )

    token_service = get_token_service()
    record = token_service.get(parsed_body.client_secret)
    if record is None:
        return _make_error_response(
            config,
            401,
            error_type="authentication_error",
            code="invalid_client_secret",
            message="Client secret invalid or expired",
        )

    base_config = RealtimeSessionConfig(**record.session_config)
    if parsed_body.session:
        merged_payload = base_config.model_dump(exclude_none=True)
        merged_payload.update(parsed_body.session.model_dump(exclude_none=True))
        session_cfg = RealtimeSessionConfig(**merged_payload)
    else:
        session_cfg = base_config

    persona_payload = parsed_body.persona or {}
    stored_payload = session_cfg.model_dump(exclude_none=True)

    service = get_session_service(config)
    model = service.create_session(
        record.session_id,
        project_id,
        stored_payload,
        record.expires_at,
        persona=persona_payload,
    )

    session_resource = RealtimeSessionResource(
        id=model.id,
        status=model.status,
        created_at=_to_epoch(model.created_at) or int(dt.datetime.utcnow().timestamp()),
        expires_at=_to_epoch(model.expires_at),
        model=model.model,
        instructions=model.instructions,
        output_modalities=model.session_config.get("output_modalities"),
        tools=model.session_config.get("tools"),
        tool_choice=model.session_config.get("tool_choice"),
        audio=model.session_config.get("audio"),
        max_output_tokens=_normalize_max_output(model.max_output_tokens),
        persona=model.persona,
    )

    response_body = RealtimeSessionResponse(session=session_resource).model_dump(exclude_none=True)
    return _make_success_response(config, response_body, status_code=201)


# NOTE: WebSocket/WebRTC endpoints will be added in a dedicated module using Flask-Sock or Hypercorn.

"""Authentication and response header utilities for the VoiceEngine API."""

from __future__ import annotations

import hmac
import uuid
from typing import Optional

from flask import Request, g
from werkzeug.exceptions import Unauthorized

from ..config import AppConfig


def _extract_token(auth_header: Optional[str]) -> str:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise Unauthorized(description="Missing bearer token")
    token = auth_header[7:].strip()
    if not token:
        raise Unauthorized(description="Empty bearer token")
    return token


def authenticate_request(request: Request, config: AppConfig) -> Optional[str]:
    """Validate the incoming bearer token against configured project keys."""

    token = _extract_token(request.headers.get("Authorization"))
    project_id = None
    for candidate_project, candidate_token in config.security.project_api_keys.items():
        if hmac.compare_digest(candidate_token, token):
            project_id = candidate_project
            break
    if project_id is None:
        raise Unauthorized(description="Invalid bearer token")

    g.project_id = project_id
    g.bearer_token = token
    return project_id


def ensure_request_id(request: Request) -> str:
    request_id = request.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex}"
    g.request_id = request_id
    return request_id


def apply_standard_headers(response, config: AppConfig) -> None:
    """Attach OpenAI-style response headers for observability and rate limits."""

    request_id = getattr(g, "request_id", None)
    if not request_id:
        request_id = f"req_{uuid.uuid4().hex}"
        g.request_id = request_id
    response.headers["x-request-id"] = request_id

    limits = config.security.rate_limits
    response.headers.setdefault("x-ratelimit-limit-requests", str(limits.requests_per_minute))
    response.headers.setdefault("x-ratelimit-remaining-requests", str(limits.requests_per_minute))
    response.headers.setdefault("x-ratelimit-reset-requests", "60")
    response.headers.setdefault("x-ratelimit-limit-tokens", str(limits.tokens_per_minute))
    response.headers.setdefault("x-ratelimit-remaining-tokens", str(limits.tokens_per_minute))
    response.headers.setdefault("x-ratelimit-reset-tokens", "60")

    project_id = getattr(g, "project_id", None)
    if project_id:
        response.headers.setdefault("openai-project", project_id)


__all__ = ["authenticate_request", "ensure_request_id", "apply_standard_headers"]

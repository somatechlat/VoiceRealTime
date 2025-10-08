"""Pydantic schemas mirroring the OpenAI Realtime REST contract."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class ExpiresAfter(BaseModel):
    anchor: Literal["created_at", "last_active_at"] = Field(default="created_at")
    seconds: int = Field(..., gt=0)


class AudioFormat(BaseModel):
    type: str
    rate: Optional[int] = None


class AudioIOConfig(BaseModel):
    format: Optional[AudioFormat] = None
    transcription: Optional[Dict[str, Any]] = None
    noise_reduction: Optional[Dict[str, Any]] = None
    turn_detection: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class AudioConfig(BaseModel):
    input: Optional[AudioIOConfig] = None
    output: Optional[AudioIOConfig] = None

    model_config = ConfigDict(extra="allow")


class RealtimeSessionConfig(BaseModel):
    type: Literal["realtime", "transcription"] = Field(default="realtime")
    model: Optional[str] = None
    instructions: Optional[str] = None
    output_modalities: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    max_output_tokens: Optional[Union[int, Literal["inf"]]] = Field(default="inf")
    tracing: Optional[Dict[str, Any]] = None
    truncation: Optional[Union[str, Dict[str, Any]]] = None
    prompt: Optional[Dict[str, Any]] = None
    audio: Optional[AudioConfig] = None
    include: Optional[List[str]] = None

    model_config = ConfigDict(extra="allow")


class ClientSecretRequest(BaseModel):
    expires_after: Optional[ExpiresAfter] = None
    session: Optional[RealtimeSessionConfig] = None

    model_config = ConfigDict(extra="forbid")


class ClientSecretResponse(BaseModel):
    value: str
    expires_at: int
    session: Dict[str, Any]


class RealtimeSessionRequest(BaseModel):
    client_secret: str
    session: Optional[RealtimeSessionConfig] = None
    persona: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")


class ErrorEnvelope(BaseModel):
    type: str
    code: str
    message: str
    param: Optional[str] = None
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorEnvelope


class RealtimeSessionResource(BaseModel):
    object: Literal["realtime.session"] = Field(default="realtime.session")
    id: str
    status: Literal["active", "closed"] = Field(default="active")
    created_at: int
    expires_at: Optional[int] = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    output_modalities: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    audio: Optional[Dict[str, Any]] = None
    max_output_tokens: Optional[Union[int, Literal["inf"]]] = None
    persona: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class RealtimeSessionResponse(BaseModel):
    session: RealtimeSessionResource


def default_session_config() -> RealtimeSessionConfig:
    """Generate a baseline session configuration aligned with OpenAI defaults."""

    return RealtimeSessionConfig(
        type="realtime",
        model="gpt-realtime-2025-08-28",
        instructions="You are a helpful assistant.",
        output_modalities=["audio"],
        audio=AudioConfig(
            input=AudioIOConfig(
                format=AudioFormat(type="audio/pcm", rate=24000),
                transcription={"model": "whisper-1"},
                turn_detection={
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200,
                },
            ),
            output=AudioIOConfig(
                format=AudioFormat(type="audio/pcm", rate=24000),
                noise_reduction=None,
            ),
        ),
    )


def to_epoch_seconds(dt_value: Optional[float | int]) -> Optional[int]:
    if dt_value is None:
        return None
    return int(dt_value)


def utc_now_epoch() -> int:
    return int(time.time())

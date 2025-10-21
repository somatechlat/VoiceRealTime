"""API compatibility tests for sprint3 FastAPI server."""

import importlib.util
import io
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_TEST_ROOT = Path(__file__).resolve()
_MODULE_PATH = _TEST_ROOT.parent.parent / "main.py"


class _StubTtsEngine:
    def __init__(self) -> None:
        self.voice = "am_onyx"
        self.speed = 1.0

    def get_available_voices(self):  # pragma: no cover - diagnostic helper
        return [self.voice]


class _StubSttEngine:
    def __init__(self) -> None:
        self.calls = []

    async def transcribe(self, audio: bytes, language: str | None = None):
        self.calls.append((audio, language))
        return {"text": "stub transcript", "language": language or "en", "confidence": 0.9}


class _StubSpeechPipeline:
    def __init__(self, session_id: str | None = None, **_kwargs) -> None:
        self.session_id = session_id
        self.voice_set: str | None = None
        self.language_set: str | None = None
        self.reset_invoked = False
        self.tts_engine = _StubTtsEngine()
        self.stt_engine = _StubSttEngine()

    def set_voice(self, voice: str):
        self.voice_set = voice
        self.tts_engine.voice = voice

    def set_language(self, language: str):  # pragma: no cover - simple setter
        self.language_set = language

    def reset_session(self):
        self.reset_invoked = True

    async def synthesize_response(self, *_, **__):
        return {"audio": b"stub-bytes"}


_created_pipelines: list[_StubSpeechPipeline] = []


def _create_realtime_pipeline(session_id: str | None = None, **kwargs):
    pipeline = _StubSpeechPipeline(session_id=session_id, **kwargs)
    _created_pipelines.append(pipeline)
    return pipeline


_speech_pipeline_stub = types.ModuleType("speech_pipeline")
_speech_pipeline_stub.SpeechPipeline = _StubSpeechPipeline
_speech_pipeline_stub.create_realtime_pipeline = _create_realtime_pipeline
_speech_pipeline_stub.TTSEngine = _StubTtsEngine
sys.modules.setdefault("speech_pipeline", _speech_pipeline_stub)


_spec = importlib.util.spec_from_file_location("sprint3_api_main", _MODULE_PATH)
api = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader  # Narrow type for mypy-like tools
_spec.loader.exec_module(api)  # type: ignore[attr-defined]
sys.modules.setdefault("sprint3_api.main", api)

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def _reset_state():
    api.active_sessions.clear()
    api.speech_pipelines.clear()
    _created_pipelines.clear()
    yield
    api.active_sessions.clear()
    api.speech_pipelines.clear()
    _created_pipelines.clear()


def _create_session(client_kwargs: dict | None = None):
    payload = {
        "model": "ovos-voice-1",
        "voice": "demo-voice",
        "language": "en-US",
        "turn_detection": True,
    }
    if client_kwargs:
        payload.update(client_kwargs)
    response = client.post("/v1/realtime/sessions", json=payload)
    response.raise_for_status()
    return response.json()


def test_create_realtime_session_stores_pipeline_state():
    data = _create_session()

    assert data["id"].startswith("sess_")
    session_id = data["id"]
    assert session_id in api.active_sessions
    pipeline = api.speech_pipelines[session_id]
    assert pipeline.voice_set == "demo-voice"
    assert pipeline.session_id == session_id


def test_delete_realtime_session_resets_pipeline():
    data = _create_session()
    session_id = data["id"]
    pipeline = api.speech_pipelines[session_id]

    response = client.delete(f"/v1/realtime/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert session_id not in api.active_sessions
    assert pipeline.reset_invoked is True


def test_create_speech_returns_audio_payload():
    response = client.post(
        "/v1/audio/speech",
        json={"model": "ovos-tts-1", "input": "Hi there", "voice": "demo-voice", "response_format": "mp3"},
    )
    assert response.status_code == 200
    assert response.content == b"stub-bytes"
    assert response.headers["content-type"] == "audio/mpeg"


def test_create_transcription_handles_upload(monkeypatch):
    stub_pipeline = _StubSpeechPipeline()

    async def _transcribe(audio: bytes, language: str | None = None):
        return {"text": "hello", "language": language or "en"}

    stub_pipeline.stt_engine.transcribe = _transcribe

    def _factory(*_args, **_kwargs):
        return stub_pipeline

    monkeypatch.setitem(sys.modules, "speech_pipeline", types.SimpleNamespace(create_realtime_pipeline=_factory))
    monkeypatch.setattr(api, "create_realtime_pipeline", _factory, raising=False)

    file_bytes = io.BytesIO(b"fake audio contents")

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", file_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "hello"
    assert data["language"] == "en"
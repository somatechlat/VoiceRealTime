"""Unit tests for realtime TTS streaming integration."""

import base64
import importlib
import sys
import types
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve()
sys.path.append(str(_TEST_DIR.parents[1]))
sys.path.append(str(_TEST_DIR.parents[2]))

_speech_pipeline_stub = types.ModuleType("speech_pipeline")


class _StubSpeechPipeline:
    def __init__(self) -> None:
        self.tts_engine = None


class _StubTtsEngine:
    pass


async def _stub_create_realtime_pipeline(*_, **__):  # pragma: no cover - only for import time
    return _StubSpeechPipeline()


_speech_pipeline_stub.SpeechPipeline = _StubSpeechPipeline
_speech_pipeline_stub.TTSEngine = _StubTtsEngine
_speech_pipeline_stub.create_realtime_pipeline = _stub_create_realtime_pipeline

sys.modules.setdefault("speech_pipeline", _speech_pipeline_stub)

_realtime_server = importlib.import_module("realtime_server")
StreamingState = _realtime_server.StreamingState
connection_manager = _realtime_server.connection_manager
event_processor = _realtime_server.event_processor


class _FakeTtsEngine:
    def __init__(self) -> None:
        self.voice = "am_onyx"
        self.speed = 1.0

    def get_available_voices(self) -> list[str]:
        return [self.voice]


class _FakePipeline:
    def __init__(self, audio_bytes: bytes | None = None) -> None:
        self.tts_engine = _FakeTtsEngine()
        self.current_voice = self.tts_engine.voice
        self.current_speed = self.tts_engine.speed
        self._audio_bytes = audio_bytes or b"fallback-bytes"

    async def synthesize_response(self, *_, **__) -> dict:
        return {"audio": self._audio_bytes}


class _FakeProvider:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def synthesize(self, *_, **__):
        for chunk in self._chunks:
            yield chunk


@pytest.fixture(autouse=True)
def _reset_manager(monkeypatch):
    original_send_event = connection_manager.send_event
    for mapping in (
        connection_manager.active_connections,
        connection_manager.session_data,
        connection_manager.speech_pipelines,
        connection_manager.conversation_states,
        connection_manager.tts_providers,
        connection_manager.tts_stream_states,
    ):
        mapping.clear()

    yield

    for mapping in (
        connection_manager.active_connections,
        connection_manager.session_data,
        connection_manager.speech_pipelines,
        connection_manager.conversation_states,
        connection_manager.tts_providers,
        connection_manager.tts_stream_states,
    ):
        mapping.clear()
    monkeypatch.setattr(connection_manager, "send_event", original_send_event, raising=False)


@pytest.mark.asyncio
async def test_generate_and_stream_audio_uses_provider_chunks(monkeypatch):
    session_id = "sess_test_stream"
    response_id = "resp_test"
    item_id = "item_test"
    text = "Hello there"

    connection_manager.session_data[session_id] = {
        "tts": {"voice": "am_onyx", "speed": 1.0},
        "voice": "am_onyx",
    }
    connection_manager.conversation_states[session_id] = {}
    connection_manager.speech_pipelines[session_id] = _FakePipeline()

    stream_state = StreamingState()
    connection_manager.tts_stream_states[session_id] = stream_state

    chunks = [
        base64.b64encode(b"chunk-1").decode("utf-8"),
        base64.b64encode(b"chunk-2").decode("utf-8"),
    ]
    connection_manager.tts_providers[session_id] = _FakeProvider(chunks)

    events: list[dict] = []

    async def _capture_event(_, event):
        events.append(event)

    monkeypatch.setattr(connection_manager, "send_event", _capture_event, raising=False)

    await event_processor.generate_and_stream_audio(session_id, response_id, item_id, text)

    audio_events = [e for e in events if e["type"] == "response.audio.delta"]
    assert [e["delta"] for e in audio_events] == chunks

    transcript_events = [e for e in events if e["type"] == "response.audio_transcript.delta"]
    assert len(transcript_events) == 1
    assert transcript_events[0]["delta"] == text

    assert events[-1]["type"] == "response.done"
    assert not stream_state._cancel_current
    assert connection_manager.conversation_states[session_id]["response"] is None


@pytest.mark.asyncio
async def test_generate_and_stream_audio_falls_back_when_provider_empty(monkeypatch):
    session_id = "sess_test_fallback"
    response_id = "resp_fallback"
    item_id = "item_fallback"
    text = "Fallback please"

    fallback_audio = b"fallback audio"
    pipeline = _FakePipeline(audio_bytes=fallback_audio)

    connection_manager.session_data[session_id] = {
        "tts": {"voice": "am_onyx", "speed": 1.0},
        "voice": "am_onyx",
    }
    connection_manager.conversation_states[session_id] = {}
    connection_manager.speech_pipelines[session_id] = pipeline

    stream_state = StreamingState()
    connection_manager.tts_stream_states[session_id] = stream_state
    connection_manager.tts_providers[session_id] = _FakeProvider([])

    events: list[dict] = []

    async def _capture_event(_, event):
        events.append(event)

    monkeypatch.setattr(connection_manager, "send_event", _capture_event, raising=False)

    await event_processor.generate_and_stream_audio(session_id, response_id, item_id, text)

    audio_events = [e for e in events if e["type"] == "response.audio.delta"]
    assert len(audio_events) == 1
    decoded = base64.b64decode(audio_events[0]["delta"])
    assert decoded == fallback_audio

    transcript_events = [e for e in events if e["type"] == "response.audio_transcript.delta"]
    assert transcript_events and transcript_events[0]["delta"] == text

    assert events[-1]["type"] == "response.done"
    assert not stream_state._cancel_current
    assert connection_manager.conversation_states[session_id]["response"] is None


@pytest.mark.asyncio
async def test_handle_response_cancel_sets_flag_and_emits_event(monkeypatch):
    session_id = "sess_cancel"
    response_id = "resp_cancel"

    stream_state = StreamingState()
    connection_manager.tts_stream_states[session_id] = stream_state
    connection_manager.conversation_states[session_id] = {
        "response": {"id": response_id, "object": "realtime.response"}
    }

    events: list[dict] = []

    async def _capture_event(_, event):
        events.append(event)

    monkeypatch.setattr(connection_manager, "send_event", _capture_event, raising=False)

    await event_processor.handle_response_cancel(session_id, {"response_id": response_id})

    assert stream_state._cancel_current is True
    assert connection_manager.conversation_states[session_id]["response"] is None
    assert any(e["type"] == "response.cancelled" for e in events)

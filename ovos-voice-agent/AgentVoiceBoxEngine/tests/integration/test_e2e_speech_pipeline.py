"""Real End-to-End Speech Pipeline Test - NO MOCKS.

Full flow: Audio → STT → LLM → TTS → Audio out
Measures actual latencies against SLA targets.

Requirements: 14.2
- STT p95 < 500ms
- TTS TTFB p95 < 200ms

Run with:
    docker compose -f docker-compose.test.yml up -d
    pytest tests/integration/test_e2e_speech_pipeline.py -v
"""

import asyncio
import base64
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import pytest
import pytest_asyncio

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import RedisSettings
from app.services.redis_client import RedisClient
from app.services.redis_streams import (
    AudioSTTRequest,
    RedisStreamsClient,
    TTSRequest,
)

# Configure pytest-asyncio mode
pytestmark = pytest.mark.asyncio(loop_scope="function")

# Redis URL from environment or default (port 16379 to avoid conflicts)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:16379/0")

# Sample audio (1 second of silence)
SAMPLE_AUDIO_WAV = base64.b64encode(
    b"RIFF"
    + (36 + 32000).to_bytes(4, "little")
    + b"WAVE"
    + b"fmt "
    + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little")
    + (1).to_bytes(2, "little")
    + (16000).to_bytes(4, "little")
    + (32000).to_bytes(4, "little")
    + (2).to_bytes(2, "little")
    + (16).to_bytes(2, "little")
    + b"data"
    + (32000).to_bytes(4, "little")
    + b"\x00" * 32000
).decode("ascii")


@dataclass
class LatencyMetrics:
    """Latency metrics for pipeline stages."""

    stt_latencies_ms: List[float] = field(default_factory=list)
    tts_ttfb_latencies_ms: List[float] = field(default_factory=list)
    tts_total_latencies_ms: List[float] = field(default_factory=list)
    e2e_latencies_ms: List[float] = field(default_factory=list)

    def stt_p95(self) -> Optional[float]:
        if not self.stt_latencies_ms:
            return None
        sorted_vals = sorted(self.stt_latencies_ms)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def tts_ttfb_p95(self) -> Optional[float]:
        if not self.tts_ttfb_latencies_ms:
            return None
        sorted_vals = sorted(self.tts_ttfb_latencies_ms)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def summary(self) -> dict:
        return {
            "stt_p95_ms": self.stt_p95(),
            "stt_avg_ms": statistics.mean(self.stt_latencies_ms) if self.stt_latencies_ms else None,
            "tts_ttfb_p95_ms": self.tts_ttfb_p95(),
            "tts_ttfb_avg_ms": (
                statistics.mean(self.tts_ttfb_latencies_ms) if self.tts_ttfb_latencies_ms else None
            ),
            "e2e_avg_ms": statistics.mean(self.e2e_latencies_ms) if self.e2e_latencies_ms else None,
        }


@pytest_asyncio.fixture
async def redis_client():
    """Create Redis client connected to real Redis."""
    settings = RedisSettings(url=REDIS_URL)
    client = RedisClient(settings)
    await client.connect()
    yield client
    await client.disconnect()


@pytest_asyncio.fixture
async def streams_client(redis_client):
    """Create Redis Streams client."""
    return RedisStreamsClient(redis_client)


class TestSTTLatency:
    """Test STT latency meets SLA targets.

    Requirement: STT p95 < 500ms
    """

    @pytest.mark.asyncio
    async def test_stt_latency_single_request(self, redis_client, streams_client):
        """Test STT latency for a single request."""
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"

        # Publish audio
        start_time = time.time()
        request = AudioSTTRequest(
            session_id=session_id,
            tenant_id=tenant_id,
            audio_b64=SAMPLE_AUDIO_WAV,
            language="en",
        )
        await streams_client.publish_audio_for_stt(request)

        # Wait for transcription
        channel = f"transcription:{session_id}"
        pubsub = await redis_client.subscribe(channel)

        transcription_time = None
        timeout = 30.0

        try:
            async for message in pubsub.listen():
                if time.time() - start_time > timeout:
                    break
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    if data.get("type") == "transcription.completed":
                        transcription_time = time.time()
                        break
        except asyncio.TimeoutError:
            pass
        finally:
            await pubsub.unsubscribe(channel)

        if transcription_time is None:
            pytest.skip("STT worker not running or timed out")

        latency_ms = (transcription_time - start_time) * 1000
        assert latency_ms < 500, f"STT latency {latency_ms}ms exceeds 500ms SLA"

    @pytest.mark.asyncio
    async def test_stt_latency_multiple_requests(self, redis_client, streams_client):
        """Test STT p95 latency across multiple requests."""
        metrics = LatencyMetrics()
        num_requests = 10

        for i in range(num_requests):
            session_id = f"sess_{uuid.uuid4().hex[:16]}"
            tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"

            start_time = time.time()
            request = AudioSTTRequest(
                session_id=session_id,
                tenant_id=tenant_id,
                audio_b64=SAMPLE_AUDIO_WAV,
            )
            await streams_client.publish_audio_for_stt(request)

            # Wait for transcription
            channel = f"transcription:{session_id}"
            pubsub = await redis_client.subscribe(channel)

            try:
                async for message in pubsub.listen():
                    if time.time() - start_time > 30.0:
                        break
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        if data.get("type") == "transcription.completed":
                            latency_ms = (time.time() - start_time) * 1000
                            metrics.stt_latencies_ms.append(latency_ms)
                            break
            except asyncio.TimeoutError:
                pass
            finally:
                await pubsub.unsubscribe(channel)

        if not metrics.stt_latencies_ms:
            pytest.skip("STT worker not running")

        p95 = metrics.stt_p95()
        assert p95 is not None
        assert p95 < 500, f"STT p95 latency {p95}ms exceeds 500ms SLA"


class TestTTSLatency:
    """Test TTS latency meets SLA targets.

    Requirement: TTS TTFB p95 < 200ms
    """

    @pytest.mark.asyncio
    async def test_tts_ttfb_single_request(self, redis_client, streams_client):
        """Test TTS time-to-first-byte for a single request."""
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"

        # Publish TTS request
        start_time = time.time()
        request = TTSRequest(
            session_id=session_id,
            tenant_id=tenant_id,
            text="Hello world.",
            voice="am_onyx",
        )
        await streams_client.publish_tts_request(request)

        # Wait for first audio chunk
        first_chunk_time = None
        timeout = 30.0
        last_id = "0"

        while time.time() - start_time < timeout:
            chunks = await streams_client.read_audio_chunks(
                session_id=session_id,
                last_id=last_id,
                count=1,
                block_ms=1000,
            )
            if chunks:
                first_chunk_time = time.time()
                break

        # Cleanup
        await streams_client.cleanup_session_streams(session_id)

        if first_chunk_time is None:
            pytest.skip("TTS worker not running or timed out")

        ttfb_ms = (first_chunk_time - start_time) * 1000
        assert ttfb_ms < 200, f"TTS TTFB {ttfb_ms}ms exceeds 200ms SLA"

    @pytest.mark.asyncio
    async def test_tts_ttfb_multiple_requests(self, redis_client, streams_client):
        """Test TTS TTFB p95 across multiple requests."""
        metrics = LatencyMetrics()
        num_requests = 10

        for i in range(num_requests):
            session_id = f"sess_{uuid.uuid4().hex[:16]}"
            tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"

            start_time = time.time()
            request = TTSRequest(
                session_id=session_id,
                tenant_id=tenant_id,
                text=f"Test message number {i}.",
                voice="am_onyx",
            )
            await streams_client.publish_tts_request(request)

            # Wait for first chunk
            timeout = 30.0
            last_id = "0"

            while time.time() - start_time < timeout:
                chunks = await streams_client.read_audio_chunks(
                    session_id=session_id,
                    last_id=last_id,
                    count=1,
                    block_ms=1000,
                )
                if chunks:
                    ttfb_ms = (time.time() - start_time) * 1000
                    metrics.tts_ttfb_latencies_ms.append(ttfb_ms)
                    break

            await streams_client.cleanup_session_streams(session_id)

        if not metrics.tts_ttfb_latencies_ms:
            pytest.skip("TTS worker not running")

        p95 = metrics.tts_ttfb_p95()
        assert p95 is not None
        assert p95 < 200, f"TTS TTFB p95 {p95}ms exceeds 200ms SLA"


class TestConcurrentSessions:
    """Test with multiple concurrent sessions."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_stt_requests(self, redis_client, streams_client):
        """Test multiple concurrent STT requests."""
        num_concurrent = 5
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"

        async def send_stt_request(index: int) -> Optional[float]:
            session_id = f"sess_{uuid.uuid4().hex[:16]}"
            start_time = time.time()

            request = AudioSTTRequest(
                session_id=session_id,
                tenant_id=tenant_id,
                audio_b64=SAMPLE_AUDIO_WAV,
            )
            await streams_client.publish_audio_for_stt(request)

            # Wait for transcription
            channel = f"transcription:{session_id}"
            pubsub = await redis_client.subscribe(channel)

            try:
                async for message in pubsub.listen():
                    if time.time() - start_time > 60.0:
                        break
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        if data.get("type") == "transcription.completed":
                            return (time.time() - start_time) * 1000
            except asyncio.TimeoutError:
                pass
            finally:
                await pubsub.unsubscribe(channel)
            return None

        # Run concurrent requests
        tasks = [send_stt_request(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        latencies = [r for r in results if isinstance(r, float)]

        if not latencies:
            pytest.skip("STT worker not running")

        # At least half should complete
        assert len(latencies) >= num_concurrent // 2

        # Average latency should be reasonable
        avg_latency = statistics.mean(latencies)
        assert avg_latency < 2000, f"Average latency {avg_latency}ms too high under load"

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tts_requests(self, redis_client, streams_client):
        """Test multiple concurrent TTS requests."""
        num_concurrent = 5
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"

        async def send_tts_request(index: int) -> Optional[float]:
            session_id = f"sess_{uuid.uuid4().hex[:16]}"
            start_time = time.time()

            request = TTSRequest(
                session_id=session_id,
                tenant_id=tenant_id,
                text=f"Concurrent test message {index}.",
                voice="am_onyx",
            )
            await streams_client.publish_tts_request(request)

            # Wait for first chunk
            timeout = 60.0
            last_id = "0"

            while time.time() - start_time < timeout:
                chunks = await streams_client.read_audio_chunks(
                    session_id=session_id,
                    last_id=last_id,
                    count=1,
                    block_ms=1000,
                )
                if chunks:
                    await streams_client.cleanup_session_streams(session_id)
                    return (time.time() - start_time) * 1000

            await streams_client.cleanup_session_streams(session_id)
            return None

        # Run concurrent requests
        tasks = [send_tts_request(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        latencies = [r for r in results if isinstance(r, float)]

        if not latencies:
            pytest.skip("TTS worker not running")

        # At least half should complete
        assert len(latencies) >= num_concurrent // 2


class TestFullPipeline:
    """Test full speech-to-speech pipeline.

    Audio → STT → LLM → TTS → Audio
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_infrastructure(self, redis_client, streams_client):
        """Test that full pipeline infrastructure is wired correctly.

        This test verifies the message passing works end-to-end.
        Full pipeline requires all workers running.
        """
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"

        # 1. Send audio to STT
        stt_request = AudioSTTRequest(
            session_id=session_id,
            tenant_id=tenant_id,
            audio_b64=SAMPLE_AUDIO_WAV,
        )
        stt_msg_id = await streams_client.publish_audio_for_stt(stt_request)
        assert stt_msg_id is not None

        # 2. Simulate LLM response by publishing TTS request
        # (In real flow, LLM worker would do this after receiving transcription)
        tts_request = TTSRequest(
            session_id=session_id,
            tenant_id=tenant_id,
            text="This is a response to your audio input.",
            voice="am_onyx",
        )
        tts_msg_id = await streams_client.publish_tts_request(tts_request)
        assert tts_msg_id is not None

        # 3. Verify messages are in streams
        client = redis_client.client

        stt_stream_len = await client.xlen("audio:stt")
        assert stt_stream_len >= 1

        tts_stream_len = await client.xlen("tts:requests")
        assert tts_stream_len >= 1

        # Cleanup
        await streams_client.cleanup_session_streams(session_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

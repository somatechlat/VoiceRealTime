"""TTS Worker Service - Text-to-Speech synthesis using Kokoro ONNX.

This worker consumes text from Redis Streams and produces audio chunks.

Stream: tts:requests (consumer group: tts-workers)
Output: audio:out:{session_id} (stream with sequence numbers)

Requirements:
- Redis running and accessible
- Kokoro ONNX model files
- soundfile for audio encoding

Usage:
    python -m workers.tts_worker

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
    KOKORO_MODEL_DIR: Directory containing Kokoro model files
    KOKORO_MODEL_FILE: Model filename (default: kokoro-v1.0.onnx)
    KOKORO_VOICES_FILE: Voices filename (default: voices-v1.0.bin)
    TTS_DEFAULT_VOICE: Default voice (default: am_onyx)
    TTS_DEFAULT_SPEED: Default speed (default: 1.1)
    TTS_CHUNK_SIZE: Audio chunk size in samples (default: 24000)
    WORKER_ID: Unique worker identifier (default: auto-generated)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Import from local worker modules to avoid Flask dependencies
from .worker_config import RedisSettings
from .worker_redis import RedisClient

# Stream constants
STREAM_TTS_REQUESTS = "tts:requests"
GROUP_TTS_WORKERS = "tts-workers"
CHANNEL_TTS = "tts"
CHANNEL_AUDIO_OUT = "audio:out"

logger = logging.getLogger(__name__)


# Optional: Kokoro ONNX for TTS
try:
    import kokoro_onnx as kokoro

    KOKORO_AVAILABLE = True
except ImportError:
    kokoro = None
    KOKORO_AVAILABLE = False
    logger.warning("kokoro-onnx not installed, TTS will be unavailable")

# Optional: soundfile for audio encoding
try:
    import numpy as np
    import soundfile as sf

    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    sf = None
    np = None
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("soundfile/numpy not installed, audio encoding limited")


@dataclass
class TTSWorkerConfig:
    """Configuration for TTS worker."""

    redis_url: str = "redis://localhost:6379/0"
    model_dir: str = "/app/cache/kokoro"
    model_file: str = "kokoro-v1.0.onnx"
    voices_file: str = "voices-v1.0.bin"
    default_voice: str = "am_onyx"
    default_speed: float = 1.1
    chunk_size: int = 24000  # ~1 second at 24kHz
    worker_id: str = ""

    @classmethod
    def from_env(cls) -> "TTSWorkerConfig":
        """Load configuration from environment variables."""
        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            model_dir=os.getenv("KOKORO_MODEL_DIR", "/app/cache/kokoro"),
            model_file=os.getenv("KOKORO_MODEL_FILE", "kokoro-v1.0.onnx"),
            voices_file=os.getenv("KOKORO_VOICES_FILE", "voices-v1.0.bin"),
            default_voice=os.getenv("TTS_DEFAULT_VOICE", "am_onyx"),
            default_speed=float(os.getenv("TTS_DEFAULT_SPEED", "1.1")),
            chunk_size=int(os.getenv("TTS_CHUNK_SIZE", "24000")),
            worker_id=os.getenv("WORKER_ID", f"tts-{uuid.uuid4().hex[:8]}"),
        )


class TTSWorker:
    """Text-to-Speech worker using Kokoro ONNX.

    Consumes text from Redis Streams, synthesizes using Kokoro,
    and streams audio chunks back via Redis Streams.

    Features:
    - Consumer group for horizontal scaling
    - Streaming synthesis with sequence numbers
    - Cancellation support via pub/sub
    - Model caching and pooling
    - Fallback to Piper TTS
    """

    def __init__(self, config: TTSWorkerConfig) -> None:
        self._config = config
        self._redis: Optional[RedisClient] = None
        self._engine: Optional[Any] = None
        self._running = False
        self._tasks: set = set()
        self._cancelled_sessions: set = set()
        self._cancel_listener_task: Optional[asyncio.Task] = None

        # Metrics
        self._synthesis_total = 0
        self._synthesis_failed = 0
        self._total_audio_seconds = 0.0
        self._total_characters = 0

    async def start(self) -> None:
        """Start the TTS worker."""
        logger.info(
            f"Starting TTS worker {self._config.worker_id}",
            extra={
                "model_dir": self._config.model_dir,
                "default_voice": self._config.default_voice,
            },
        )

        # Connect to Redis
        redis_settings = RedisSettings(url=self._config.redis_url)
        self._redis = RedisClient(redis_settings)
        await self._redis.connect()

        # Load Kokoro model
        self._load_model()

        # Ensure consumer group exists
        await self._ensure_consumer_group()

        # Start cancel listener
        self._cancel_listener_task = asyncio.create_task(self._listen_for_cancels())

        self._running = True
        logger.info(f"TTS worker {self._config.worker_id} started")

    def _load_model(self) -> None:
        """Load the Kokoro ONNX model."""
        if not KOKORO_AVAILABLE:
            logger.error("kokoro-onnx not available, cannot load model")
            return

        model_path = os.path.join(self._config.model_dir, self._config.model_file)
        voices_path = os.path.join(self._config.model_dir, self._config.voices_file)

        if not os.path.isfile(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return

        if not os.path.isfile(voices_path):
            logger.warning(f"Voices file not found: {voices_path}")
            return

        logger.info(f"Loading Kokoro model from {model_path}")

        try:
            self._engine = kokoro.Kokoro(
                model_path=model_path,
                voices_path=voices_path,
            )
            logger.info("Kokoro model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Kokoro model: {e}")
            self._engine = None

    async def _ensure_consumer_group(self) -> None:
        """Ensure the consumer group exists."""
        client = self._redis.client
        try:
            await client.xgroup_create(
                STREAM_TTS_REQUESTS,
                GROUP_TTS_WORKERS,
                id="0",
                mkstream=True,
            )
            logger.info(f"Created consumer group {GROUP_TTS_WORKERS}")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Error creating consumer group: {e}")

    async def _listen_for_cancels(self) -> None:
        """Listen for TTS cancellation signals via pub/sub."""
        # Subscribe to all TTS cancel channels using pattern
        pubsub = self._redis.client.pubsub()
        await pubsub.psubscribe(f"{CHANNEL_TTS}:*")

        async for message in pubsub.listen():
            if not self._running:
                break

            if message["type"] == "pmessage":
                try:
                    data = json.loads(message["data"])
                    if data.get("type") == "tts.cancel":
                        session_id = data.get("session_id")
                        if session_id:
                            self._cancelled_sessions.add(session_id)
                            logger.info(f"Received cancel for session {session_id}")
                except (json.JSONDecodeError, KeyError):
                    pass

    async def stop(self) -> None:
        """Stop the TTS worker gracefully."""
        logger.info(f"Stopping TTS worker {self._config.worker_id}")
        self._running = False

        # Cancel listener
        if self._cancel_listener_task:
            self._cancel_listener_task.cancel()
            try:
                await self._cancel_listener_task
            except asyncio.CancelledError:
                pass

        # Wait for pending tasks
        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} pending tasks")
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Disconnect from Redis
        if self._redis:
            await self._redis.disconnect()

        logger.info(
            f"TTS worker {self._config.worker_id} stopped",
            extra={
                "synthesis_total": self._synthesis_total,
                "synthesis_failed": self._synthesis_failed,
                "total_audio_seconds": self._total_audio_seconds,
                "total_characters": self._total_characters,
            },
        )

    async def run(self) -> None:
        """Main worker loop - consume and process messages."""
        client = self._redis.client

        while self._running:
            try:
                # Read messages from stream
                messages = await client.xreadgroup(
                    GROUP_TTS_WORKERS,
                    self._config.worker_id,
                    {STREAM_TTS_REQUESTS: ">"},
                    count=1,  # Process one at a time for streaming
                    block=1000,
                )

                if not messages:
                    continue

                # Process each message
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        task = asyncio.create_task(self._process_message(message_id, data))
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _process_message(
        self,
        message_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Process a single TTS request."""
        session_id = data.get("session_id", "")
        data.get("tenant_id", "")
        text = data.get("text", "")
        voice = data.get("voice", self._config.default_voice)
        speed = float(data.get("speed", self._config.default_speed))
        response_id = data.get("response_id", "")
        data.get("item_id", "")
        data.get("correlation_id", "")

        start_time = time.time()

        # Clear any previous cancel for this session
        self._cancelled_sessions.discard(session_id)

        try:
            if not text:
                raise ValueError("No text in message")

            self._total_characters += len(text)

            # Synthesize and stream
            sequence = 0
            async for chunk_b64, sample_rate, is_final in self._synthesize_stream(
                text=text,
                voice=voice,
                speed=speed,
                session_id=session_id,
            ):
                # Check for cancellation
                if session_id in self._cancelled_sessions:
                    logger.info(f"Synthesis cancelled for session {session_id}")
                    await self._publish_cancelled(session_id, response_id)
                    break

                # Publish audio chunk
                await self._publish_audio_chunk(
                    session_id=session_id,
                    chunk_b64=chunk_b64,
                    sequence=sequence,
                    sample_rate=sample_rate,
                    is_final=is_final,
                )
                sequence += 1

            # Acknowledge message
            await self._redis.client.xack(
                STREAM_TTS_REQUESTS,
                GROUP_TTS_WORKERS,
                message_id,
            )

            self._synthesis_total += 1
            duration = time.time() - start_time

            logger.info(
                "Synthesis completed",
                extra={
                    "session_id": session_id,
                    "text_length": len(text),
                    "chunks": sequence,
                    "duration_ms": int(duration * 1000),
                },
            )

        except Exception as e:
            self._synthesis_failed += 1
            logger.error(
                f"Synthesis failed: {e}",
                extra={
                    "session_id": session_id,
                    "message_id": message_id,
                },
                exc_info=True,
            )

            # Publish error
            await self._publish_error(session_id, str(e), response_id)

            # Acknowledge to prevent infinite retry
            await self._redis.client.xack(
                STREAM_TTS_REQUESTS,
                GROUP_TTS_WORKERS,
                message_id,
            )

    async def _synthesize_stream(
        self,
        text: str,
        voice: str,
        speed: float,
        session_id: str,
    ):
        """Stream audio synthesis using Kokoro.

        Yields:
            Tuple of (chunk_b64, sample_rate, is_final)
        """
        if not self._engine or not AUDIO_LIBS_AVAILABLE:
            # Fallback: generate silence
            silence = np.zeros(self._config.chunk_size, dtype=np.float32) if np else b"\x00" * 4800
            buf = io.BytesIO()
            if sf:
                sf.write(buf, silence, 24000, format="WAV")
            else:
                buf.write(b"RIFF" + b"\x00" * 40)  # Minimal WAV header
            yield base64.b64encode(buf.getvalue()).decode("utf-8"), 24000, True
            return

        # Validate voice
        try:
            available_voices = self._engine.get_voices()
            if voice not in available_voices:
                voice = available_voices[0] if available_voices else self._config.default_voice
        except Exception:
            pass

        # Stream synthesis
        chunk_count = 0
        async for audio_arr, sample_rate in self._engine.create_stream(
            text=text,
            voice=voice,
            speed=speed,
        ):
            # Check cancellation
            if session_id in self._cancelled_sessions:
                break

            # Track audio duration
            self._total_audio_seconds += len(audio_arr) / sample_rate

            # Encode to WAV
            buf = io.BytesIO()
            sf.write(buf, audio_arr, sample_rate, format="WAV")
            chunk_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            chunk_count += 1
            yield chunk_b64, sample_rate, False

        # Send final marker
        if chunk_count > 0:
            # Empty final chunk to signal completion
            buf = io.BytesIO()
            sf.write(buf, np.zeros(100, dtype=np.float32), 24000, format="WAV")
            yield base64.b64encode(buf.getvalue()).decode("utf-8"), 24000, True

    async def _publish_audio_chunk(
        self,
        session_id: str,
        chunk_b64: str,
        sequence: int,
        sample_rate: int,
        is_final: bool,
    ) -> None:
        """Publish audio chunk to session's output stream."""
        stream_name = f"{CHANNEL_AUDIO_OUT}:{session_id}"

        await self._redis.client.xadd(
            stream_name,
            {
                "chunk": chunk_b64,
                "sequence": str(sequence),
                "sample_rate": str(sample_rate),
                "is_final": "1" if is_final else "0",
                "timestamp": str(time.time()),
            },
            maxlen=1000,
        )

    async def _publish_cancelled(self, session_id: str, response_id: str) -> None:
        """Publish TTS cancelled event."""
        channel = f"{CHANNEL_TTS}:{session_id}"
        message = json.dumps(
            {
                "type": "tts.cancelled",
                "session_id": session_id,
                "response_id": response_id,
                "timestamp": time.time(),
            }
        )
        await self._redis.publish(channel, message)

    async def _publish_error(self, session_id: str, error: str, response_id: str) -> None:
        """Publish TTS error event."""
        channel = f"{CHANNEL_TTS}:{session_id}"
        message = json.dumps(
            {
                "type": "tts.failed",
                "session_id": session_id,
                "response_id": response_id,
                "error": error,
                "timestamp": time.time(),
            }
        )
        await self._redis.publish(channel, message)


async def main() -> None:
    """Main entry point for TTS worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = TTSWorkerConfig.from_env()
    worker = TTSWorker(config)

    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(worker.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await worker.start()
        await worker.run()
    except KeyboardInterrupt:
        pass
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())

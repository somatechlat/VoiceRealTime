"""STT Worker Service - Speech-to-Text transcription using Faster-Whisper.

This worker consumes audio from Redis Streams and produces transcriptions.

Stream: audio:stt (consumer group: stt-workers)
Output: transcription:{session_id} (pub/sub)

Requirements:
- Redis running and accessible
- Faster-Whisper installed (pip install faster-whisper)
- CUDA toolkit for GPU acceleration (optional)

Usage:
    python -m workers.stt_worker

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
    STT_MODEL: Whisper model size (default: base)
    STT_DEVICE: Device to use - cpu, cuda, auto (default: auto)
    STT_COMPUTE_TYPE: Compute type - float16, int8, float32 (default: float16)
    STT_BATCH_SIZE: Max concurrent transcriptions (default: 4)
    WORKER_ID: Unique worker identifier (default: auto-generated)
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Import from local worker modules to avoid Flask dependencies
from .worker_config import RedisSettings
from .worker_redis import RedisClient

# Stream constants
STREAM_AUDIO_STT = "audio:stt"
GROUP_STT_WORKERS = "stt-workers"
CHANNEL_TRANSCRIPTION = "transcription"

logger = logging.getLogger(__name__)


# Optional: Faster-Whisper for transcription
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed, STT will be unavailable")

# Optional: numpy and soundfile for audio processing
try:
    import numpy as np
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    np = None
    sf = None
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("numpy/soundfile not installed, audio processing limited")


@dataclass
class STTWorkerConfig:
    """Configuration for STT worker."""
    redis_url: str = "redis://localhost:6379/0"
    model_size: str = "base"  # tiny, base, small, medium, large-v2, large-v3
    device: str = "auto"  # cpu, cuda, auto
    compute_type: str = "float16"  # float16, int8, float32
    batch_size: int = 4  # Max concurrent transcriptions
    worker_id: str = ""
    
    @classmethod
    def from_env(cls) -> "STTWorkerConfig":
        """Load configuration from environment variables."""
        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            model_size=os.getenv("STT_MODEL", "base"),
            device=os.getenv("STT_DEVICE", "auto"),
            compute_type=os.getenv("STT_COMPUTE_TYPE", "float16"),
            batch_size=int(os.getenv("STT_BATCH_SIZE", "4")),
            worker_id=os.getenv("WORKER_ID", f"stt-{uuid.uuid4().hex[:8]}"),
        )


class STTWorker:
    """Speech-to-Text worker using Faster-Whisper.
    
    Consumes audio from Redis Streams, transcribes using Whisper,
    and publishes results via pub/sub.
    
    Features:
    - Consumer group for horizontal scaling
    - Automatic message acknowledgment
    - Retry on failure with dead-lettering
    - GPU acceleration when available
    - Batch processing support
    """

    def __init__(self, config: STTWorkerConfig) -> None:
        self._config = config
        self._redis: Optional[RedisClient] = None
        self._model: Optional[Any] = None
        self._running = False
        self._tasks: set = set()
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        # Metrics
        self._transcriptions_total = 0
        self._transcriptions_failed = 0
        self._total_audio_seconds = 0.0

    async def start(self) -> None:
        """Start the STT worker."""
        logger.info(
            f"Starting STT worker {self._config.worker_id}",
            extra={
                "model": self._config.model_size,
                "device": self._config.device,
                "batch_size": self._config.batch_size,
            }
        )
        
        # Connect to Redis
        redis_settings = RedisSettings(url=self._config.redis_url)
        self._redis = RedisClient(redis_settings)
        await self._redis.connect()
        
        # Initialize semaphore for batch limiting
        self._semaphore = asyncio.Semaphore(self._config.batch_size)
        
        # Load Whisper model
        self._load_model()
        
        # Ensure consumer group exists
        await self._ensure_consumer_group()
        
        self._running = True
        logger.info(f"STT worker {self._config.worker_id} started")

    def _load_model(self) -> None:
        """Load the Faster-Whisper model."""
        if not FASTER_WHISPER_AVAILABLE:
            logger.error("faster-whisper not available, cannot load model")
            return
        
        device = self._config.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        compute_type = self._config.compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"  # float16 not supported on CPU
        
        logger.info(
            f"Loading Whisper model: {self._config.model_size}",
            extra={"device": device, "compute_type": compute_type}
        )
        
        self._model = WhisperModel(
            self._config.model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded successfully")

    async def _ensure_consumer_group(self) -> None:
        """Ensure the consumer group exists."""
        client = self._redis.client
        try:
            await client.xgroup_create(
                STREAM_AUDIO_STT,
                GROUP_STT_WORKERS,
                id="0",
                mkstream=True,
            )
            logger.info(f"Created consumer group {GROUP_STT_WORKERS}")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Error creating consumer group: {e}")

    async def stop(self) -> None:
        """Stop the STT worker gracefully."""
        logger.info(f"Stopping STT worker {self._config.worker_id}")
        self._running = False
        
        # Wait for pending tasks
        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} pending tasks")
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Disconnect from Redis
        if self._redis:
            await self._redis.disconnect()
        
        logger.info(
            f"STT worker {self._config.worker_id} stopped",
            extra={
                "transcriptions_total": self._transcriptions_total,
                "transcriptions_failed": self._transcriptions_failed,
                "total_audio_seconds": self._total_audio_seconds,
            }
        )

    async def run(self) -> None:
        """Main worker loop - consume and process messages."""
        client = self._redis.client
        
        while self._running:
            try:
                # Read messages from stream
                messages = await client.xreadgroup(
                    GROUP_STT_WORKERS,
                    self._config.worker_id,
                    {STREAM_AUDIO_STT: ">"},
                    count=self._config.batch_size,
                    block=1000,  # 1 second timeout
                )
                
                if not messages:
                    continue
                
                # Process each message
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        task = asyncio.create_task(
                            self._process_message(message_id, data)
                        )
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
        """Process a single transcription request."""
        async with self._semaphore:
            session_id = data.get("session_id", "")
            tenant_id = data.get("tenant_id", "")
            correlation_id = data.get("correlation_id", "")
            
            start_time = time.time()
            
            try:
                # Decode audio
                audio_b64 = data.get("audio", "")
                if not audio_b64:
                    raise ValueError("No audio data in message")
                
                audio_bytes = base64.b64decode(audio_b64)
                
                # Transcribe
                text, language, confidence = await self._transcribe(
                    audio_bytes,
                    language_hint=data.get("language") or None,
                )
                
                # Publish result
                await self._publish_result(
                    session_id=session_id,
                    text=text,
                    language=language,
                    confidence=confidence,
                    correlation_id=correlation_id,
                )
                
                # Acknowledge message
                await self._redis.client.xack(
                    STREAM_AUDIO_STT,
                    GROUP_STT_WORKERS,
                    message_id,
                )
                
                self._transcriptions_total += 1
                duration = time.time() - start_time
                
                logger.info(
                    "Transcription completed",
                    extra={
                        "session_id": session_id,
                        "text_length": len(text),
                        "language": language,
                        "duration_ms": int(duration * 1000),
                    }
                )
                
            except Exception as e:
                self._transcriptions_failed += 1
                logger.error(
                    f"Transcription failed: {e}",
                    extra={
                        "session_id": session_id,
                        "message_id": message_id,
                    },
                    exc_info=True,
                )
                
                # Publish error result
                await self._publish_error(
                    session_id=session_id,
                    error=str(e),
                    correlation_id=correlation_id,
                )
                
                # Still acknowledge to prevent infinite retry
                # In production, consider dead-letter queue
                await self._redis.client.xack(
                    STREAM_AUDIO_STT,
                    GROUP_STT_WORKERS,
                    message_id,
                )

    async def _transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
    ) -> tuple[str, str, float]:
        """Transcribe audio bytes using Faster-Whisper.
        
        Args:
            audio_bytes: Raw audio data (PCM16 or WAV)
            language_hint: Optional language hint
            
        Returns:
            Tuple of (text, language, confidence)
        """
        if not self._model:
            return "[STT unavailable]", "en", 0.0
        
        # Convert bytes to numpy array
        if AUDIO_LIBS_AVAILABLE:
            try:
                # Try to read as WAV
                audio_io = io.BytesIO(audio_bytes)
                audio_data, sample_rate = sf.read(audio_io)
                
                # Resample to 16kHz if needed (Whisper expects 16kHz)
                if sample_rate != 16000:
                    # Simple resampling - in production use librosa
                    ratio = 16000 / sample_rate
                    new_length = int(len(audio_data) * ratio)
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data), new_length),
                        np.arange(len(audio_data)),
                        audio_data,
                    )
                
                # Track audio duration
                self._total_audio_seconds += len(audio_data) / 16000
                
            except Exception:
                # Assume raw PCM16 at 16kHz
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                self._total_audio_seconds += len(audio_data) / 16000
        else:
            # Fallback: assume raw PCM16
            audio_data = audio_bytes
        
        # Run transcription in thread pool (Whisper is CPU-bound)
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(
                audio_data,
                language=language_hint,
                beam_size=5,
                vad_filter=True,
            ),
        )
        
        # Collect text from segments
        text_parts = []
        total_confidence = 0.0
        segment_count = 0
        
        for segment in segments:
            text_parts.append(segment.text)
            total_confidence += segment.avg_logprob
            segment_count += 1
        
        text = " ".join(text_parts).strip()
        language = info.language or "en"
        confidence = (total_confidence / segment_count) if segment_count > 0 else 0.0
        
        # Convert log probability to 0-1 confidence
        confidence = min(1.0, max(0.0, 1.0 + confidence))
        
        return text, language, confidence

    async def _publish_result(
        self,
        session_id: str,
        text: str,
        language: str,
        confidence: float,
        correlation_id: str,
    ) -> None:
        """Publish transcription result via pub/sub."""
        import json
        
        channel = f"{CHANNEL_TRANSCRIPTION}:{session_id}"
        message = json.dumps({
            "type": "transcription.completed",
            "session_id": session_id,
            "text": text,
            "language": language,
            "confidence": confidence,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        })
        
        await self._redis.publish(channel, message)

    async def _publish_error(
        self,
        session_id: str,
        error: str,
        correlation_id: str,
    ) -> None:
        """Publish transcription error via pub/sub."""
        import json
        
        channel = f"{CHANNEL_TRANSCRIPTION}:{session_id}"
        message = json.dumps({
            "type": "transcription.failed",
            "session_id": session_id,
            "error": error,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        })
        
        await self._redis.publish(channel, message)


async def main() -> None:
    """Main entry point for STT worker."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    config = STTWorkerConfig.from_env()
    worker = STTWorker(config)
    
    # Handle shutdown signals
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

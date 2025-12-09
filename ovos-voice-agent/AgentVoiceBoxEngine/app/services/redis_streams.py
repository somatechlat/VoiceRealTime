"""Redis Streams client for worker communication.

This module provides Redis Streams-based message passing between:
- Gateway → STT Worker (audio:stt stream)
- Gateway → TTS Worker (tts:requests stream)
- TTS Worker → Gateway (audio:out:{session_id} stream)
- STT Worker → Gateway (transcription:{session_id} pub/sub)

Stream names per design.md:
- audio:stt - Audio chunks for transcription
- tts:requests - Text for synthesis
- audio:out:{session_id} - Synthesized audio chunks (per session)

Pub/Sub channels:
- transcription:{session_id} - Transcription results
- tts:{session_id} - TTS completion events
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from .redis_client import RedisClient

logger = logging.getLogger(__name__)


# Stream names
STREAM_AUDIO_STT = "audio:stt"
STREAM_TTS_REQUESTS = "tts:requests"

# Consumer groups
GROUP_STT_WORKERS = "stt-workers"
GROUP_TTS_WORKERS = "tts-workers"

# Pub/Sub channel prefixes
CHANNEL_TRANSCRIPTION = "transcription"
CHANNEL_TTS = "tts"
CHANNEL_AUDIO_OUT = "audio:out"


@dataclass
class StreamMessage:
    """A message from a Redis Stream."""
    message_id: str
    stream_name: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AudioSTTRequest:
    """Request to transcribe audio."""
    session_id: str
    tenant_id: str
    audio_b64: str  # Base64 encoded audio
    language: Optional[str] = None
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, str]:
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "audio": self.audio_b64,
            "language": self.language or "",
            "correlation_id": self.correlation_id,
            "timestamp": str(time.time()),
        }


@dataclass
class TTSRequest:
    """Request to synthesize text to speech."""
    session_id: str
    tenant_id: str
    text: str
    voice: str = "am_onyx"
    speed: float = 1.1
    response_id: str = ""
    item_id: str = ""
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, str]:
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "text": self.text,
            "voice": self.voice,
            "speed": str(self.speed),
            "response_id": self.response_id,
            "item_id": self.item_id,
            "correlation_id": self.correlation_id,
            "timestamp": str(time.time()),
        }


@dataclass
class TranscriptionResult:
    """Result from STT worker."""
    session_id: str
    text: str
    language: str
    confidence: float
    correlation_id: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionResult":
        return cls(
            session_id=data["session_id"],
            text=data["text"],
            language=data.get("language", "en"),
            confidence=float(data.get("confidence", 0.0)),
            correlation_id=data.get("correlation_id", ""),
        )


class RedisStreamsClient:
    """Client for Redis Streams-based worker communication.
    
    Provides:
    - Publishing audio to STT workers
    - Publishing text to TTS workers
    - Subscribing to transcription results
    - Subscribing to TTS audio chunks
    """

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client
        self._initialized_streams: set[str] = set()

    async def _ensure_stream_exists(self, stream_name: str, group_name: str) -> None:
        """Ensure stream and consumer group exist."""
        if stream_name in self._initialized_streams:
            return
        
        client = self._redis.client
        try:
            # Create consumer group (creates stream if not exists)
            await client.xgroup_create(
                stream_name,
                group_name,
                id="0",
                mkstream=True,
            )
            logger.info(f"Created stream {stream_name} with group {group_name}")
        except Exception as e:
            # Group may already exist
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Error creating stream group: {e}")
        
        self._initialized_streams.add(stream_name)

    async def publish_audio_for_stt(self, request: AudioSTTRequest) -> str:
        """Publish audio to STT worker stream.
        
        Args:
            request: Audio transcription request
            
        Returns:
            Message ID from Redis
        """
        await self._ensure_stream_exists(STREAM_AUDIO_STT, GROUP_STT_WORKERS)
        
        client = self._redis.client
        message_id = await client.xadd(
            STREAM_AUDIO_STT,
            request.to_dict(),
            maxlen=10000,  # Keep last 10K messages
        )
        
        logger.debug(
            "Published audio for STT",
            extra={
                "session_id": request.session_id,
                "message_id": message_id,
                "correlation_id": request.correlation_id,
            }
        )
        return message_id

    async def publish_tts_request(self, request: TTSRequest) -> str:
        """Publish text to TTS worker stream.
        
        Args:
            request: TTS synthesis request
            
        Returns:
            Message ID from Redis
        """
        await self._ensure_stream_exists(STREAM_TTS_REQUESTS, GROUP_TTS_WORKERS)
        
        client = self._redis.client
        message_id = await client.xadd(
            STREAM_TTS_REQUESTS,
            request.to_dict(),
            maxlen=10000,
        )
        
        logger.debug(
            "Published TTS request",
            extra={
                "session_id": request.session_id,
                "message_id": message_id,
                "text_length": len(request.text),
            }
        )
        return message_id

    async def publish_audio_chunk(
        self,
        session_id: str,
        chunk_b64: str,
        sequence: int,
        sample_rate: int = 24000,
        is_final: bool = False,
    ) -> str:
        """Publish audio chunk to session's output stream.
        
        Used by TTS workers to stream audio back to gateway.
        
        Args:
            session_id: Target session
            chunk_b64: Base64 encoded audio chunk
            sequence: Chunk sequence number for ordering
            sample_rate: Audio sample rate
            is_final: Whether this is the last chunk
            
        Returns:
            Message ID from Redis
        """
        stream_name = f"{CHANNEL_AUDIO_OUT}:{session_id}"
        client = self._redis.client
        
        message_id = await client.xadd(
            stream_name,
            {
                "chunk": chunk_b64,
                "sequence": str(sequence),
                "sample_rate": str(sample_rate),
                "is_final": "1" if is_final else "0",
                "timestamp": str(time.time()),
            },
            maxlen=1000,  # Keep last 1K chunks per session
        )
        return message_id

    async def publish_transcription_result(
        self,
        session_id: str,
        text: str,
        language: str = "en",
        confidence: float = 1.0,
        correlation_id: str = "",
    ) -> int:
        """Publish transcription result via pub/sub.
        
        Used by STT workers to send results back to gateway.
        
        Args:
            session_id: Target session
            text: Transcribed text
            language: Detected language
            confidence: Confidence score
            correlation_id: Request correlation ID
            
        Returns:
            Number of subscribers that received the message
        """
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
        
        return await self._redis.publish(channel, message)

    async def subscribe_to_transcriptions(
        self,
        session_id: str,
    ) -> AsyncIterator[TranscriptionResult]:
        """Subscribe to transcription results for a session.
        
        Args:
            session_id: Session to subscribe to
            
        Yields:
            TranscriptionResult objects as they arrive
        """
        channel = f"{CHANNEL_TRANSCRIPTION}:{session_id}"
        pubsub = await self._redis.subscribe(channel)
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    yield TranscriptionResult.from_dict(data)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Invalid transcription message: {e}")

    async def read_audio_chunks(
        self,
        session_id: str,
        last_id: str = "0",
        count: int = 100,
        block_ms: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Read audio chunks from session's output stream.
        
        Used by gateway to receive TTS audio chunks.
        
        Args:
            session_id: Session to read from
            last_id: Last message ID received (for continuation)
            count: Maximum messages to read
            block_ms: Block timeout in milliseconds
            
        Returns:
            List of audio chunk dictionaries
        """
        stream_name = f"{CHANNEL_AUDIO_OUT}:{session_id}"
        client = self._redis.client
        
        try:
            result = await client.xread(
                {stream_name: last_id},
                count=count,
                block=block_ms,
            )
            
            if not result:
                return []
            
            chunks = []
            for stream, messages in result:
                for msg_id, data in messages:
                    chunks.append({
                        "message_id": msg_id,
                        "chunk": data.get("chunk", ""),
                        "sequence": int(data.get("sequence", 0)),
                        "sample_rate": int(data.get("sample_rate", 24000)),
                        "is_final": data.get("is_final") == "1",
                    })
            return chunks
            
        except Exception as e:
            logger.warning(f"Error reading audio chunks: {e}")
            return []

    async def cancel_tts(self, session_id: str) -> int:
        """Signal TTS cancellation for a session.
        
        Args:
            session_id: Session to cancel
            
        Returns:
            Number of subscribers notified
        """
        channel = f"{CHANNEL_TTS}:{session_id}"
        message = json.dumps({
            "type": "tts.cancel",
            "session_id": session_id,
            "timestamp": time.time(),
        })
        return await self._redis.publish(channel, message)

    async def cleanup_session_streams(self, session_id: str) -> None:
        """Clean up streams for a closed session.
        
        Args:
            session_id: Session to clean up
        """
        stream_name = f"{CHANNEL_AUDIO_OUT}:{session_id}"
        client = self._redis.client
        
        try:
            await client.delete(stream_name)
            logger.debug(f"Cleaned up stream {stream_name}")
        except Exception as e:
            logger.warning(f"Error cleaning up session stream: {e}")


# Global instance
_streams_client: Optional[RedisStreamsClient] = None


def get_streams_client() -> Optional[RedisStreamsClient]:
    """Get the global Redis Streams client."""
    return _streams_client


def init_streams_client(redis_client: RedisClient) -> RedisStreamsClient:
    """Initialize the global Redis Streams client."""
    global _streams_client
    _streams_client = RedisStreamsClient(redis_client)
    return _streams_client


__all__ = [
    "RedisStreamsClient",
    "StreamMessage",
    "AudioSTTRequest",
    "TTSRequest",
    "TranscriptionResult",
    "get_streams_client",
    "init_streams_client",
    "STREAM_AUDIO_STT",
    "STREAM_TTS_REQUESTS",
    "GROUP_STT_WORKERS",
    "GROUP_TTS_WORKERS",
]

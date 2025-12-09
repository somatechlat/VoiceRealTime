"""LLM Worker Service - Language Model inference with multi-provider support.

This worker handles LLM requests with support for multiple providers:
- OpenAI (GPT-4, GPT-3.5)
- Groq (Llama, Mixtral)
- Ollama (self-hosted models)

Features:
- Circuit breaker for provider failover
- Streaming token generation
- Per-tenant provider configuration
- BYOK (Bring Your Own Key) support

Usage:
    python -m workers.llm_worker

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
    LLM_DEFAULT_PROVIDER: Default provider (default: groq)
    OPENAI_API_KEY: OpenAI API key
    GROQ_API_KEY: Groq API key
    OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    LLM_DEFAULT_MODEL: Default model (default: llama-3.1-70b-versatile)
    CIRCUIT_BREAKER_THRESHOLD: Failures before opening (default: 5)
    CIRCUIT_BREAKER_TIMEOUT: Recovery timeout in seconds (default: 30)
    WORKER_ID: Unique worker identifier (default: auto-generated)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import RedisSettings
from app.services.redis_client import RedisClient

logger = logging.getLogger(__name__)


# Optional: httpx for async HTTP requests
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed, LLM requests will fail")


# Stream names for LLM communication
STREAM_LLM_REQUESTS = "llm:requests"
GROUP_LLM_WORKERS = "llm-workers"
CHANNEL_LLM_RESPONSE = "llm:response"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for provider failover.
    
    Opens after threshold failures, closes after timeout.
    """
    threshold: int = 5
    timeout: float = 30.0
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: CircuitState = CircuitState.CLOSED
    
    def record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker half-open, testing recovery")
                return True
            return False
        
        # Half-open: allow one request to test
        return True


@dataclass
class LLMWorkerConfig:
    """Configuration for LLM worker."""
    redis_url: str = "redis://localhost:6379/0"
    default_provider: str = "groq"
    openai_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama-3.1-70b-versatile"
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    max_tokens: int = 1024
    temperature: float = 0.7
    worker_id: str = ""
    
    @classmethod
    def from_env(cls) -> "LLMWorkerConfig":
        """Load configuration from environment variables."""
        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            default_provider=os.getenv("LLM_DEFAULT_PROVIDER", "groq"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            default_model=os.getenv("LLM_DEFAULT_MODEL", "llama-3.1-70b-versatile"),
            circuit_breaker_threshold=int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
            circuit_breaker_timeout=float(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "30")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            worker_id=os.getenv("WORKER_ID", f"llm-{uuid.uuid4().hex[:8]}"),
        )


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMWorkerConfig) -> None:
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None
    
    async def start(self) -> None:
        """Initialize the provider."""
        if HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(timeout=60.0)
    
    async def stop(self) -> None:
        """Cleanup the provider."""
        if self._client:
            await self._client.aclose()
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Generate streaming response. Override in subclasses."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        if not self._client or not self._config.openai_api_key:
            raise RuntimeError("OpenAI not configured")
        
        model = model or "gpt-4o-mini"
        
        async with self._client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._config.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": temperature or self._config.temperature,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass


class GroqProvider(LLMProvider):
    """Groq API provider."""
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        if not self._client or not self._config.groq_api_key:
            raise RuntimeError("Groq not configured")
        
        model = model or self._config.default_model
        
        async with self._client.stream(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._config.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": temperature or self._config.temperature,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass


class OllamaProvider(LLMProvider):
    """Ollama (self-hosted) provider."""
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        if not self._client:
            raise RuntimeError("Ollama client not initialized")
        
        model = model or "llama3.1"
        
        async with self._client.stream(
            "POST",
            f"{self._config.ollama_base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "num_predict": max_tokens or self._config.max_tokens,
                    "temperature": temperature or self._config.temperature,
                },
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        pass


class LLMWorker:
    """LLM Worker with multi-provider support and circuit breaker.
    
    Features:
    - Multiple provider support (OpenAI, Groq, Ollama)
    - Circuit breaker for automatic failover
    - Streaming token generation
    - Per-tenant configuration
    """

    def __init__(self, config: LLMWorkerConfig) -> None:
        self._config = config
        self._redis: Optional[RedisClient] = None
        self._running = False
        self._tasks: set = set()
        
        # Providers with circuit breakers
        self._providers: Dict[str, LLMProvider] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._provider_priority = ["groq", "openai", "ollama"]
        
        # Metrics
        self._requests_total = 0
        self._requests_failed = 0
        self._tokens_generated = 0

    async def start(self) -> None:
        """Start the LLM worker."""
        logger.info(f"Starting LLM worker {self._config.worker_id}")
        
        # Connect to Redis
        redis_settings = RedisSettings(url=self._config.redis_url)
        self._redis = RedisClient(redis_settings)
        await self._redis.connect()
        
        # Initialize providers
        await self._init_providers()
        
        # Ensure consumer group exists
        await self._ensure_consumer_group()
        
        self._running = True
        logger.info(f"LLM worker {self._config.worker_id} started")

    async def _init_providers(self) -> None:
        """Initialize LLM providers."""
        # OpenAI
        if self._config.openai_api_key:
            provider = OpenAIProvider(self._config)
            await provider.start()
            self._providers["openai"] = provider
            self._circuit_breakers["openai"] = CircuitBreaker(
                threshold=self._config.circuit_breaker_threshold,
                timeout=self._config.circuit_breaker_timeout,
            )
            logger.info("OpenAI provider initialized")
        
        # Groq
        if self._config.groq_api_key:
            provider = GroqProvider(self._config)
            await provider.start()
            self._providers["groq"] = provider
            self._circuit_breakers["groq"] = CircuitBreaker(
                threshold=self._config.circuit_breaker_threshold,
                timeout=self._config.circuit_breaker_timeout,
            )
            logger.info("Groq provider initialized")
        
        # Ollama (always available if URL is set)
        provider = OllamaProvider(self._config)
        await provider.start()
        self._providers["ollama"] = provider
        self._circuit_breakers["ollama"] = CircuitBreaker(
            threshold=self._config.circuit_breaker_threshold,
            timeout=self._config.circuit_breaker_timeout,
        )
        logger.info("Ollama provider initialized")

    async def _ensure_consumer_group(self) -> None:
        """Ensure the consumer group exists."""
        client = self._redis.client
        try:
            await client.xgroup_create(
                STREAM_LLM_REQUESTS,
                GROUP_LLM_WORKERS,
                id="0",
                mkstream=True,
            )
            logger.info(f"Created consumer group {GROUP_LLM_WORKERS}")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Error creating consumer group: {e}")

    async def stop(self) -> None:
        """Stop the LLM worker gracefully."""
        logger.info(f"Stopping LLM worker {self._config.worker_id}")
        self._running = False
        
        # Wait for pending tasks
        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} pending tasks")
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop providers
        for provider in self._providers.values():
            await provider.stop()
        
        # Disconnect from Redis
        if self._redis:
            await self._redis.disconnect()
        
        logger.info(
            f"LLM worker {self._config.worker_id} stopped",
            extra={
                "requests_total": self._requests_total,
                "requests_failed": self._requests_failed,
                "tokens_generated": self._tokens_generated,
            }
        )

    async def run(self) -> None:
        """Main worker loop."""
        client = self._redis.client
        
        while self._running:
            try:
                messages = await client.xreadgroup(
                    GROUP_LLM_WORKERS,
                    self._config.worker_id,
                    {STREAM_LLM_REQUESTS: ">"},
                    count=1,
                    block=1000,
                )
                
                if not messages:
                    continue
                
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
        """Process a single LLM request."""
        session_id = data.get("session_id", "")
        tenant_id = data.get("tenant_id", "")
        messages_json = data.get("messages", "[]")
        provider_name = data.get("provider", self._config.default_provider)
        model = data.get("model")
        correlation_id = data.get("correlation_id", "")
        
        start_time = time.time()
        
        try:
            messages = json.loads(messages_json) if isinstance(messages_json, str) else messages_json
            
            # Generate response with failover
            full_response = ""
            async for token in self._generate_with_failover(
                messages=messages,
                preferred_provider=provider_name,
                model=model,
            ):
                full_response += token
                self._tokens_generated += 1
                
                # Stream token to TTS immediately
                await self._publish_token(
                    session_id=session_id,
                    token=token,
                    correlation_id=correlation_id,
                )
            
            # Publish completion
            await self._publish_completion(
                session_id=session_id,
                text=full_response,
                correlation_id=correlation_id,
            )
            
            # Acknowledge message
            await self._redis.client.xack(
                STREAM_LLM_REQUESTS,
                GROUP_LLM_WORKERS,
                message_id,
            )
            
            self._requests_total += 1
            duration = time.time() - start_time
            
            logger.info(
                "LLM request completed",
                extra={
                    "session_id": session_id,
                    "response_length": len(full_response),
                    "duration_ms": int(duration * 1000),
                }
            )
            
        except Exception as e:
            self._requests_failed += 1
            logger.error(f"LLM request failed: {e}", extra={"session_id": session_id}, exc_info=True)
            
            await self._publish_error(session_id, str(e), correlation_id)
            
            await self._redis.client.xack(
                STREAM_LLM_REQUESTS,
                GROUP_LLM_WORKERS,
                message_id,
            )

    async def _generate_with_failover(
        self,
        messages: List[Dict[str, str]],
        preferred_provider: str,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Generate response with automatic failover."""
        # Build provider order: preferred first, then others
        providers_to_try = [preferred_provider] + [
            p for p in self._provider_priority if p != preferred_provider
        ]
        
        last_error = None
        
        for provider_name in providers_to_try:
            provider = self._providers.get(provider_name)
            circuit = self._circuit_breakers.get(provider_name)
            
            if not provider or not circuit:
                continue
            
            if not circuit.can_execute():
                logger.debug(f"Circuit open for {provider_name}, skipping")
                continue
            
            try:
                async for token in provider.generate_stream(
                    messages=messages,
                    model=model,
                ):
                    yield token
                
                circuit.record_success()
                return
                
            except Exception as e:
                circuit.record_failure()
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        # All providers failed
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    async def _publish_token(
        self,
        session_id: str,
        token: str,
        correlation_id: str,
    ) -> None:
        """Publish token for streaming to TTS."""
        channel = f"{CHANNEL_LLM_RESPONSE}:{session_id}"
        message = json.dumps({
            "type": "llm.token",
            "session_id": session_id,
            "token": token,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        })
        await self._redis.publish(channel, message)

    async def _publish_completion(
        self,
        session_id: str,
        text: str,
        correlation_id: str,
    ) -> None:
        """Publish completion event."""
        channel = f"{CHANNEL_LLM_RESPONSE}:{session_id}"
        message = json.dumps({
            "type": "llm.completed",
            "session_id": session_id,
            "text": text,
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
        """Publish error event."""
        channel = f"{CHANNEL_LLM_RESPONSE}:{session_id}"
        message = json.dumps({
            "type": "llm.failed",
            "session_id": session_id,
            "error": error,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        })
        await self._redis.publish(channel, message)


async def main() -> None:
    """Main entry point for LLM worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    config = LLMWorkerConfig.from_env()
    worker = LLMWorker(config)
    
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

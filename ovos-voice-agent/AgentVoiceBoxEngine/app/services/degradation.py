"""Graceful degradation service for fault tolerance.

Implements Requirement 16.6:
- Text-only mode when TTS fails
- Echo mode when LLM fails
- Degraded responses with appropriate messaging

Degradation Modes:
| Failure          | Degraded Mode      | User Impact                                    |
|------------------|--------------------|-------------------------------------------------|
| LLM unavailable  | Echo mode          | "I'm having trouble thinking right now..."     |
| TTS unavailable  | Text-only          | Transcript sent without audio                  |
| STT unavailable  | Manual text input  | User must type instead of speak                |
| Redis unavailable| Reject new conn    | Existing sessions continue in-memory           |
| PostgreSQL down  | Skip persistence   | Conversations not saved, core function works   |
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DegradationMode(str, Enum):
    """System degradation modes."""
    
    NORMAL = "normal"
    TTS_DEGRADED = "tts_degraded"
    STT_DEGRADED = "stt_degraded"
    LLM_DEGRADED = "llm_degraded"
    PERSISTENCE_DEGRADED = "persistence_degraded"
    CACHE_DEGRADED = "cache_degraded"
    FULL_DEGRADED = "full_degraded"


class ServiceStatus(str, Enum):
    """Individual service status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class ServiceHealth:
    """Health status of a service."""
    
    name: str
    status: ServiceStatus
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_count: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


@dataclass
class DegradationState:
    """Current degradation state of the system."""
    
    mode: DegradationMode = DegradationMode.NORMAL
    services: Dict[str, ServiceHealth] = field(default_factory=dict)
    active_degradations: List[str] = field(default_factory=list)
    
    def is_degraded(self) -> bool:
        """Check if system is in any degraded state."""
        return self.mode != DegradationMode.NORMAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "is_degraded": self.is_degraded(),
            "services": {k: v.to_dict() for k, v in self.services.items()},
            "active_degradations": self.active_degradations,
        }


class DegradationService:
    """Service for managing graceful degradation.
    
    Tracks service health and provides degraded responses
    when components fail.
    """
    
    # Default degraded messages
    LLM_DEGRADED_MESSAGE = (
        "I'm having trouble thinking right now. "
        "Please try again in a moment."
    )
    TTS_DEGRADED_MESSAGE = (
        "Audio synthesis is temporarily unavailable. "
        "Here's the text response instead."
    )
    STT_DEGRADED_MESSAGE = (
        "Voice recognition is temporarily unavailable. "
        "Please type your message instead."
    )
    
    def __init__(self):
        self._state = DegradationState()
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize service health tracking."""
        services = ["stt", "tts", "llm", "redis", "postgres"]
        for service in services:
            self._state.services[service] = ServiceHealth(
                name=service,
                status=ServiceStatus.HEALTHY,
            )
    
    @property
    def state(self) -> DegradationState:
        """Get current degradation state."""
        return self._state
    
    def report_service_healthy(self, service: str) -> None:
        """Report a service as healthy."""
        if service in self._state.services:
            health = self._state.services[service]
            health.status = ServiceStatus.HEALTHY
            health.last_check = datetime.now(timezone.utc)
            health.error_count = 0
            health.last_error = None
            self._update_degradation_mode()
            logger.info(f"Service {service} reported healthy")
    
    def report_service_error(
        self,
        service: str,
        error: Optional[str] = None,
        threshold: int = 3,
    ) -> None:
        """Report a service error.
        
        Args:
            service: Service name
            error: Error message
            threshold: Number of errors before marking unavailable
        """
        if service not in self._state.services:
            self._state.services[service] = ServiceHealth(
                name=service,
                status=ServiceStatus.HEALTHY,
            )
        
        health = self._state.services[service]
        health.error_count += 1
        health.last_error = error
        health.last_check = datetime.now(timezone.utc)
        
        if health.error_count >= threshold:
            health.status = ServiceStatus.UNAVAILABLE
            logger.warning(f"Service {service} marked unavailable after {threshold} errors")
        elif health.error_count >= threshold // 2:
            health.status = ServiceStatus.DEGRADED
            logger.warning(f"Service {service} marked degraded")
        
        self._update_degradation_mode()
    
    def _update_degradation_mode(self) -> None:
        """Update overall degradation mode based on service health."""
        degradations = []
        
        # Check each service
        for name, health in self._state.services.items():
            if health.status == ServiceStatus.UNAVAILABLE:
                degradations.append(f"{name}_unavailable")
            elif health.status == ServiceStatus.DEGRADED:
                degradations.append(f"{name}_degraded")
        
        self._state.active_degradations = degradations
        
        # Determine overall mode
        if not degradations:
            self._state.mode = DegradationMode.NORMAL
        elif "llm_unavailable" in degradations:
            self._state.mode = DegradationMode.LLM_DEGRADED
        elif "tts_unavailable" in degradations:
            self._state.mode = DegradationMode.TTS_DEGRADED
        elif "stt_unavailable" in degradations:
            self._state.mode = DegradationMode.STT_DEGRADED
        elif "redis_unavailable" in degradations:
            self._state.mode = DegradationMode.CACHE_DEGRADED
        elif "postgres_unavailable" in degradations:
            self._state.mode = DegradationMode.PERSISTENCE_DEGRADED
        elif len(degradations) > 2:
            self._state.mode = DegradationMode.FULL_DEGRADED
        else:
            # Some degradation but not critical
            self._state.mode = DegradationMode.NORMAL
    
    def get_degraded_llm_response(
        self,
        user_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a degraded response when LLM is unavailable.
        
        Returns echo mode response with apology.
        """
        if user_input:
            response_text = f"{self.LLM_DEGRADED_MESSAGE}\n\nYou said: {user_input}"
        else:
            response_text = self.LLM_DEGRADED_MESSAGE
        
        return {
            "type": "response.text.delta",
            "delta": response_text,
            "degraded": True,
            "degradation_reason": "llm_unavailable",
        }
    
    def get_degraded_tts_response(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """Get a degraded response when TTS is unavailable.
        
        Returns text-only response without audio.
        """
        return {
            "type": "response.text.done",
            "text": text,
            "audio": None,
            "degraded": True,
            "degradation_reason": "tts_unavailable",
            "message": self.TTS_DEGRADED_MESSAGE,
        }
    
    def get_degraded_stt_response(self) -> Dict[str, Any]:
        """Get a degraded response when STT is unavailable.
        
        Instructs user to type instead.
        """
        return {
            "type": "input_audio_transcription.failed",
            "degraded": True,
            "degradation_reason": "stt_unavailable",
            "message": self.STT_DEGRADED_MESSAGE,
        }
    
    def should_skip_persistence(self) -> bool:
        """Check if persistence should be skipped."""
        postgres = self._state.services.get("postgres")
        return postgres and postgres.status == ServiceStatus.UNAVAILABLE
    
    def should_reject_new_connections(self) -> bool:
        """Check if new connections should be rejected."""
        redis = self._state.services.get("redis")
        return redis and redis.status == ServiceStatus.UNAVAILABLE
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for monitoring."""
        return {
            "overall_status": "degraded" if self._state.is_degraded() else "healthy",
            "mode": self._state.mode.value,
            "services": {
                name: health.status.value
                for name, health in self._state.services.items()
            },
            "active_degradations": self._state.active_degradations,
        }


# Global degradation service instance
_degradation_service: Optional[DegradationService] = None


def get_degradation_service() -> DegradationService:
    """Get the global degradation service instance."""
    global _degradation_service
    if _degradation_service is None:
        _degradation_service = DegradationService()
    return _degradation_service


def report_healthy(service: str) -> None:
    """Report a service as healthy."""
    get_degradation_service().report_service_healthy(service)


def report_error(service: str, error: Optional[str] = None) -> None:
    """Report a service error."""
    get_degradation_service().report_service_error(service, error)


def is_degraded() -> bool:
    """Check if system is in degraded mode."""
    return get_degradation_service().state.is_degraded()


def get_degradation_mode() -> DegradationMode:
    """Get current degradation mode."""
    return get_degradation_service().state.mode


__all__ = [
    "DegradationMode",
    "ServiceStatus",
    "ServiceHealth",
    "DegradationState",
    "DegradationService",
    "get_degradation_service",
    "report_healthy",
    "report_error",
    "is_degraded",
    "get_degradation_mode",
]

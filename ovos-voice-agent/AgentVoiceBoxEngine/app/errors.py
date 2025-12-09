"""OpenAI-compatible error taxonomy for AgentVoiceBox.

Implements Requirement 16.1:
- OpenAI-compatible error types and codes
- Structured error responses
- Error event format for WebSocket

Error Types (OpenAI compatible):
- invalid_request_error: Malformed request
- authentication_error: Invalid/expired token
- permission_error: Insufficient permissions
- not_found_error: Resource doesn't exist
- rate_limit_error: Quota exceeded
- api_error: Internal server error
- overloaded_error: System at capacity
- timeout_error: Operation timed out
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ErrorType(str, Enum):
    """OpenAI-compatible error types."""
    
    INVALID_REQUEST_ERROR = "invalid_request_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    API_ERROR = "api_error"
    OVERLOADED_ERROR = "overloaded_error"
    TIMEOUT_ERROR = "timeout_error"


class ErrorCode(str, Enum):
    """Specific error codes for detailed error handling."""
    
    # Authentication errors
    MISSING_API_KEY = "missing_api_key"
    INVALID_API_KEY = "invalid_api_key"
    EXPIRED_API_KEY = "expired_api_key"
    MISSING_CLIENT_SECRET = "missing_client_secret"
    INVALID_CLIENT_SECRET = "invalid_client_secret"
    EXPIRED_CLIENT_SECRET = "expired_client_secret"
    
    # Permission errors
    INSUFFICIENT_SCOPE = "insufficient_scope"
    TENANT_SUSPENDED = "tenant_suspended"
    PROJECT_DISABLED = "project_disabled"
    
    # Rate limit errors
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    CONCURRENT_LIMIT_EXCEEDED = "concurrent_limit_exceeded"
    QUOTA_EXCEEDED = "quota_exceeded"
    
    # Request errors
    INVALID_JSON = "invalid_json"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_FIELD_VALUE = "invalid_field_value"
    INVALID_MESSAGE_TYPE = "invalid_message_type"
    INVALID_AUDIO_FORMAT = "invalid_audio_format"
    MESSAGE_TOO_LARGE = "message_too_large"
    
    # Resource errors
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_EXPIRED = "session_expired"
    ITEM_NOT_FOUND = "item_not_found"
    VOICE_NOT_FOUND = "voice_not_found"
    MODEL_NOT_FOUND = "model_not_found"
    
    # Server errors
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    WORKER_UNAVAILABLE = "worker_unavailable"
    DATABASE_ERROR = "database_error"
    REDIS_ERROR = "redis_error"
    
    # Capacity errors
    SERVER_OVERLOADED = "server_overloaded"
    QUEUE_FULL = "queue_full"
    SERVER_SHUTTING_DOWN = "server_shutting_down"
    
    # Timeout errors
    REQUEST_TIMEOUT = "request_timeout"
    STT_TIMEOUT = "stt_timeout"
    TTS_TIMEOUT = "tts_timeout"
    LLM_TIMEOUT = "llm_timeout"


@dataclass
class ErrorDetail:
    """Detailed error information."""
    
    type: ErrorType
    code: ErrorCode
    message: str
    param: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "type": self.type.value,
            "code": self.code.value,
            "message": self.message,
        }
        if self.param:
            result["param"] = self.param
        return result


@dataclass
class ErrorEvent:
    """WebSocket error event (OpenAI Realtime API compatible)."""
    
    type: str = "error"
    error: ErrorDetail = field(default_factory=lambda: ErrorDetail(
        type=ErrorType.API_ERROR,
        code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred",
    ))
    event_id: str = field(default_factory=lambda: f"event_{uuid.uuid4().hex[:16]}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "event_id": self.event_id,
            "error": self.error.to_dict(),
        }


class APIError(Exception):
    """Base exception for API errors."""
    
    def __init__(
        self,
        error_type: ErrorType,
        code: ErrorCode,
        message: str,
        param: Optional[str] = None,
        http_status: int = 500,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.code = code
        self.message = message
        self.param = param
        self.http_status = http_status
    
    def to_error_detail(self) -> ErrorDetail:
        """Convert to ErrorDetail."""
        return ErrorDetail(
            type=self.error_type,
            code=self.code,
            message=self.message,
            param=self.param,
        )
    
    def to_error_event(self) -> ErrorEvent:
        """Convert to ErrorEvent for WebSocket."""
        return ErrorEvent(error=self.to_error_detail())
    
    def to_response(self) -> Dict[str, Any]:
        """Convert to HTTP response body."""
        return {"error": self.to_error_detail().to_dict()}


# Convenience exception classes
class InvalidRequestError(APIError):
    """Invalid request error."""
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.INVALID_FIELD_VALUE,
        message: str = "Invalid request",
        param: Optional[str] = None,
    ):
        super().__init__(
            error_type=ErrorType.INVALID_REQUEST_ERROR,
            code=code,
            message=message,
            param=param,
            http_status=400,
        )


class AuthenticationError(APIError):
    """Authentication error."""
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.INVALID_API_KEY,
        message: str = "Authentication failed",
    ):
        super().__init__(
            error_type=ErrorType.AUTHENTICATION_ERROR,
            code=code,
            message=message,
            http_status=401,
        )


class PermissionError(APIError):
    """Permission error."""
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.INSUFFICIENT_SCOPE,
        message: str = "Permission denied",
    ):
        super().__init__(
            error_type=ErrorType.PERMISSION_ERROR,
            code=code,
            message=message,
            http_status=403,
        )


class NotFoundError(APIError):
    """Resource not found error."""
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.SESSION_NOT_FOUND,
        message: str = "Resource not found",
        param: Optional[str] = None,
    ):
        super().__init__(
            error_type=ErrorType.NOT_FOUND_ERROR,
            code=code,
            message=message,
            param=param,
            http_status=404,
        )


class RateLimitError(APIError):
    """Rate limit error."""
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.RATE_LIMIT_EXCEEDED,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
    ):
        super().__init__(
            error_type=ErrorType.RATE_LIMIT_ERROR,
            code=code,
            message=message,
            http_status=429,
        )
        self.retry_after = retry_after
    
    def to_response(self) -> Dict[str, Any]:
        """Convert to HTTP response body with retry_after."""
        response = super().to_response()
        if self.retry_after:
            response["retry_after"] = self.retry_after
        return response


class InternalError(APIError):
    """Internal server error."""
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        message: str = "Internal server error",
    ):
        super().__init__(
            error_type=ErrorType.API_ERROR,
            code=code,
            message=message,
            http_status=500,
        )


class OverloadedError(APIError):
    """Server overloaded error."""
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.SERVER_OVERLOADED,
        message: str = "Server is overloaded",
        retry_after: Optional[int] = None,
    ):
        super().__init__(
            error_type=ErrorType.OVERLOADED_ERROR,
            code=code,
            message=message,
            http_status=503,
        )
        self.retry_after = retry_after


class TimeoutError(APIError):
    """Timeout error."""
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.REQUEST_TIMEOUT,
        message: str = "Request timed out",
    ):
        super().__init__(
            error_type=ErrorType.TIMEOUT_ERROR,
            code=code,
            message=message,
            http_status=504,
        )


def create_error_event(
    error_type: ErrorType,
    code: ErrorCode,
    message: str,
    param: Optional[str] = None,
) -> ErrorEvent:
    """Create an error event for WebSocket responses."""
    return ErrorEvent(
        error=ErrorDetail(
            type=error_type,
            code=code,
            message=message,
            param=param,
        )
    )


def create_error_response(
    error_type: ErrorType,
    code: ErrorCode,
    message: str,
    param: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an error response for HTTP responses."""
    return {
        "error": ErrorDetail(
            type=error_type,
            code=code,
            message=message,
            param=param,
        ).to_dict()
    }


__all__ = [
    # Enums
    "ErrorType",
    "ErrorCode",
    # Data classes
    "ErrorDetail",
    "ErrorEvent",
    # Base exception
    "APIError",
    # Specific exceptions
    "InvalidRequestError",
    "AuthenticationError",
    "PermissionError",
    "NotFoundError",
    "RateLimitError",
    "InternalError",
    "OverloadedError",
    "TimeoutError",
    # Factory functions
    "create_error_event",
    "create_error_response",
]

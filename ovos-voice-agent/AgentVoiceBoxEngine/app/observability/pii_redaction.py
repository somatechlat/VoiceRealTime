"""PII redaction for logs and audit trails.

Implements Requirements 15.3, 14.4:
- Redact or hash sensitive fields in logs
- Protect transcripts, user identifiers, API keys
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class PIIPatterns:
    """Regex patterns for PII detection."""

    EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    PHONE = re.compile(r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}")
    API_KEY = re.compile(r"avb_[a-zA-Z0-9]{8}_[a-zA-Z0-9_-]{10,}")
    EPH_TOKEN = re.compile(r"eph_[a-zA-Z0-9_-]{18,}")
    BEARER = re.compile(r"Bearer\s+[a-zA-Z0-9._-]+", re.IGNORECASE)
    CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    SSN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")


class PIIRedactor:
    """Redacts PII from strings and dictionaries."""

    MASK = "[REDACTED]"
    SENSITIVE_FIELDS: Set[str] = {
        "password", "secret", "token", "api_key", "apikey",
        "bearer", "authorization", "credit_card", "ssn",
        "transcript", "content", "audio", "audio_data",
    }
    PARTIAL_FIELDS: Set[str] = {"email", "phone", "ip_address"}

    def __init__(self, redact_transcripts: bool = True, redact_ips: bool = False):
        self.redact_transcripts = redact_transcripts
        self.redact_ips = redact_ips

    def _partial_email(self, email: str) -> str:
        parts = email.split("@")
        return f"[REDACTED]@{parts[1]}" if len(parts) == 2 else self.MASK

    def _partial_phone(self, phone: str) -> str:
        digits = re.sub(r"\D", "", phone)
        return f"[PHONE:***{digits[-4:]}]" if len(digits) >= 4 else self.MASK

    def redact_string(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        result = text
        result = PIIPatterns.API_KEY.sub(lambda m: f"avb_{m.group(0).split('_')[1]}_***", result)
        result = PIIPatterns.EPH_TOKEN.sub("eph_***", result)
        result = PIIPatterns.BEARER.sub("Bearer ***", result)
        result = PIIPatterns.EMAIL.sub(lambda m: self._partial_email(m.group(0)), result)
        result = PIIPatterns.PHONE.sub(lambda m: self._partial_phone(m.group(0)), result)
        result = PIIPatterns.CREDIT_CARD.sub("[CARD:****]", result)
        result = PIIPatterns.SSN.sub("[SSN:***]", result)
        return result

    def redact_dict(self, data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        if depth > 10:
            return {"_truncated": True}
        result = {}
        for key, value in data.items():
            key_lower = key.lower()
            if key_lower in self.SENSITIVE_FIELDS:
                if key_lower in ("transcript", "content") and not self.redact_transcripts:
                    result[key] = value
                else:
                    result[key] = self.MASK
            elif key_lower in self.PARTIAL_FIELDS:
                if key_lower == "email" and isinstance(value, str):
                    result[key] = self._partial_email(value)
                elif key_lower == "phone" and isinstance(value, str):
                    result[key] = self._partial_phone(value)
                elif key_lower == "ip_address" and self.redact_ips:
                    result[key] = self.MASK
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value, depth + 1)
            elif isinstance(value, str):
                result[key] = self.redact_string(value)
            else:
                result[key] = value
        return result


class PIIRedactingFilter(logging.Filter):
    """Logging filter that redacts PII from log records."""

    def __init__(self, name: str = "", redact_transcripts: bool = True):
        super().__init__(name)
        self.redactor = PIIRedactor(redact_transcripts=redact_transcripts)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg and isinstance(record.msg, str):
            record.msg = self.redactor.redact_string(record.msg)
        return True


_redactor: Optional[PIIRedactor] = None


def get_redactor() -> PIIRedactor:
    global _redactor
    if _redactor is None:
        _redactor = PIIRedactor()
    return _redactor


def redact(data: Any) -> Any:
    r = get_redactor()
    if isinstance(data, str):
        return r.redact_string(data)
    elif isinstance(data, dict):
        return r.redact_dict(data)
    return data


__all__ = ["PIIPatterns", "PIIRedactor", "PIIRedactingFilter", "get_redactor", "redact"]

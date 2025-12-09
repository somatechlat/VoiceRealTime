"""PII redaction for logs and audit trails.

Implements Requirements 15.3, 14.4:
- Redact or hash sensitive fields in logs
- Protect transcripts, user identifiers, API keys

Patterns redacted:
- Email addresses
- Phone numbers
- API keys (avb_*, eph_*, Bearer tokens)
- IP addresses (optionally)
- Credit card numbers
- SSN patterns
- Transcript content
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PIIRedactor:
    """PII redaction utility for logs and data.

    Supports multiple redaction strategies:
    - MASK: Replace with [REDACTED]
    - HASH: Replace with SHA-256 hash prefix
    - PARTIAL: Show partial value (e.g., email domain)
    """

    # Regex patterns for PII detection
    PATTERNS = {
        "email": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            re.IGNORECASE,
        ),
        "phone": re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
        ),
        "api_key": re.compile(r"\b(avb_[a-zA-Z0-9_-]+|eph_[a-zA-Z0-9_-]+)\b"),
        "bearer_token": re.compile(r"Bearer\s+[a-zA-Z0-9._-]+", re.IGNORECASE),
        "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }

    # Fields that should always be redacted
    SENSITIVE_FIELDS: Set[str] = {
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "credential",
        "transcript",
        "content",
        "audio",
        "ssn",
        "credit_card",
        "card_number",
    }

    def __init__(
        self,
        redact_emails: bool = True,
        redact_phones: bool = True,
        redact_ips: bool = False,
        hash_instead_of_mask: bool = False,
    ):
        self.redact_emails = redact_emails
        self.redact_phones = redact_phones
        self.redact_ips = redact_ips
        self.hash_instead_of_mask = hash_instead_of_mask

    def _hash_value(self, value: str) -> str:
        """Generate a short hash for a value."""
        hash_full = hashlib.sha256(value.encode()).hexdigest()
        return f"[HASH:{hash_full[:8]}]"

    def _mask_value(self, value: str, pattern_name: str) -> str:
        """Mask a value based on pattern type."""
        if self.hash_instead_of_mask:
            return self._hash_value(value)

        if pattern_name == "email":
            # Show domain only: user@domain.com -> [REDACTED]@domain.com
            parts = value.split("@")
            if len(parts) == 2:
                return f"[REDACTED]@{parts[1]}"
        elif pattern_name == "api_key":
            # Show prefix only: avb_abc123_xyz -> avb_abc1****
            if "_" in value:
                prefix = value.split("_")[0]
                return f"{prefix}_****"

        return "[REDACTED]"

    def redact_string(self, text: str) -> str:
        """Redact PII from a string.

        Args:
            text: Input string

        Returns:
            String with PII redacted
        """
        if not text:
            return text

        result = text

        # Always redact API keys and bearer tokens
        result = self.PATTERNS["api_key"].sub(
            lambda m: self._mask_value(m.group(), "api_key"), result
        )
        result = self.PATTERNS["bearer_token"].sub("[REDACTED_TOKEN]", result)
        result = self.PATTERNS["credit_card"].sub("[REDACTED_CC]", result)
        result = self.PATTERNS["ssn"].sub("[REDACTED_SSN]", result)

        if self.redact_emails:
            result = self.PATTERNS["email"].sub(
                lambda m: self._mask_value(m.group(), "email"), result
            )

        if self.redact_phones:
            result = self.PATTERNS["phone"].sub("[REDACTED_PHONE]", result)

        if self.redact_ips:
            result = self.PATTERNS["ipv4"].sub("[REDACTED_IP]", result)

        return result

    def redact_dict(
        self,
        data: Dict[str, Any],
        depth: int = 0,
        max_depth: int = 10,
    ) -> Dict[str, Any]:
        """Redact PII from a dictionary recursively.

        Args:
            data: Input dictionary
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            Dictionary with PII redacted
        """
        if depth > max_depth:
            return {"_truncated": True}

        result = {}
        for key, value in data.items():
            key_lower = key.lower()

            # Check if field name indicates sensitive data
            if any(sensitive in key_lower for sensitive in self.SENSITIVE_FIELDS):
                if isinstance(value, str):
                    result[key] = self._mask_value(value, "sensitive_field")
                elif value is not None:
                    result[key] = "[REDACTED]"
                else:
                    result[key] = None
            elif isinstance(value, str):
                result[key] = self.redact_string(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value, depth + 1, max_depth)
            elif isinstance(value, list):
                result[key] = [
                    self.redact_dict(item, depth + 1, max_depth)
                    if isinstance(item, dict)
                    else self.redact_string(item)
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        return result


class PIIRedactionFilter(logging.Filter):
    """Logging filter that redacts PII from log records.

    Attach to handlers to automatically redact PII from all logs.
    """

    def __init__(self, name: str = "", redactor: Optional[PIIRedactor] = None):
        super().__init__(name)
        self.redactor = redactor or PIIRedactor()

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact PII from log record."""
        # Redact the message
        if isinstance(record.msg, str):
            record.msg = self.redactor.redact_string(record.msg)

        # Redact args if present
        if record.args:
            if isinstance(record.args, dict):
                record.args = self.redactor.redact_dict(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self.redactor.redact_string(arg)
                    if isinstance(arg, str)
                    else arg
                    for arg in record.args
                )

        # Redact extra fields
        if hasattr(record, "__dict__"):
            for key in list(record.__dict__.keys()):
                if key.startswith("_") or key in logging.LogRecord.__dict__:
                    continue
                value = getattr(record, key)
                if isinstance(value, str):
                    setattr(record, key, self.redactor.redact_string(value))
                elif isinstance(value, dict):
                    setattr(record, key, self.redactor.redact_dict(value))

        return True


# Global redactor instance
_default_redactor = PIIRedactor()


def redact_pii(data: Any) -> Any:
    """Convenience function to redact PII from any data.

    Args:
        data: String, dict, or other data

    Returns:
        Data with PII redacted
    """
    if isinstance(data, str):
        return _default_redactor.redact_string(data)
    elif isinstance(data, dict):
        return _default_redactor.redact_dict(data)
    return data


__all__ = [
    "PIIRedactor",
    "PIIRedactionFilter",
    "redact_pii",
]

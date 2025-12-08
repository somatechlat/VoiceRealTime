"""In-memory token store for realtime client secrets."""

from __future__ import annotations

import datetime as dt
import secrets
import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ClientSecretRecord:
    secret: str
    project_id: Optional[str]
    session_id: str
    session_config: Dict
    expires_at: dt.datetime

    def is_expired(self, now: Optional[dt.datetime] = None) -> bool:
        current = now or dt.datetime.utcnow()
        return current >= self.expires_at


class TokenService:
    def __init__(self):
        self._records: Dict[str, ClientSecretRecord] = {}
        self._lock = threading.Lock()

    @staticmethod
    def generate_secret() -> str:
        return f"ek_{secrets.token_urlsafe(20)}"

    def issue(
        self,
        project_id: Optional[str],
        session_id: str,
        session_config: Dict,
        ttl_seconds: int,
    ) -> ClientSecretRecord:
        secret_value = self.generate_secret()
        expires_at = dt.datetime.utcnow() + dt.timedelta(seconds=ttl_seconds)
        record = ClientSecretRecord(
            secret=secret_value,
            project_id=project_id,
            session_id=session_id,
            session_config=session_config,
            expires_at=expires_at,
        )
        with self._lock:
            self._records[secret_value] = record
        return record

    def get(self, secret: str) -> Optional[ClientSecretRecord]:
        with self._lock:
            record = self._records.get(secret)
        if not record:
            return None
        if record.is_expired():
            self.revoke(secret)
            return None
        return record

    def revoke(self, secret: str) -> None:
        with self._lock:
            self._records.pop(secret, None)

    def cleanup(self) -> None:
        now = dt.datetime.utcnow()
        with self._lock:
            expired = [key for key, record in self._records.items() if record.is_expired(now)]
            for key in expired:
                self._records.pop(key, None)


__all__ = ["TokenService", "ClientSecretRecord"]

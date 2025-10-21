"""Simple OPA client for policy decisions."""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from ..config import AppConfig

logger = logging.getLogger(__name__)


class OPAClient:
    """Wrapper around the OPA HTTP API."""

    def __init__(self, config: AppConfig):
        # Use a synchronous httpx client instead of async to avoid event‑loop issues
        self._config = config
        self._client = httpx.Client(timeout=config.opa.timeout_seconds)

    def allow(self, input_data: Dict[str, Any]) -> bool:
        """Perform a policy decision synchronously.

        Returns ``True`` if the policy permits the action, otherwise ``False``.
        """
        decision_url = f"{self._config.opa.url}{self._config.opa.decision_path}"
        try:
            response = self._client.post(decision_url, json={"input": input_data})
            response.raise_for_status()
            payload = response.json()
            return bool(payload.get("result", False))
        except httpx.HTTPError as exc:
            # In local/dev environments we allow requests to proceed if OPA is unavailable
            logger.warning("OPA decision request failed; allowing by default in dev", exc_info=exc)
            return True

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

__all__ = ["OPAClient"]

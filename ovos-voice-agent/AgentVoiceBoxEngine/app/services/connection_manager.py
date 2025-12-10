"""Connection manager for tracking active WebSocket connections and graceful shutdown.

This module provides:
- Tracking of active WebSocket connections
- Graceful shutdown with connection draining
- SIGTERM signal handling for Kubernetes deployments
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about an active connection."""

    session_id: str
    tenant_id: str
    connected_at: float
    websocket: object  # The actual WebSocket object


class ConnectionManager:
    """Manages active WebSocket connections for graceful shutdown.

    Features:
    - Track all active connections
    - Graceful shutdown with configurable drain period
    - SIGTERM/SIGINT signal handling
    - Thread-safe operations
    """

    def __init__(self, drain_timeout_seconds: int = 30) -> None:
        self._connections: Dict[str, ConnectionInfo] = {}
        self._lock = threading.Lock()
        self._shutting_down = False
        self._drain_timeout = drain_timeout_seconds
        self._shutdown_event = threading.Event()
        self._shutdown_callbacks: list[Callable[[], None]] = []

    @property
    def is_shutting_down(self) -> bool:
        """Check if the server is in shutdown mode."""
        return self._shutting_down

    @property
    def active_connection_count(self) -> int:
        """Get the number of active connections."""
        with self._lock:
            return len(self._connections)

    def register_connection(
        self,
        session_id: str,
        tenant_id: str,
        websocket: object,
    ) -> None:
        """Register a new WebSocket connection."""
        with self._lock:
            self._connections[session_id] = ConnectionInfo(
                session_id=session_id,
                tenant_id=tenant_id,
                connected_at=time.time(),
                websocket=websocket,
            )
        logger.debug(
            "Connection registered",
            extra={"session_id": session_id, "total_connections": self.active_connection_count},
        )

    def unregister_connection(self, session_id: str) -> Optional[ConnectionInfo]:
        """Unregister a WebSocket connection."""
        with self._lock:
            info = self._connections.pop(session_id, None)
        if info:
            logger.debug(
                "Connection unregistered",
                extra={"session_id": session_id, "total_connections": self.active_connection_count},
            )
        return info

    def get_connection(self, session_id: str) -> Optional[ConnectionInfo]:
        """Get connection info by session ID."""
        with self._lock:
            return self._connections.get(session_id)

    def get_all_connections(self) -> list[ConnectionInfo]:
        """Get all active connections."""
        with self._lock:
            return list(self._connections.values())

    def add_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to be called during shutdown."""
        self._shutdown_callbacks.append(callback)

    def initiate_shutdown(self) -> None:
        """Initiate graceful shutdown.

        This method:
        1. Sets shutting_down flag (new connections should be rejected)
        2. Waits for drain_timeout for connections to close naturally
        3. Forcefully closes remaining connections
        4. Calls registered shutdown callbacks
        """
        if self._shutting_down:
            return

        self._shutting_down = True
        logger.info(
            "Initiating graceful shutdown",
            extra={
                "drain_timeout_seconds": self._drain_timeout,
                "active_connections": self.active_connection_count,
            },
        )

        # Start drain period
        drain_start = time.time()
        drain_end = drain_start + self._drain_timeout

        # Wait for connections to drain naturally
        while time.time() < drain_end:
            count = self.active_connection_count
            if count == 0:
                logger.info("All connections drained successfully")
                break

            remaining = int(drain_end - time.time())
            logger.info(f"Draining connections: {count} remaining, {remaining}s left")
            time.sleep(1)

        # Force close any remaining connections
        remaining_connections = self.get_all_connections()
        if remaining_connections:
            logger.warning(
                f"Force closing {len(remaining_connections)} connections after drain timeout"
            )
            for conn in remaining_connections:
                try:
                    # Try to close the WebSocket gracefully
                    if hasattr(conn.websocket, "close"):
                        conn.websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing connection {conn.session_id}: {e}")

        # Call shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Shutdown callback error: {e}")

        self._shutdown_event.set()
        logger.info("Graceful shutdown complete")

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to complete."""
        return self._shutdown_event.wait(timeout)


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


def init_connection_manager(drain_timeout_seconds: int = 30) -> ConnectionManager:
    """Initialize the global connection manager."""
    global _connection_manager
    _connection_manager = ConnectionManager(drain_timeout_seconds)
    return _connection_manager


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown.

    Handles:
    - SIGTERM: Kubernetes sends this before killing the pod
    - SIGINT: Ctrl+C in development
    """
    manager = get_connection_manager()

    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown")

        # Run shutdown in a separate thread to not block signal handler
        shutdown_thread = threading.Thread(
            target=manager.initiate_shutdown,
            daemon=False,
        )
        shutdown_thread.start()

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Signal handlers registered for graceful shutdown")


__all__ = [
    "ConnectionManager",
    "ConnectionInfo",
    "get_connection_manager",
    "init_connection_manager",
    "setup_signal_handlers",
]

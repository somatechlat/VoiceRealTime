"""Session management service backed by SQLAlchemy."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Optional

from ..models.session import ConversationItem, SessionModel
from ..observability.metrics import active_sessions, session_starts
from ..utils.database import session_scope

logger = logging.getLogger(__name__)


class SessionService:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def create_session(
        self,
        session_id: str,
        project_id: Optional[str],
        session_payload: Dict[str, Any],
        expires_at: Optional[dt.datetime],
        persona: Optional[Dict[str, Any]] = None,
    ) -> SessionModel:
        session_starts.inc()
        with session_scope(self._session_factory) as session:
            model = SessionModel(
                id=session_id,
                project_id=project_id,
                persona=persona or {},
                session_config=session_payload,
                model=session_payload.get("model"),
                instructions=session_payload.get("instructions"),
                output_modalities=session_payload.get("output_modalities"),
                tools=session_payload.get("tools"),
                tool_choice=session_payload.get("tool_choice"),
                audio_config=session_payload.get("audio"),
                max_output_tokens=(
                    str(session_payload.get("max_output_tokens"))
                    if session_payload.get("max_output_tokens") is not None
                    else None
                ),
                expires_at=expires_at,
                status="active",
                created_at=dt.datetime.utcnow(),
            )
            session.add(model)
            session.commit()
        active_sessions.inc()
        logger.info("Session created", extra={"session_id": session_id})
        return model

    def close_session(self, session_id: str) -> None:
        with session_scope(self._session_factory) as session:
            model = session.get(SessionModel, session_id)
            if not model:
                logger.warning(
                    "Attempted to close missing session", extra={"session_id": session_id}
                )
                return
            model.status = "closed"
            model.closed_at = dt.datetime.utcnow()
            session.add(model)
            session.commit()
        active_sessions.dec()
        logger.info("Session closed", extra={"session_id": session_id})

    def append_conversation_item(self, session_id: str, item: Dict[str, Any]) -> ConversationItem:
        with session_scope(self._session_factory) as session:
            record = ConversationItem(
                session_id=session_id,
                role=item.get("role"),
                content=item,
                created_at=dt.datetime.utcnow(),
            )
            session.add(record)
            session.commit()
            return record

    def get_session(self, session_id: str) -> Optional[SessionModel]:
        with session_scope(self._session_factory) as session:
            return session.get(SessionModel, session_id)

    def update_session(
        self,
        session_id: str,
        *,
        session_updates: Optional[Dict[str, Any]] = None,
        persona: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> Optional[SessionModel]:
        with session_scope(self._session_factory) as session:
            model = session.get(SessionModel, session_id)
            if not model:
                logger.warning(
                    "Attempted to update missing session", extra={"session_id": session_id}
                )
                return None

            if session_updates:
                merged = dict(model.session_config or {})
                merged.update(session_updates)
                model.session_config = merged

                if "model" in session_updates:
                    model.model = session_updates["model"]
                if "instructions" in session_updates:
                    model.instructions = session_updates["instructions"]
                if "output_modalities" in session_updates:
                    model.output_modalities = session_updates["output_modalities"]
                if "tools" in session_updates:
                    model.tools = session_updates["tools"]
                if "tool_choice" in session_updates:
                    model.tool_choice = session_updates["tool_choice"]
                if "audio" in session_updates:
                    model.audio_config = session_updates["audio"]
                if "max_output_tokens" in session_updates:
                    max_value = session_updates["max_output_tokens"]
                    if isinstance(max_value, str) and max_value.lower() == "inf":
                        model.max_output_tokens = "inf"
                    elif max_value is None:
                        model.max_output_tokens = None
                    else:
                        model.max_output_tokens = str(max_value)

            if persona is not None:
                model.persona = persona

            if status is not None:
                model.status = status

            session.add(model)
            session.commit()
            session.refresh(model)
            logger.info("Session updated", extra={"session_id": session_id})
            return model


__all__ = ["SessionService"]

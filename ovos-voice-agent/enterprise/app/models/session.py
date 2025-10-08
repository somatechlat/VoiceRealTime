"""Session and conversation item models."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

JSONField = JSONB(astext_type=Text()).with_variant(JSON(), "sqlite")


class SessionModel(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    persona: Mapped[Dict[str, Any]] = mapped_column(JSONField)
    session_config: Mapped[Dict[str, Any]] = mapped_column(JSONField, default=dict)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_modalities: Mapped[Dict[str, Any] | None] = mapped_column(JSONField, nullable=True)
    tools: Mapped[Dict[str, Any] | None] = mapped_column(JSONField, nullable=True)
    tool_choice: Mapped[Dict[str, Any] | None] = mapped_column(JSONField, nullable=True)
    audio_config: Mapped[Dict[str, Any] | None] = mapped_column(JSONField, nullable=True)
    max_output_tokens: Mapped[str | None] = mapped_column(String(32), nullable=True)
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=False), default=dt.datetime.utcnow
    )
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)

    conversation_items: Mapped[list["ConversationItem"]] = relationship(
        "ConversationItem", back_populates="session"
    )


class ConversationItem(Base):
    __tablename__ = "conversation_items"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id"))
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[Dict[str, Any]] = mapped_column(JSONField)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=False), default=dt.datetime.utcnow
    )

    session: Mapped[SessionModel] = relationship(
        "SessionModel", back_populates="conversation_items"
    )


__all__ = ["SessionModel", "ConversationItem"]

"""Answer-level score model."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class AnswerScore(Base):
    __tablename__ = "answer_scores"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    interview_id: Mapped[str] = mapped_column(String(36), ForeignKey("interviews.id", ondelete="CASCADE"), index=True)
    transcript_turn_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("transcript_turns.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    score_payload: Mapped[dict] = mapped_column(JSON)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    rubric_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    interview = relationship("Interview", back_populates="answer_scores")
    transcript_turn = relationship("TranscriptTurn")

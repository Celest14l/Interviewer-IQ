"""Interview model."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Interview(Base):
    __tablename__ = "interviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(255))
    persona: Mapped[str] = mapped_column(String(100), default="friendly_hr")
    status: Mapped[str] = mapped_column(String(50), default="draft")
    resume_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_resume: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="interviews")
    blueprint = relationship("InterviewBlueprint", back_populates="interview", uselist=False, cascade="all, delete-orphan")
    transcript_turns = relationship("TranscriptTurn", back_populates="interview", cascade="all, delete-orphan")
    answer_scores = relationship("AnswerScore", back_populates="interview", cascade="all, delete-orphan")
    snapshot_events = relationship("SnapshotEvent", back_populates="interview", cascade="all, delete-orphan")
    final_report = relationship("FinalReport", back_populates="interview", uselist=False, cascade="all, delete-orphan")
    uploaded_files = relationship("UploadedFile", back_populates="interview")

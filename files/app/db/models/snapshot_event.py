"""Snapshot-based face analysis event model."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class SnapshotEvent(Base):
    __tablename__ = "snapshot_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    interview_id: Mapped[str] = mapped_column(String(36), ForeignKey("interviews.id", ondelete="CASCADE"), index=True)
    image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    analysis_payload: Mapped[dict] = mapped_column(JSON)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    interview = relationship("Interview", back_populates="snapshot_events")

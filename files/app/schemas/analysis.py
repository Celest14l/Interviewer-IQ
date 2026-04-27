"""Schemas for face snapshot analysis and coaching metadata."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SnapshotAnalysisRequest(BaseModel):
    """Payload for analyzing one face snapshot image."""

    interview_id: str
    image: str


class SnapshotAnalysisResponse(BaseModel):
    """Realtime face-analysis response returned to the frontend."""

    emotion: str
    emotion_confidence: float
    gaze_score: float
    posture_score: float
    feedback: str
    details: dict = Field(default_factory=dict)
    ts: float | None = None
    snapshot_event_id: str | None = None


class SnapshotEventResponse(BaseModel):
    """Persisted snapshot event exposed via the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    interview_id: str
    image_path: str | None = None
    analysis_payload: dict
    model_name: str | None = None
    created_at: datetime


class SnapshotSummaryResponse(BaseModel):
    """Collection response with aggregated coaching signals."""

    interview_id: str
    count: int
    snapshots: list[SnapshotEventResponse] = Field(default_factory=list)
    average_scores: dict = Field(default_factory=dict)

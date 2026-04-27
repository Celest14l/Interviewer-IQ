"""Interview and realtime session schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class InterviewCreate(BaseModel):
    role: str = Field(min_length=1, max_length=255)
    persona: str = Field(default="friendly_hr", min_length=1, max_length=100)


class InterviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    role: str
    persona: str
    status: str
    created_at: datetime
    updated_at: datetime


class TranscriptTurnResponse(BaseModel):
    """A single persisted transcript turn exposed to the client."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    turn_index: int
    role: str
    content: str
    metadata_json: dict | None = Field(default=None, serialization_alias="metadata")
    created_at: datetime


class RealtimeSessionState(BaseModel):
    """Session state sent to the client when a websocket session starts."""

    interview: InterviewResponse
    current_topic: str | None = None
    transcript: list[TranscriptTurnResponse] = Field(default_factory=list)


class ClientRealtimeEvent(BaseModel):
    """Incoming websocket event from the browser client."""

    type: str
    content: str | None = None
    audio_base64: str | None = None
    filename: str | None = None
    content_type: str | None = None
    language: str | None = None
    prompt: str | None = None


class ServerRealtimeEvent(BaseModel):
    """Outgoing websocket event emitted by the realtime interview loop."""

    type: str
    message: str | None = None
    session: RealtimeSessionState | None = None
    turn: TranscriptTurnResponse | None = None
    metadata: dict | None = None

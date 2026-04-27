"""Schemas for speech-to-text and text-to-speech endpoints."""

from pydantic import BaseModel, Field


class AudioTranscriptionResponse(BaseModel):
    """Response returned after transcribing an uploaded audio file."""

    text: str
    language: str | None = None
    duration_seconds: float | None = None
    file_id: str | None = None
    interview_id: str | None = None
    provider: str = "groq"
    model: str


class AudioSynthesisRequest(BaseModel):
    """Request body for synthesizing a line of interviewer speech."""

    transcript: str = Field(min_length=1, max_length=4000)
    interview_id: str | None = None
    language: str | None = None


class AudioSynthesisResponse(BaseModel):
    """Metadata returned after generating interviewer audio."""

    file_id: str
    interview_id: str | None = None
    provider: str = "cartesia"
    model: str
    voice_id: str
    content_type: str
    download_url: str

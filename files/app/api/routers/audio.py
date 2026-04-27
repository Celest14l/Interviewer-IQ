"""Audio endpoints for STT uploads, TTS generation, and file download."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.db.models.user import User
from app.db.session import get_db
from app.schemas.audio import AudioSynthesisRequest, AudioSynthesisResponse, AudioTranscriptionResponse
from app.services.audio_service import (
    get_audio_file_for_user,
    synthesize_interviewer_audio,
    transcribe_uploaded_audio,
)

router = APIRouter(prefix="/audio", tags=["Audio"])


@router.post("/transcribe", response_model=AudioTranscriptionResponse, status_code=status.HTTP_201_CREATED)
async def transcribe_audio_endpoint(
    audio: UploadFile = File(...),
    interview_id: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AudioTranscriptionResponse:
    """Persist an uploaded audio clip and return the Groq transcript."""

    return await transcribe_uploaded_audio(
        db,
        user=current_user,
        audio=audio,
        interview_id=interview_id,
        language=language,
        prompt=prompt,
    )


@router.post("/synthesize", response_model=AudioSynthesisResponse, status_code=status.HTTP_201_CREATED)
async def synthesize_audio_endpoint(
    payload: AudioSynthesisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AudioSynthesisResponse:
    """Generate interviewer speech using Cartesia and return the saved artifact info."""

    return await synthesize_interviewer_audio(
        db,
        user=current_user,
        transcript=payload.transcript,
        interview_id=payload.interview_id,
        language=payload.language,
    )


@router.get("/files/{file_id}")
def download_audio_file_endpoint(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    """Download a stored audio artifact owned by the current user."""

    record = get_audio_file_for_user(db, user=current_user, file_id=file_id)
    path = Path(record.storage_path)
    return FileResponse(
        path=path,
        media_type=record.content_type or "application/octet-stream",
        filename=record.original_filename,
    )

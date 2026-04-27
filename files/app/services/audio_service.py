"""Deterministic services for STT/TTS and persisted audio artifacts."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.db.models.uploaded_file import UploadedFile
from app.db.models.user import User
from app.providers.cartesia_tts import synthesize_speech
from app.providers.groq_stt import transcribe_audio
from app.schemas.audio import AudioSynthesisResponse, AudioTranscriptionResponse
from app.services.file_storage import save_binary_payload
from app.services.interview_service import get_user_interview


def _persist_uploaded_file(
    db: Session,
    *,
    user_id: str,
    interview_id: str | None,
    original_filename: str,
    stored_filename: str,
    content_type: str | None,
    size_bytes: int,
    storage_path: str,
) -> UploadedFile:
    record = UploadedFile(
        user_id=user_id,
        interview_id=interview_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        content_type=content_type,
        size_bytes=size_bytes,
        storage_path=storage_path,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


async def transcribe_audio_bytes(
    db: Session,
    *,
    user: User,
    audio_bytes: bytes,
    filename: str,
    content_type: str | None = None,
    interview_id: str | None = None,
    language: str | None = None,
    prompt: str | None = None,
) -> AudioTranscriptionResponse:
    """Persist raw audio bytes and send them to Groq for transcription."""

    settings = get_settings()
    if interview_id:
        interview = get_user_interview(db, user, interview_id)
        if interview is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")

    suffix = Path(filename or "audio.webm").suffix or ".webm"
    stored_filename, storage_path = save_binary_payload(
        payload=audio_bytes,
        namespace=f"{user.id}/audio",
        suffix=suffix,
    )
    provider_response = await transcribe_audio(
        file_bytes=audio_bytes,
        filename=filename or f"audio{suffix}",
        content_type=content_type,
        language=language,
        prompt=prompt,
    )
    file_record = _persist_uploaded_file(
        db,
        user_id=user.id,
        interview_id=interview_id,
        original_filename=filename or f"audio{suffix}",
        stored_filename=stored_filename,
        content_type=content_type,
        size_bytes=len(audio_bytes),
        storage_path=storage_path,
    )
    return AudioTranscriptionResponse(
        text=provider_response["text"],
        language=provider_response.get("language"),
        duration_seconds=provider_response.get("duration"),
        file_id=file_record.id,
        interview_id=interview_id,
        model=settings.groq_stt_model,
    )


async def transcribe_uploaded_audio(
    db: Session,
    *,
    user: User,
    audio: UploadFile,
    interview_id: str | None = None,
    language: str | None = None,
    prompt: str | None = None,
) -> AudioTranscriptionResponse:
    """Persist an uploaded audio file and send it to Groq for transcription."""

    raw_bytes = await audio.read()
    # Reuse the shared path so REST uploads and websocket audio stay identical.
    return await transcribe_audio_bytes(
        db,
        user=user,
        audio_bytes=raw_bytes,
        filename=audio.filename or "audio.webm",
        content_type=audio.content_type,
        interview_id=interview_id,
        language=language,
        prompt=prompt,
    )


async def synthesize_interviewer_audio(
    db: Session,
    *,
    user: User,
    transcript: str,
    interview_id: str | None = None,
    language: str | None = None,
) -> AudioSynthesisResponse:
    """Generate interviewer audio, persist it, and expose a download URL."""

    settings = get_settings()
    if interview_id:
        interview = get_user_interview(db, user, interview_id)
        if interview is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")

    audio_bytes, content_type = await synthesize_speech(
        transcript=transcript,
        language=language,
    )
    suffix = ".wav"
    stored_filename, storage_path = save_binary_payload(
        payload=audio_bytes,
        namespace=f"{user.id}/tts",
        suffix=suffix,
    )
    file_record = _persist_uploaded_file(
        db,
        user_id=user.id,
        interview_id=interview_id,
        original_filename="interviewer_reply.wav",
        stored_filename=stored_filename,
        content_type=content_type,
        size_bytes=len(audio_bytes),
        storage_path=storage_path,
    )
    return AudioSynthesisResponse(
        file_id=file_record.id,
        interview_id=interview_id,
        model=settings.cartesia_model_id,
        voice_id=settings.cartesia_voice_id,
        content_type=content_type,
        download_url=f"/api/audio/files/{file_record.id}",
    )


def get_audio_file_for_user(db: Session, *, user: User, file_id: str) -> UploadedFile:
    """Load a generated or uploaded audio artifact owned by the current user."""

    record = db.get(UploadedFile, file_id)
    if record is None or record.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found")
    return record

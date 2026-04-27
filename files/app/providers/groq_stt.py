"""Groq speech-to-text adapter."""

from __future__ import annotations

import httpx

from app.core.settings import get_settings

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


async def transcribe_audio(
    *,
    file_bytes: bytes,
    filename: str,
    content_type: str | None = None,
    language: str | None = None,
    prompt: str | None = None,
) -> dict:
    """Send an audio file to Groq STT and return the parsed JSON response."""

    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("Groq API key is not configured")

    data = {
        "model": settings.groq_stt_model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    files = {
        "file": (
            filename,
            file_bytes,
            content_type or "application/octet-stream",
        )
    }
    headers = {"Authorization": f"Bearer {settings.groq_api_key}"}

    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            GROQ_TRANSCRIPTION_URL,
            headers=headers,
            data=data,
            files=files,
        )
        response.raise_for_status()
        return response.json()

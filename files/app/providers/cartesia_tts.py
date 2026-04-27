"""Cartesia text-to-speech adapter."""

from __future__ import annotations

import httpx

from app.core.settings import get_settings

CARTESIA_TTS_BYTES_URL = "https://api.cartesia.ai/tts/bytes"


async def synthesize_speech(
    *,
    transcript: str,
    language: str | None = None,
) -> tuple[bytes, str]:
    """Generate WAV audio from interviewer text using Cartesia."""

    settings = get_settings()
    if not settings.cartesia_api_key:
        raise RuntimeError("Cartesia API key is not configured")

    payload = {
        "model_id": settings.cartesia_model_id,
        "transcript": transcript,
        "voice": {
            "mode": "id",
            "id": settings.cartesia_voice_id,
        },
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 44100,
        },
        "language": language or settings.cartesia_language,
    }
    headers = {
        "Authorization": f"Bearer {settings.cartesia_api_key}",
        "Cartesia-Version": settings.cartesia_version,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(CARTESIA_TTS_BYTES_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "audio/wav")

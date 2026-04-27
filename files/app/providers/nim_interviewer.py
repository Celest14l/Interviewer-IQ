"""NVIDIA NIM adapter for rendering interviewer phrasing in realtime."""

from __future__ import annotations

import httpx

from app.core.settings import get_settings

NIM_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


async def render_interviewer_reply(
    *,
    role: str,
    persona: str,
    strategy_instruction: str,
) -> str:
    """Turn a structured interviewing instruction into natural interviewer phrasing."""

    settings = get_settings()
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA API key is not configured")

    system_prompt = (
        "You are a realistic interview simulator speaking to a student candidate. "
        f"Interview role: {role}. Persona: {persona}. "
        "Respond as the interviewer only. "
        "Your job is to ask the next interview question or probe, not to answer on behalf of the candidate. "
        "Never roleplay as the candidate. Never say what the candidate did, built, learned, or achieved in first person. "
        "Never provide a sample answer unless explicitly asked, which you are not here. "
        "Output only the exact interviewer utterance that should be spoken next. "
        "Keep it concise, natural, and challenging when needed. "
        "Prefer one short paragraph, usually ending as a direct question. "
        "Do not explain your internal reasoning."
    )

    payload = {
        "model": settings.nvidia_live_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": strategy_instruction},
        ],
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 220,
    }

    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(NIM_CHAT_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    return coerce_to_interviewer_question(
        data["choices"][0]["message"]["content"].strip(),
        strategy_instruction=strategy_instruction,
    )


def coerce_to_interviewer_question(reply: str, *, strategy_instruction: str) -> str:
    """Guard against model drift into candidate-answer mode."""

    text = " ".join(reply.split())
    lowered = text.lower()

    candidate_markers = (
        "i built",
        "i worked on",
        "i developed",
        "i implemented",
        "my project",
        "in my project",
        "i used",
        "i focused on",
        "i was particularly drawn",
    )

    if any(marker in lowered for marker in candidate_markers):
        fallback = strategy_instruction.strip()
        if not fallback.endswith("?"):
            fallback = f"{fallback.rstrip('.')}?"
        return fallback

    if "?" not in text:
        fallback = strategy_instruction.strip()
        if not fallback.endswith("?"):
            fallback = f"{fallback.rstrip('.')}?"
        return fallback

    return text

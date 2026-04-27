"""Services that coordinate the realtime interview loop."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, status
from redis import Redis
from sqlalchemy.orm import Session

from app.db.models.interview import Interview
from app.db.models.user import User
from app.db.session import redis_client
from app.providers.cartesia_tts import synthesize_speech
from app.providers.gemini_reasoning import analyze_turn_and_plan_reply
from app.providers.nim_interviewer import render_interviewer_reply
from app.schemas.blueprint import InterviewBlueprintPayload
from app.schemas.interview import InterviewResponse, RealtimeSessionState, TranscriptTurnResponse
from app.services.file_storage import save_binary_payload
from app.services.interview_service import (
    create_transcript_turn,
    get_blueprint_payload,
    get_user_interview,
    list_transcript_turns,
)


@dataclass
class RealtimeKeys:
    """Redis key names for a single interview session."""

    presence: str
    lock: str


def get_realtime_keys(interview_id: str, user_id: str) -> RealtimeKeys:
    """Build redis key names used by the realtime session."""

    return RealtimeKeys(
        presence=f"presence:interview:{interview_id}:user:{user_id}",
        lock=f"lock:interview:{interview_id}",
    )


def mark_presence(*, redis: Redis, key: str, ttl_seconds: int) -> None:
    """Refresh ephemeral presence for the connected interview session."""

    redis.set(key, "online", ex=ttl_seconds)


def clear_presence(*, redis: Redis, key: str) -> None:
    """Remove ephemeral presence when the websocket disconnects."""

    redis.delete(key)


def acquire_interview_lock(*, redis: Redis, key: str, ttl_seconds: int) -> bool:
    """Acquire a short-lived lock so only one response is generated at a time."""

    return bool(redis.set(key, "locked", nx=True, ex=ttl_seconds))


def release_interview_lock(*, redis: Redis, key: str) -> None:
    """Release the short-lived interview generation lock."""

    redis.delete(key)


def require_realtime_interview(db: Session, *, user: User, interview_id: str) -> Interview:
    """Load an owned interview or raise a user-facing 404."""

    interview = get_user_interview(db, user, interview_id)
    if interview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")
    if interview.blueprint is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Interview blueprint is missing. Bootstrap the interview first.",
        )
    return interview


def build_session_state(db: Session, interview: Interview) -> RealtimeSessionState:
    """Return the client-facing session snapshot used at connect time."""

    transcript = list_transcript_turns(db, interview.id)
    blueprint = get_blueprint_payload(db, interview)
    current_topic = None
    if transcript and transcript[-1].metadata_json:
        current_topic = transcript[-1].metadata_json.get("target_topic")
    elif blueprint and blueprint.topics:
        current_topic = blueprint.topics[0].topic

    return RealtimeSessionState(
        interview=InterviewResponse.model_validate(interview),
        current_topic=current_topic,
        transcript=[TranscriptTurnResponse.model_validate(item) for item in transcript],
    )


async def ensure_opening_turn(db: Session, interview: Interview) -> TranscriptTurnResponse | None:
    """Create the opening interviewer message once for a new interview."""

    existing_turns = list_transcript_turns(db, interview.id)
    if existing_turns:
        return None

    blueprint = get_blueprint_payload(db, interview)
    if blueprint is None:
        return None

    opening_turn = create_transcript_turn(
        db,
        interview=interview,
        role="assistant",
        content=blueprint.opening_prompt,
        metadata={"target_topic": blueprint.topics[0].topic if blueprint.topics else None, "source": "opening_prompt"},
    )
    return TranscriptTurnResponse.model_validate(opening_turn)


async def process_user_answer(
    db: Session,
    *,
    interview: Interview,
    answer: str,
) -> TranscriptTurnResponse:
    """Persist the user's answer and return the saved transcript turn."""

    user_turn = create_transcript_turn(
        db,
        interview=interview,
        role="user",
        content=answer,
    )
    return TranscriptTurnResponse.model_validate(user_turn)


async def generate_assistant_reply(
    db: Session,
    *,
    interview: Interview,
) -> TranscriptTurnResponse:
    """Run adaptive reasoning and persist the assistant reply."""

    blueprint = get_blueprint_payload(db, interview)
    if blueprint is None:
        raise RuntimeError("Interview blueprint is missing")

    transcript = list_transcript_turns(db, interview.id)
    reasoning_result = await analyze_turn_and_plan_reply(
        role=interview.role,
        persona=interview.persona,
        parsed_resume=interview.parsed_resume,
        blueprint=blueprint,
        transcript=[
            {
                "turn_index": item.turn_index,
                "role": item.role,
                "content": item.content,
                "metadata": item.metadata_json,
            }
            for item in transcript
        ],
    )

    strategy_instruction = reasoning_result.get("strategy_instruction", blueprint.opening_prompt)
    assistant_text = await render_interviewer_reply(
        role=interview.role,
        persona=interview.persona,
        strategy_instruction=strategy_instruction,
    )
    audio_metadata: dict = {}
    try:
        audio_bytes, content_type = await synthesize_speech(transcript=assistant_text)
        stored_filename, storage_path = save_binary_payload(
            payload=audio_bytes,
            namespace=f"{interview.user_id}/tts",
            suffix=".wav",
        )
        audio_metadata = {
            "audio_content_type": content_type,
            "audio_storage_path": storage_path,
            "audio_stored_filename": stored_filename,
        }
    except Exception as exc:  # noqa: BLE001
        audio_metadata = {"tts_error": str(exc)}

    assistant_turn = create_transcript_turn(
        db,
        interview=interview,
        role="assistant",
        content=assistant_text,
        metadata={**reasoning_result, **audio_metadata},
    )

    return TranscriptTurnResponse.model_validate(assistant_turn)

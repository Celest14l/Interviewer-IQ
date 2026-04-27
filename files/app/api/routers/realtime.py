"""Realtime websocket endpoints for the live interview loop."""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.core.settings import get_settings
from app.db.models.user import User
from app.db.session import SessionLocal, redis_client
from app.schemas.interview import ServerRealtimeEvent
from app.services.audio_service import transcribe_audio_bytes
from app.services.realtime_service import (
    acquire_interview_lock,
    build_session_state,
    clear_presence,
    ensure_opening_turn,
    generate_assistant_reply,
    get_realtime_keys,
    mark_presence,
    process_user_answer,
    release_interview_lock,
    require_realtime_interview,
)

router = APIRouter(tags=["Realtime"])


def _get_user_from_token(db: Session, token: str | None) -> User | None:
    """Resolve a websocket bearer token into the owning user."""

    if not token:
        return None
    payload = decode_access_token(token)
    if payload is None:
        return None
    user = db.get(User, payload.get("sub"))
    if user is None or not user.is_active:
        return None
    return user


@router.websocket("/ws/interviews/{interview_id}")
async def interview_websocket(
    websocket: WebSocket,
    interview_id: str,
    token: str | None = Query(default=None),
) -> None:
    """Drive a single live interview session over websocket."""

    settings = get_settings()
    db = SessionLocal()
    keys = None

    try:
        user = _get_user_from_token(db, token)
        if user is None:
            await websocket.close(code=4401, reason="Invalid or expired token")
            return

        interview = require_realtime_interview(db, user=user, interview_id=interview_id)
        keys = get_realtime_keys(interview.id, user.id)
        await websocket.accept()

        mark_presence(
            redis=redis_client,
            key=keys.presence,
            ttl_seconds=settings.realtime_presence_ttl_seconds,
        )

        await ensure_opening_turn(db, interview)
        session_state = build_session_state(db, interview)
        await websocket.send_json(
            ServerRealtimeEvent(
                type="session_state",
                session=session_state,
            ).model_dump(mode="json")
        )

        while True:
            payload = await websocket.receive_json()
            event_type = payload.get("type")

            mark_presence(
                redis=redis_client,
                key=keys.presence,
                ttl_seconds=settings.realtime_presence_ttl_seconds,
            )

            if event_type == "ping":
                await websocket.send_json(ServerRealtimeEvent(type="pong").model_dump(mode="json"))
                continue

            if event_type not in {"user_answer", "user_audio"}:
                await websocket.send_json(
                    ServerRealtimeEvent(
                        type="error",
                        message="Unsupported realtime event",
                    ).model_dump(mode="json")
                )
                continue

            if not acquire_interview_lock(
                redis=redis_client,
                key=keys.lock,
                ttl_seconds=settings.realtime_interview_lock_seconds,
            ):
                await websocket.send_json(
                    ServerRealtimeEvent(
                        type="busy",
                        message="The interview is already processing another answer.",
                    ).model_dump(mode="json")
                )
                continue

            transcription = None
            user_turn = None
            error_stage = "input_processing"
            try:
                if event_type == "user_audio":
                    audio_base64 = payload.get("audio_base64") or ""
                    if not audio_base64:
                        await websocket.send_json(
                            ServerRealtimeEvent(
                                type="error",
                                message="Audio payload is required",
                            ).model_dump(mode="json")
                        )
                        continue

                    try:
                        audio_bytes = base64.b64decode(audio_base64, validate=True)
                    except (binascii.Error, ValueError):
                        await websocket.send_json(
                            ServerRealtimeEvent(
                                type="error",
                                message="Audio payload must be valid base64",
                            ).model_dump(mode="json")
                        )
                        continue

                    transcription = await transcribe_audio_bytes(
                        db,
                        user=user,
                        audio_bytes=audio_bytes,
                        filename=payload.get("filename") or "audio.webm",
                        content_type=payload.get("content_type"),
                        interview_id=interview.id,
                        language=payload.get("language"),
                        prompt=payload.get("prompt"),
                    )
                    content = transcription.text.strip()
                    if not content:
                        await websocket.send_json(
                            ServerRealtimeEvent(
                                type="error",
                                message="Transcription produced no text",
                            ).model_dump(mode="json")
                        )
                        continue
                    error_stage = "user_turn_persistence"
                else:
                    content = (payload.get("content") or "").strip()
                    if not content:
                        await websocket.send_json(
                            ServerRealtimeEvent(
                                type="error",
                                message="Answer content is required",
                            ).model_dump(mode="json")
                        )
                        continue
                    error_stage = "user_turn_persistence"

                user_turn = await process_user_answer(
                    db,
                    interview=interview,
                    answer=content,
                )
                await websocket.send_json(
                    ServerRealtimeEvent(
                        type="user_turn",
                        turn=user_turn,
                        metadata=(
                            {
                                "transcription": transcription.model_dump(),
                                "input_mode": "audio",
                            }
                            if transcription is not None
                            else {"input_mode": "text"}
                        ),
                    ).model_dump(mode="json")
                )
                error_stage = "assistant_generation"
                assistant_turn = await generate_assistant_reply(
                    db,
                    interview=interview,
                )
                await websocket.send_json(
                    ServerRealtimeEvent(
                        type="assistant_turn",
                        turn=assistant_turn,
                        metadata=assistant_turn.metadata_json,
                    ).model_dump(mode="json")
                )
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json(
                    ServerRealtimeEvent(
                        type="error",
                        message=str(exc),
                        metadata={
                            "stage": error_stage,
                            "input_mode": "audio" if transcription is not None else "text",
                        },
                    ).model_dump(mode="json")
                )
            finally:
                release_interview_lock(redis=redis_client, key=keys.lock)

    except WebSocketDisconnect:
        pass
    finally:
        if keys is not None:
            clear_presence(redis=redis_client, key=keys.presence)
            release_interview_lock(redis=redis_client, key=keys.lock)
        db.close()

"""Interview service functions."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.interview import Interview
from app.db.models.interview_blueprint import InterviewBlueprint
from app.db.models.transcript_turn import TranscriptTurn
from app.db.models.user import User
from app.schemas.blueprint import InterviewBlueprintPayload
from app.schemas.interview import InterviewCreate


def create_interview(db: Session, user: User, payload: InterviewCreate) -> Interview:
    interview = Interview(
        user_id=user.id,
        role=payload.role,
        persona=payload.persona,
        status="draft",
    )
    db.add(interview)
    db.commit()
    db.refresh(interview)
    return interview


def list_user_interviews(db: Session, user: User) -> list[Interview]:
    result = db.scalars(
        select(Interview).where(Interview.user_id == user.id).order_by(Interview.created_at.desc())
    )
    return list(result)


def get_user_interview(db: Session, user: User, interview_id: str) -> Interview | None:
    """Fetch a single interview owned by the authenticated user."""

    return db.scalar(
        select(Interview).where(
            Interview.id == interview_id,
            Interview.user_id == user.id,
        )
    )


def list_transcript_turns(db: Session, interview_id: str) -> list[TranscriptTurn]:
    """Return transcript turns for an interview in stable chronological order."""

    result = db.scalars(
        select(TranscriptTurn)
        .where(TranscriptTurn.interview_id == interview_id)
        .order_by(TranscriptTurn.turn_index.asc(), TranscriptTurn.created_at.asc())
    )
    return list(result)


def create_transcript_turn(
    db: Session,
    *,
    interview: Interview,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> TranscriptTurn:
    """Persist a new turn and return the refreshed ORM object."""

    next_index = len(list_transcript_turns(db, interview.id))
    turn = TranscriptTurn(
        interview_id=interview.id,
        turn_index=next_index,
        role=role,
        content=content,
        metadata_json=metadata,
    )
    db.add(turn)
    if interview.status == "draft":
        interview.status = "active"
    if interview.started_at is None:
        interview.started_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(turn)
    db.refresh(interview)
    return turn


def get_blueprint_payload(db: Session, interview: Interview) -> InterviewBlueprintPayload | None:
    """Hydrate the stored interview blueprint into a typed payload."""

    blueprint = db.scalar(
        select(InterviewBlueprint).where(InterviewBlueprint.interview_id == interview.id)
    )
    if blueprint is None:
        return None
    return InterviewBlueprintPayload.model_validate(blueprint.blueprint)

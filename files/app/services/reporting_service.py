"""Services for answer-level scoring and final report generation."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.db.models.answer_score import AnswerScore
from app.db.models.final_report import FinalReport
from app.db.models.user import User
from app.providers.gemini_reasoning import score_interview_answer, synthesize_final_report
from app.schemas.reporting import AnswerScorePayload, FinalReportPayload
from app.services.interview_service import get_user_interview, list_transcript_turns

RUBRIC_VERSION = "v1"
PROMPT_VERSION = "v1"


def list_answer_scores(db: Session, *, interview_id: str) -> list[AnswerScore]:
    """Return answer scores for an interview in creation order."""

    result = db.scalars(
        select(AnswerScore)
        .where(AnswerScore.interview_id == interview_id)
        .order_by(AnswerScore.created_at.asc())
    )
    return list(result)


async def generate_answer_scores_for_interview(
    db: Session,
    *,
    user: User,
    interview_id: str,
) -> list[AnswerScore]:
    """Score any user turns that do not already have persisted answer scores."""

    interview = get_user_interview(db, user, interview_id)
    if interview is None:
        raise ValueError("Interview not found")

    transcript = list_transcript_turns(db, interview.id)
    existing_scores = {item.transcript_turn_id for item in list_answer_scores(db, interview_id=interview.id)}

    previous_assistant_question: str | None = None
    for turn in transcript:
        if turn.role == "assistant":
            previous_assistant_question = turn.content
            continue
        if turn.role != "user" or turn.id in existing_scores:
            continue

        existing_score = db.scalar(
            select(AnswerScore).where(AnswerScore.transcript_turn_id == turn.id)
        )
        if existing_score is not None:
            existing_scores.add(turn.id)
            continue

        score_payload = await score_interview_answer(
            role=interview.role,
            question=previous_assistant_question,
            answer=turn.content,
            parsed_resume=interview.parsed_resume,
        )
        payload_model = AnswerScorePayload.model_validate(score_payload)
        score = AnswerScore(
            interview_id=interview.id,
            transcript_turn_id=turn.id,
            score_payload=payload_model.model_dump(),
            model_name="gemini-2.5-flash",
            rubric_version=RUBRIC_VERSION,
            prompt_version=PROMPT_VERSION,
        )
        try:
            db.add(score)
            db.commit()
            db.refresh(score)
            existing_scores.add(turn.id)
        except IntegrityError:
            db.rollback()
            existing_scores.add(turn.id)
            continue
        except OperationalError as exc:
            db.rollback()
            locked_score = db.scalar(
                select(AnswerScore).where(AnswerScore.transcript_turn_id == turn.id)
            )
            if locked_score is not None:
                existing_scores.add(turn.id)
                continue
            raise RuntimeError(
                "Answer scoring is currently busy for this interview. Please retry the final report in a few seconds."
            ) from exc

    return list_answer_scores(db, interview_id=interview.id)


async def generate_final_report_for_interview(
    db: Session,
    *,
    user: User,
    interview_id: str,
) -> FinalReport:
    """Generate or refresh the final report for an interview."""

    interview = get_user_interview(db, user, interview_id)
    if interview is None:
        raise ValueError("Interview not found")

    scores = await generate_answer_scores_for_interview(db, user=user, interview_id=interview_id)
    transcript = list_transcript_turns(db, interview.id)

    report_payload = await synthesize_final_report(
        role=interview.role,
        parsed_resume=interview.parsed_resume,
        transcript=[
            {
                "turn_index": item.turn_index,
                "role": item.role,
                "content": item.content,
                "metadata": item.metadata_json,
            }
            for item in transcript
        ],
        answer_scores=[item.score_payload for item in scores],
    )
    payload_model = FinalReportPayload.model_validate(report_payload)

    report = db.scalar(select(FinalReport).where(FinalReport.interview_id == interview.id))
    if report is None:
        report = FinalReport(
            interview_id=interview.id,
            report=payload_model.model_dump(),
            model_name="gemini-2.5-flash",
            prompt_version=PROMPT_VERSION,
        )
        db.add(report)
    else:
        report.report = payload_model.model_dump()
        report.model_name = "gemini-2.5-flash"
        report.prompt_version = PROMPT_VERSION

    db.commit()
    db.refresh(report)
    return report


def get_final_report(db: Session, *, user: User, interview_id: str) -> FinalReport | None:
    """Fetch a persisted final report for an owned interview."""

    interview = get_user_interview(db, user, interview_id)
    if interview is None:
        return None
    return db.scalar(select(FinalReport).where(FinalReport.interview_id == interview.id))

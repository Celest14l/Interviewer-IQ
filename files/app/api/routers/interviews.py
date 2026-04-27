"""Interview, scoring, and report endpoints backed by Postgres."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.db.models.user import User
from app.db.session import get_db
from app.schemas.blueprint import InterviewBootstrapResponse
from app.schemas.interview import InterviewCreate, InterviewResponse
from app.schemas.reporting import AnswerScoreResponse, FinalReportResponse
from app.services.interview_service import create_interview, get_user_interview, list_user_interviews
from app.services.reporting_service import (
    generate_answer_scores_for_interview,
    generate_final_report_for_interview,
    get_final_report,
    list_answer_scores,
)
from app.services.resume_service import bootstrap_interview_from_resume

router = APIRouter(prefix="/interviews", tags=["Interviews"])


@router.post("", response_model=InterviewResponse, status_code=status.HTTP_201_CREATED)
def create_interview_endpoint(
    payload: InterviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> InterviewResponse:
    interview = create_interview(db, current_user, payload)
    return InterviewResponse.model_validate(interview)


@router.get("", response_model=list[InterviewResponse])
def list_interviews_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[InterviewResponse]:
    interviews = list_user_interviews(db, current_user)
    return [InterviewResponse.model_validate(item) for item in interviews]


@router.post("/bootstrap", response_model=InterviewBootstrapResponse, status_code=status.HTTP_201_CREATED)
async def bootstrap_interview_endpoint(
    resume: UploadFile = File(...),
    role: str = Form(...),
    persona: str = Form("friendly_hr"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> InterviewBootstrapResponse:
    return await bootstrap_interview_from_resume(
        db=db,
        user=current_user,
        resume=resume,
        role=role,
        persona=persona,
    )


@router.post("/{interview_id}/score", response_model=list[AnswerScoreResponse])
async def score_interview_endpoint(
    interview_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[AnswerScoreResponse]:
    """Generate rubric-based answer scores for an interview transcript."""

    try:
        scores = await generate_answer_scores_for_interview(
            db,
            user=current_user,
            interview_id=interview_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return [
        AnswerScoreResponse(
            id=item.id,
            interview_id=item.interview_id,
            transcript_turn_id=item.transcript_turn_id,
            score_payload=item.score_payload,
            model_name=item.model_name,
            rubric_version=item.rubric_version,
            prompt_version=item.prompt_version,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
        for item in scores
    ]


@router.get("/{interview_id}/score", response_model=list[AnswerScoreResponse])
def list_interview_scores_endpoint(
    interview_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[AnswerScoreResponse]:
    """Return persisted answer-level scores for an interview."""

    interview = get_user_interview(db, current_user, interview_id)
    if interview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")

    return [
        AnswerScoreResponse(
            id=item.id,
            interview_id=item.interview_id,
            transcript_turn_id=item.transcript_turn_id,
            score_payload=item.score_payload,
            model_name=item.model_name,
            rubric_version=item.rubric_version,
            prompt_version=item.prompt_version,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
        for item in list_answer_scores(db, interview_id=interview_id)
    ]


@router.post("/{interview_id}/report", response_model=FinalReportResponse)
async def generate_report_endpoint(
    interview_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FinalReportResponse:
    """Generate or refresh the final interview report."""

    try:
        report = await generate_final_report_for_interview(
            db,
            user=current_user,
            interview_id=interview_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return FinalReportResponse(
        id=report.id,
        interview_id=report.interview_id,
        report=report.report,
        model_name=report.model_name,
        prompt_version=report.prompt_version,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )


@router.get("/{interview_id}/report", response_model=FinalReportResponse)
def get_report_endpoint(
    interview_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FinalReportResponse:
    """Return the persisted final report for an interview."""

    report = get_final_report(db, user=current_user, interview_id=interview_id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Final report not found")

    return FinalReportResponse(
        id=report.id,
        interview_id=report.interview_id,
        report=report.report,
        model_name=report.model_name,
        prompt_version=report.prompt_version,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )

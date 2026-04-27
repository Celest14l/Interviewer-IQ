"""Analysis endpoints for face snapshot ingestion and retrieval."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.db.models.user import User
from app.db.session import get_db
from app.schemas.analysis import (
    SnapshotAnalysisRequest,
    SnapshotAnalysisResponse,
    SnapshotEventResponse,
    SnapshotSummaryResponse,
)
from app.services.face_analysis_service import (
    analyze_face_snapshot,
    list_snapshot_events,
    summarize_snapshot_events,
)

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/snapshot", response_model=SnapshotAnalysisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_snapshot_endpoint(
    payload: SnapshotAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SnapshotAnalysisResponse:
    """Analyze a camera snapshot and persist the resulting coaching signal."""

    return await analyze_face_snapshot(
        db,
        user=current_user,
        interview_id=payload.interview_id,
        image_b64=payload.image,
    )


@router.get("/snapshots/{interview_id}", response_model=SnapshotSummaryResponse)
def list_snapshots_endpoint(
    interview_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SnapshotSummaryResponse:
    """Return stored snapshot events and aggregate coaching scores."""

    events = list_snapshot_events(db, user=current_user, interview_id=interview_id)
    return SnapshotSummaryResponse(
        interview_id=interview_id,
        count=len(events),
        snapshots=[
            SnapshotEventResponse(
                id=item.id,
                interview_id=item.interview_id,
                image_path=item.image_path,
                analysis_payload=item.analysis_payload,
                model_name=item.model_name,
                created_at=item.created_at,
            )
            for item in events
        ],
        average_scores=summarize_snapshot_events(events),
    )

"""Schemas for answer scoring and final report generation."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ScoreDimension(BaseModel):
    """One rubric dimension with numeric score and supporting rationale."""

    score: int = Field(ge=1, le=10)
    rationale: str
    evidence: str


class AnswerScorePayload(BaseModel):
    """Structured answer-level score returned by the evaluator."""

    question: str | None = None
    answer: str
    answer_quality: str
    overall_score: int = Field(ge=1, le=10)
    relevance: ScoreDimension
    completeness: ScoreDimension
    technical_correctness: ScoreDimension
    clarity: ScoreDimension
    structure: ScoreDimension
    ownership: ScoreDimension
    communication: ScoreDimension
    improvement_tip: str


class AnswerScoreResponse(BaseModel):
    """Persisted answer score exposed via the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    interview_id: str
    transcript_turn_id: str
    score_payload: AnswerScorePayload
    model_name: str | None = None
    rubric_version: str | None = None
    prompt_version: str | None = None
    created_at: datetime
    updated_at: datetime


class FinalReportPayload(BaseModel):
    """Final interview report synthesized from transcript and structured scores."""

    overall_score: int = Field(ge=1, le=10)
    readiness_level: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    improvement_areas: list[str] = Field(default_factory=list)
    coaching_recommendations: list[str] = Field(default_factory=list)
    topic_breakdown: list[dict] = Field(default_factory=list)


class FinalReportResponse(BaseModel):
    """Persisted final report exposed via the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    interview_id: str
    report: FinalReportPayload
    model_name: str | None = None
    prompt_version: str | None = None
    created_at: datetime
    updated_at: datetime

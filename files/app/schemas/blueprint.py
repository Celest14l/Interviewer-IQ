"""Blueprint and grounded resume schemas."""

from pydantic import BaseModel, Field


class ResumeProject(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    claimed_role: str | None = None


class ResumeGrounding(BaseModel):
    candidate_summary: str
    top_skills: list[str] = Field(default_factory=list)
    projects: list[ResumeProject] = Field(default_factory=list)
    claim_checks: list[str] = Field(default_factory=list)
    depth_areas: list[str] = Field(default_factory=list)


class BlueprintTopic(BaseModel):
    topic: str
    resume_basis: str
    primary_question: str
    follow_up_probe: str
    challenge_probe: str
    ownership_probe: str
    completion_criteria: list[str] = Field(default_factory=list)


class InterviewBlueprintPayload(BaseModel):
    opening_prompt: str
    topics: list[BlueprintTopic] = Field(default_factory=list)


class InterviewBootstrapResponse(BaseModel):
    interview_id: str
    role: str
    persona: str
    parsed_resume: ResumeGrounding
    blueprint: InterviewBlueprintPayload

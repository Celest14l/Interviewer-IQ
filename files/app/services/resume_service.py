"""Resume extraction and bootstrap services."""

from __future__ import annotations

import io

from fastapi import HTTPException, UploadFile, status
from PyPDF2 import PdfReader
from sqlalchemy.orm import Session

from app.db.models.interview import Interview
from app.db.models.interview_blueprint import InterviewBlueprint
from app.db.models.uploaded_file import UploadedFile
from app.db.models.user import User
from app.providers.gemini_reasoning import generate_grounding_and_blueprint
from app.schemas.blueprint import InterviewBootstrapResponse
from app.services.file_storage import save_upload_file


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not read PDF: {exc}") from exc

    text_parts: list[str] = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    text = "\n".join(part.strip() for part in text_parts if part.strip()).strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Resume PDF contains no extractable text")
    return text


async def bootstrap_interview_from_resume(
    *,
    db: Session,
    user: User,
    resume: UploadFile,
    role: str,
    persona: str,
) -> InterviewBootstrapResponse:
    if not (resume.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF resumes are supported")

    raw_bytes, stored_filename, storage_path = await save_upload_file(resume, namespace=user.id)
    resume_text = extract_text_from_pdf(raw_bytes)
    grounding, blueprint_payload = await generate_grounding_and_blueprint(
        resume_text=resume_text,
        role=role,
        persona=persona,
    )

    interview = Interview(
        user_id=user.id,
        role=role,
        persona=persona,
        status="bootstrapped",
        resume_text=resume_text,
        parsed_resume=grounding.model_dump(),
    )
    db.add(interview)
    db.flush()

    uploaded_file = UploadedFile(
        user_id=user.id,
        interview_id=interview.id,
        original_filename=resume.filename or "resume.pdf",
        stored_filename=stored_filename,
        content_type=resume.content_type,
        size_bytes=len(raw_bytes),
        storage_path=storage_path,
    )
    db.add(uploaded_file)

    blueprint = InterviewBlueprint(
        interview_id=interview.id,
        blueprint=blueprint_payload.model_dump(),
        model_name="gemini-2.5-flash",
        prompt_version="v1",
    )
    db.add(blueprint)
    db.commit()
    db.refresh(interview)

    return InterviewBootstrapResponse(
        interview_id=interview.id,
        role=interview.role,
        persona=interview.persona,
        parsed_resume=grounding,
        blueprint=blueprint_payload,
    )

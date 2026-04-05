"""
routers/interview.py
REST endpoints for starting and managing an interview session.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from services.resume_parser import extract_text_from_pdf, parse_resume, generate_questions
from services.session_store import session_store
from services.interview_engine import InterviewEngine

router = APIRouter()


# ── 1. Upload resume + create session ─────────────────────────────────────
@router.post("/start")
async def start_interview(
    resume: UploadFile = File(...),
    persona: str = Form("friendly_hr"),
    role: str    = Form("Software Engineer"),
):
    """
    Upload a PDF resume, parse it, generate questions, create a session.
    Returns { session_id, parsed_resume, questions }
    """
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are accepted.")

    pdf_bytes = await resume.read()
    if len(pdf_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Resume too large (max 5 MB).")

    raw_text      = await extract_text_from_pdf(pdf_bytes)
    parsed_resume = await parse_resume(raw_text)
    questions     = await generate_questions(parsed_resume, persona, role)

    session_id = session_store.create(
        resume_text=raw_text,
        persona=persona,
        role=role,
    )
    # Store questions in session for the engine to use
    sess = session_store.get(session_id)
    sess["parsed_resume"] = parsed_resume
    sess["questions"]     = questions

    return {
        "session_id":    session_id,
        "parsed_resume": parsed_resume,
        "questions":     questions,
        "persona":       persona,
        "role":          role,
        "ws_url":        f"/ws/interview/{session_id}",
    }


# ── 2. Get the opening message (before WS is ready) ───────────────────────
@router.get("/{session_id}/open")
async def get_opening(session_id: str):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    engine = InterviewEngine(session_id)
    opening = await engine.opening()
    return opening


# ── 3. Send a text message (non-WS alternative) ───────────────────────────
class MessagePayload(BaseModel):
    text: str

@router.post("/{session_id}/message")
async def send_message(session_id: str, payload: MessagePayload):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    engine = InterviewEngine(session_id)
    # Replay existing conversation into engine
    engine.conversation = list(sess.get("messages", []))
    engine.q_index = _estimate_q_index(sess)

    reply = await engine.respond(payload.text)
    return reply


# ── 4. End interview + get report ─────────────────────────────────────────
@router.post("/{session_id}/end")
async def end_interview(session_id: str):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    engine = InterviewEngine(session_id)
    engine.conversation = list(sess.get("messages", []))
    engine.q_index = len(sess.get("questions", []))

    report = await engine.finalize()
    session_store.save_report(session_id, report)
    return {"report_url": f"/api/session/{session_id}/report", "summary": report["scores"]}


# ── Helpers ────────────────────────────────────────────────────────────────
def _estimate_q_index(sess: dict) -> int:
    user_turns = sum(1 for m in sess.get("messages", []) if m["role"] == "user")
    return min(user_turns, len(sess.get("questions", [])))

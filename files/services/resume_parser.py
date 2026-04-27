"""
services/resume_parser.py
Extracts structured info from a PDF resume.
Uses PyPDF2 for text extraction, then Claude to parse key fields.
"""

import io
import os
import json
import httpx
from typing import BinaryIO


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"


async def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from PDF bytes."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")


async def parse_resume(raw_text: str) -> dict:
    """
    Use Claude to parse resume into structured fields:
    name, skills, experience, projects, education
    """
    prompt = f"""You are a resume parser. Extract the following from this resume text and return ONLY valid JSON (no markdown, no explanation):
{{
  "name": "...",
  "email": "...",
  "skills": ["skill1", "skill2"],
  "experience": [
    {{"company": "...", "role": "...", "duration": "...", "highlights": ["..."]}}
  ],
  "projects": [
    {{"name": "...", "description": "...", "tech_stack": ["..."]}}
  ],
  "education": [
    {{"institution": "...", "degree": "...", "year": "..."}}
  ],
  "summary": "2-3 sentence professional summary of the candidate"
}}

Resume text:
{raw_text[:4000]}
"""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["content"][0]["text"]
        # strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # fallback: return raw text as summary
            return {"summary": raw, "skills": [], "experience": [], "projects": [], "education": []}


async def generate_questions(parsed_resume: dict, persona: str, role: str) -> list[str]:
    """
    Generate 8-10 personalised interview questions from the parsed resume.
    """
    persona_descriptions = {
        "friendly_hr":      "You are a warm, encouraging HR interviewer conducting a campus placement interview.",
        "strict_technical": "You are a rigorous technical interviewer who digs deep into implementation details.",
        "stress_interviewer":"You are a challenging interviewer who uses stress-testing and rapid-fire questions.",
        "placement_panel":  "You are a panel of 3 campus placement officers evaluating overall fit.",
    }
    persona_desc = persona_descriptions.get(persona, persona_descriptions["friendly_hr"])

    skills   = ", ".join(parsed_resume.get("skills", [])[:8])
    projects = "; ".join(p.get("name", "") for p in parsed_resume.get("projects", [])[:3])
    summary  = parsed_resume.get("summary", "")

    prompt = f"""{persona_desc}

The candidate is applying for: {role}
Skills: {skills}
Projects: {projects}
Summary: {summary}

Generate exactly 8 interview questions tailored to this candidate. Mix behavioural, technical, and situational questions. Return ONLY a JSON array of 8 strings — no numbering, no markdown:
["question1", "question2", ..., "question8"]
"""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 800,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["content"][0]["text"].replace("```json", "").replace("```", "").strip()
        try:
            questions = json.loads(raw)
            return questions if isinstance(questions, list) else []
        except Exception:
            # fallback generic questions
            return [
                f"Tell me about yourself and your interest in the {role} role.",
                "Walk me through your most impactful project.",
                "Describe a challenging technical problem you solved.",
                "How do you handle tight deadlines?",
                "What is your strongest technical skill and how did you develop it?",
                "Describe a time you worked in a team under pressure.",
                "Where do you see yourself in 5 years?",
                "Do you have any questions for us?",
            ]

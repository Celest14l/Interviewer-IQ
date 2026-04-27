"""Gemini adapters for grounding resumes and adaptive interview reasoning."""

from __future__ import annotations

import json

import httpx

from app.core.settings import get_settings
from app.schemas.blueprint import InterviewBlueprintPayload, ResumeGrounding

GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


async def generate_grounding_and_blueprint(
    *,
    resume_text: str,
    role: str,
    persona: str,
) -> tuple[ResumeGrounding, InterviewBlueprintPayload]:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini API key is not configured")

    prompt = f"""
You are building an adaptive interview blueprint for a student interview simulator.

Role: {role}
Persona: {persona}

Given the resume text below, return STRICT JSON with this shape:
{{
  "parsed_resume": {{
    "candidate_summary": "short grounded summary",
    "top_skills": ["skill1", "skill2"],
    "projects": [
      {{
        "name": "project name",
        "description": "short project summary",
        "technologies": ["tech1", "tech2"],
        "claimed_role": "what the candidate seems to have owned"
      }}
    ],
    "claim_checks": ["claims worth verifying"],
    "depth_areas": ["areas worth probing deeply"]
  }},
  "blueprint": {{
    "opening_prompt": "one opening interviewer message",
    "topics": [
      {{
        "topic": "topic name",
        "resume_basis": "what in the resume caused this topic",
        "primary_question": "main question",
        "follow_up_probe": "probe if answer is vague",
        "challenge_probe": "probe if answer sounds overclaimed or too shallow",
        "ownership_probe": "probe personal contribution",
        "completion_criteria": ["criterion 1", "criterion 2"]
      }}
    ]
  }}
}}

Keep it grounded in the actual resume. Prefer skills and projects the student explicitly mentions.
Do not invent employers, projects, metrics, or technologies that are not supported by the resume.
Return JSON only.

Resume text:
\"\"\"
{resume_text[:12000]}
\"\"\"
""".strip()

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
        },
    }

    url = GEMINI_URL_TEMPLATE.format(model=settings.gemini_reasoning_model)
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{url}?key={settings.gemini_api_key}",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(raw_text)
    return (
        ResumeGrounding.model_validate(parsed["parsed_resume"]),
        InterviewBlueprintPayload.model_validate(parsed["blueprint"]),
    )


async def analyze_turn_and_plan_reply(
    *,
    role: str,
    persona: str,
    parsed_resume: dict | None,
    blueprint: InterviewBlueprintPayload,
    transcript: list[dict],
) -> dict:
    """Choose the next interviewing move from the blueprint and recent transcript."""

    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini API key is not configured")

    compact_topics = [
        {
            "topic": item.topic,
            "primary_question": item.primary_question,
            "follow_up_probe": item.follow_up_probe,
            "challenge_probe": item.challenge_probe,
            "ownership_probe": item.ownership_probe,
            "completion_criteria": item.completion_criteria,
        }
        for item in blueprint.topics
    ]

    prompt = f"""
You are the adaptive reasoning controller for an interview simulator.

Role: {role}
Persona: {persona}

Your job is to decide the next interviewing move based on the structured blueprint and transcript so far.
Return STRICT JSON with this shape:
{{
  "current_topic": "topic name or null",
  "answer_quality": "strong|partial|weak|opening",
  "next_action": "ask_primary|follow_up|challenge|ownership_probe|move_to_next_topic|wrap_up",
  "target_topic": "topic name or null",
  "reason": "one short sentence",
  "strategy_instruction": "instruction for the interviewer renderer that says exactly what to ask next"
}}

Guidance:
- If there is no candidate answer yet, open with the blueprint opening prompt.
- Stay grounded in the blueprint topics and prior answers.
- Prefer follow-up questions when the answer is vague, shallow, or overclaimed.
- Prefer ownership probes when the candidate uses vague team language.
- Move to the next topic only when the current topic feels sufficiently covered.
- Keep strategy_instruction practical and specific.
- Return JSON only.

Parsed resume:
{json.dumps(parsed_resume or {}, ensure_ascii=True)}

Blueprint:
{json.dumps({"opening_prompt": blueprint.opening_prompt, "topics": compact_topics}, ensure_ascii=True)}

Transcript:
{json.dumps(transcript[-12:], ensure_ascii=True)}
""".strip()

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
        },
    }

    url = GEMINI_URL_TEMPLATE.format(model=settings.gemini_reasoning_model)
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{url}?key={settings.gemini_api_key}",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(raw_text)


async def score_interview_answer(
    *,
    role: str,
    question: str | None,
    answer: str,
    parsed_resume: dict | None,
) -> dict:
    """Score a single answer with a rubric-driven Gemini pass."""

    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini API key is not configured")

    prompt = f"""
You are scoring one interview answer for a student interview simulator.

Role: {role}
Question: {question or "unknown"}
Answer: {answer}

Return STRICT JSON with this shape:
{{
  "question": "copy of the question or null",
  "answer": "copy of the answer",
  "answer_quality": "strong|partial|weak",
  "overall_score": 1-10,
  "relevance": {{"score": 1-10, "rationale": "...", "evidence": "..."}},
  "completeness": {{"score": 1-10, "rationale": "...", "evidence": "..."}},
  "technical_correctness": {{"score": 1-10, "rationale": "...", "evidence": "..."}},
  "clarity": {{"score": 1-10, "rationale": "...", "evidence": "..."}},
  "structure": {{"score": 1-10, "rationale": "...", "evidence": "..."}},
  "ownership": {{"score": 1-10, "rationale": "...", "evidence": "..."}},
  "communication": {{"score": 1-10, "rationale": "...", "evidence": "..."}},
  "improvement_tip": "one practical coaching suggestion"
}}

Use only grounded evidence from the answer. Keep scores strict and realistic.
Return JSON only.

Parsed resume context:
{json.dumps(parsed_resume or {}, ensure_ascii=True)}
""".strip()

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }

    url = GEMINI_URL_TEMPLATE.format(model=settings.gemini_reasoning_model)
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{url}?key={settings.gemini_api_key}", json=payload)
        response.raise_for_status()
        data = response.json()

    return json.loads(data["candidates"][0]["content"]["parts"][0]["text"])


async def synthesize_final_report(
    *,
    role: str,
    parsed_resume: dict | None,
    transcript: list[dict],
    answer_scores: list[dict],
) -> dict:
    """Generate a final report from transcript and structured answer scores."""

    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini API key is not configured")

    prompt = f"""
You are writing a final interview coaching report for a student interview simulator.

Role: {role}

Return STRICT JSON with this shape:
{{
  "overall_score": 1-10,
  "readiness_level": "Needs Practice|Emerging|Interview Ready|Strong",
  "summary": "short grounded summary",
  "strengths": ["..."],
  "improvement_areas": ["..."],
  "coaching_recommendations": ["..."],
  "topic_breakdown": [
    {{
      "topic": "topic name",
      "assessment": "brief assessment",
      "score": 1-10
    }}
  ]
}}

Stay grounded in the transcript and answer scores. Do not invent evidence.
Return JSON only.

Parsed resume:
{json.dumps(parsed_resume or {}, ensure_ascii=True)}

Transcript:
{json.dumps(transcript[-30:], ensure_ascii=True)}

Answer scores:
{json.dumps(answer_scores, ensure_ascii=True)}
""".strip()

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.15,
        },
    }

    url = GEMINI_URL_TEMPLATE.format(model=settings.gemini_reasoning_model)
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(f"{url}?key={settings.gemini_api_key}", json=payload)
        response.raise_for_status()
        data = response.json()

    return json.loads(data["candidates"][0]["content"]["parts"][0]["text"])

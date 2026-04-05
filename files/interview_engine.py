"""
services/interview_engine.py
Drives the live AI interview conversation.
Manages question sequencing, follow-ups, and per-answer scoring.
"""

import os
import json
import httpx
from services.session_store import session_store

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"


PERSONA_SYSTEM_PROMPTS = {
    "friendly_hr": (
        "You are Priya, a warm and encouraging HR interviewer at a top Indian tech firm. "
        "You conduct campus placement interviews. Be conversational, supportive, and give "
        "brief natural follow-up questions when the candidate's answer is vague. "
        "Keep your replies concise (2-4 sentences max). Never repeat questions already asked."
    ),
    "strict_technical": (
        "You are Rajesh, a strict senior software engineer interviewing for a technical role. "
        "You expect precise, detailed answers. If the candidate is vague, press for specifics. "
        "Be professional but demanding. Keep replies concise (2-4 sentences)."
    ),
    "stress_interviewer": (
        "You are a stress interviewer testing how the candidate handles pressure. "
        "Ask rapid follow-ups, mildly challenge their answers, and create urgency. "
        "Stay professional but intense. Keep replies very short (1-3 sentences)."
    ),
    "placement_panel": (
        "You are a campus placement panel. Speak collectively ('we', 'the panel'). "
        "Be formal and thorough. Cover technical, behavioural, and cultural fit. "
        "Keep replies concise (2-4 sentences)."
    ),
}


class InterviewEngine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.sess = session_store.get(session_id)
        self.persona = self.sess.get("persona", "friendly_hr")
        self.questions: list[str] = self.sess.get("questions", [])
        self.q_index = 0
        self.conversation = []  # local copy for this engine instance

        # Seed conversation with questions context
        self._system = PERSONA_SYSTEM_PROMPTS.get(self.persona, PERSONA_SYSTEM_PROMPTS["friendly_hr"])

    # ── Opening message ────────────────────────────────────────────────────
    async def opening(self) -> dict:
        opening_text = (
            f"Hello! Welcome to your mock interview for the {self.sess.get('role', 'Software Engineer')} position. "
            f"I'm glad you're here. Let's begin. — {self.questions[0]}"
            if self.questions
            else "Welcome! Let's begin your mock interview. Tell me about yourself."
        )
        self.conversation.append({"role": "assistant", "content": opening_text})
        session_store.append_message(self.session_id, "assistant", opening_text)
        self.q_index = 1
        return {"text": opening_text, "question_index": 0, "done": False}

    # ── Respond to candidate ───────────────────────────────────────────────
    async def respond(self, user_text: str) -> dict:
        if not user_text.strip():
            return {"text": "I didn't catch that. Could you please repeat?", "done": False}

        # Save candidate message
        self.conversation.append({"role": "user", "content": user_text})
        session_store.append_message(self.session_id, "user", user_text)

        # Build messages for Claude
        # Inject remaining questions as context for the AI
        remaining = self.questions[self.q_index:] if self.q_index < len(self.questions) else []
        system_with_context = (
            self._system
            + (
                f"\n\nRemaining questions to cover (ask them naturally, one at a time):\n"
                + "\n".join(f"- {q}" for q in remaining)
                if remaining
                else "\n\nYou have covered all prepared questions. Wrap up naturally."
            )
        )

        messages = self.conversation[-14:]  # keep last 14 turns for context

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
                    "max_tokens": 300,
                    "system": system_with_context,
                    "messages": messages,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

        ai_text = data["content"][0]["text"].strip()
        self.conversation.append({"role": "assistant", "content": ai_text})
        session_store.append_message(self.session_id, "assistant", ai_text)

        # Advance question index heuristically
        if self.q_index < len(self.questions):
            self.q_index += 1

        done = self.q_index >= len(self.questions)
        return {"text": ai_text, "question_index": self.q_index, "done": done}

    # ── Score a single answer ──────────────────────────────────────────────
    async def _score_answer(self, question: str, answer: str) -> dict:
        prompt = f"""Rate this interview answer on 4 dimensions (0-10 each). Return ONLY JSON:
{{
  "content_score": 0-10,
  "clarity_score": 0-10,
  "structure_score": 0-10,
  "relevance_score": 0-10,
  "feedback": "1-2 sentence constructive feedback"
}}

Question: {question}
Answer: {answer[:500]}
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
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=20,
            )
        data = resp.json()
        raw = data["content"][0]["text"].replace("```json","").replace("```","").strip()
        try:
            return json.loads(raw)
        except Exception:
            return {"content_score": 5, "clarity_score": 5, "structure_score": 5,
                    "relevance_score": 5, "feedback": "Keep practising!"}

    # ── Finalize session → build full report ───────────────────────────────
    async def finalize(self) -> dict:
        sess = session_store.get(self.session_id)
        messages = sess.get("messages", [])
        snapshots = sess.get("snapshots", [])

        # Extract Q&A pairs
        qa_pairs = []
        q_iter = iter(self.questions)
        user_turns = [m for m in messages if m["role"] == "user"]
        for i, answer in enumerate(user_turns):
            q = next(q_iter, f"Question {i+1}")
            score = await self._score_answer(q, answer["content"])
            qa_pairs.append({"question": q, "answer": answer["content"], **score})

        # Aggregate face snapshot scores
        avg_emotion = _aggregate_emotions(snapshots)
        avg_gaze    = _safe_avg(snapshots, "gaze_score")
        avg_posture = _safe_avg(snapshots, "posture_score")

        # Aggregate answer scores
        avg_content   = _safe_avg(qa_pairs, "content_score")
        avg_clarity   = _safe_avg(qa_pairs, "clarity_score")
        avg_structure = _safe_avg(qa_pairs, "structure_score")

        # Weighted final score
        # 30% content, 15% clarity, 15% structure, 10% pace, 10% eye contact, 10% emotion, 10% posture
        final_score = round(
            avg_content   * 0.30
            + avg_clarity   * 0.15
            + avg_structure * 0.15
            + avg_gaze      * 0.10
            + avg_emotion   * 0.10
            + avg_posture   * 0.10
            + 5             * 0.10,   # pace placeholder (Librosa not wired here)
            1,
        )

        swot = await _generate_swot(qa_pairs, avg_emotion, avg_gaze, avg_posture)

        return {
            "session_id": self.session_id,
            "role": sess.get("role"),
            "persona": self.persona,
            "total_questions": len(self.questions),
            "answers_given": len(user_turns),
            "scores": {
                "content":   round(avg_content, 1),
                "clarity":   round(avg_clarity, 1),
                "structure": round(avg_structure, 1),
                "eye_contact": round(avg_gaze, 1),
                "emotion":   round(avg_emotion, 1),
                "posture":   round(avg_posture, 1),
                "final":     final_score,
            },
            "dominant_emotion": _dominant_emotion(snapshots),
            "swot": swot,
            "qa_breakdown": qa_pairs,
            "snapshot_count": len(snapshots),
        }


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_avg(items: list, key: str, default: float = 5.0) -> float:
    vals = [i.get(key, default) for i in items if i.get(key) is not None]
    return sum(vals) / len(vals) if vals else default


def _aggregate_emotions(snapshots: list) -> float:
    """Map emotion labels to confidence scores."""
    positive_emotions = {"happy", "confident", "neutral", "surprise"}
    scores = []
    for s in snapshots:
        e = s.get("emotion", "neutral")
        scores.append(8.0 if e in positive_emotions else 4.0)
    return sum(scores) / len(scores) if scores else 5.0


def _dominant_emotion(snapshots: list) -> str:
    from collections import Counter
    emotions = [s.get("emotion", "neutral") for s in snapshots]
    if not emotions:
        return "neutral"
    return Counter(emotions).most_common(1)[0][0]


async def _generate_swot(qa_pairs: list, avg_emotion: float, avg_gaze: float, avg_posture: float) -> dict:
    summary_text = "\n".join(
        f"Q: {qa['question'][:80]}\nA: {qa['answer'][:120]}\nScore: {qa.get('content_score',5)}/10"
        for qa in qa_pairs[:6]
    )
    prompt = f"""Based on this mock interview performance, generate a SWOT analysis. Return ONLY JSON:
{{
  "strengths": ["point1", "point2", "point3"],
  "weaknesses": ["point1", "point2", "point3"],
  "opportunities": ["point1", "point2"],
  "threats": ["point1", "point2"],
  "coaching_tips": ["tip1", "tip2", "tip3"]
}}

Q&A Performance:
{summary_text}

Face analysis — avg emotion score: {avg_emotion:.1f}/10, gaze score: {avg_gaze:.1f}/10, posture: {avg_posture:.1f}/10
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
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
    data = resp.json()
    raw = data["content"][0]["text"].replace("```json","").replace("```","").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "strengths": ["Good effort"],
            "weaknesses": ["Needs more practice"],
            "opportunities": ["Online mock interviews"],
            "threats": ["Limited preparation time"],
            "coaching_tips": ["Practice STAR method", "Record yourself", "Read more about your domain"],
        }

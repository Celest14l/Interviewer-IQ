"""
services/session_store.py
Simple in-memory session store.
Swap out for Firebase / Redis in production.
"""

import uuid
import time
from typing import Optional, Dict, Any


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # ── Create ─────────────────────────────────────────────────────────────
    def create(
        self,
        resume_text: str,
        persona: str = "friendly_hr",
        role: str = "Software Engineer",
    ) -> str:
        sid = str(uuid.uuid4())
        self._sessions[sid] = {
            "id": sid,
            "created_at": time.time(),
            "resume_text": resume_text,
            "persona": persona,
            "role": role,
            "messages": [],          # full conversation history
            "snapshots": [],         # face snapshot analysis results
            "audio_scores": [],      # per-turn audio scores
            "report": None,
        }
        return sid

    # ── Read ───────────────────────────────────────────────────────────────
    def get(self, sid: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(sid)

    # ── Append helpers ─────────────────────────────────────────────────────
    def append_message(self, sid: str, role: str, content: str):
        sess = self._sessions.get(sid)
        if sess:
            sess["messages"].append({"role": role, "content": content})

    def append_snapshot(self, sid: str, result: Dict[str, Any]):
        sess = self._sessions.get(sid)
        if sess:
            result["ts"] = time.time()
            sess["snapshots"].append(result)

    def append_audio_score(self, sid: str, score: Dict[str, Any]):
        sess = self._sessions.get(sid)
        if sess:
            sess["audio_scores"].append(score)

    # ── Report ─────────────────────────────────────────────────────────────
    def save_report(self, sid: str, report: Dict[str, Any]):
        sess = self._sessions.get(sid)
        if sess:
            sess["report"] = report

    def get_report(self, sid: str) -> Optional[Dict[str, Any]]:
        sess = self._sessions.get(sid)
        return sess["report"] if sess else None

    # ── Delete ─────────────────────────────────────────────────────────────
    def delete(self, sid: str):
        self._sessions.pop(sid, None)


# Singleton
session_store = SessionStore()

"""
routers/analysis.py
REST endpoints for face snapshot analysis and audio scoring.
The face snapshot endpoint is the key non-streaming alternative to live video.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.face_analyzer import analyze_snapshot
from services.session_store import session_store

router = APIRouter()


# ── Face snapshot (called by frontend every 3-5 s) ─────────────────────────
class SnapshotPayload(BaseModel):
    session_id: str
    image: str          # base64-encoded JPEG/PNG (data-URL or raw base64)


@router.post("/snapshot")
async def face_snapshot(payload: SnapshotPayload):
    """
    Accepts a base64 face image, runs emotion + gaze + posture analysis,
    persists the result in the session, and returns feedback.

    Frontend should call this every 3-5 seconds during the interview.
    """
    sess = session_store.get(payload.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await analyze_snapshot(payload.image, payload.session_id)
    return result


# ── Get all snapshots for a session ───────────────────────────────────────
@router.get("/snapshots/{session_id}")
async def get_snapshots(session_id: str):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    snapshots = sess.get("snapshots", [])
    return {
        "session_id":     session_id,
        "count":          len(snapshots),
        "snapshots":      snapshots,
        "average_scores": _average_snapshot_scores(snapshots),
    }


# ── Audio score (sent after each answer) ──────────────────────────────────
class AudioScorePayload(BaseModel):
    session_id: str
    wpm: Optional[float] = None            # words per minute
    filler_count: Optional[int] = None     # umm, like, you know
    pause_count: Optional[int] = None      # long pauses
    duration_seconds: Optional[float] = None


@router.post("/audio-score")
async def submit_audio_score(payload: AudioScorePayload):
    """
    Frontend can compute basic audio stats (WPM, filler words) via the
    Web Speech API and POST them here. Full Librosa analysis can be added later.
    """
    sess = session_store.get(payload.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    wpm           = payload.wpm or 120
    filler_count  = payload.filler_count or 0
    pause_count   = payload.pause_count or 0

    # Simple scoring
    pace_score    = _score_wpm(wpm)
    filler_penalty = min(filler_count * 0.5, 3.0)
    pause_penalty  = min(pause_count  * 0.3, 2.0)
    vocal_score   = round(max(0, 8.0 - filler_penalty - pause_penalty), 1)

    score_entry = {
        "wpm":         wpm,
        "filler_count": filler_count,
        "pause_count":  pause_count,
        "pace_score":   pace_score,
        "vocal_score":  vocal_score,
    }
    session_store.append_audio_score(payload.session_id, score_entry)
    return score_entry


# ── Helpers ────────────────────────────────────────────────────────────────
def _average_snapshot_scores(snapshots: list) -> dict:
    if not snapshots:
        return {"gaze": 5.0, "posture": 5.0, "emotion_score": 5.0}
    gaze    = sum(s.get("gaze_score",    5) for s in snapshots) / len(snapshots)
    posture = sum(s.get("posture_score", 5) for s in snapshots) / len(snapshots)
    return {
        "gaze":          round(gaze, 1),
        "posture":       round(posture, 1),
    }


def _score_wpm(wpm: float) -> float:
    """Ideal WPM for interviews: 120-160. Penalise too fast or too slow."""
    if 120 <= wpm <= 160:
        return 9.0
    elif 100 <= wpm < 120 or 160 < wpm <= 180:
        return 7.0
    elif 80 <= wpm < 100 or 180 < wpm <= 200:
        return 5.0
    else:
        return 3.0

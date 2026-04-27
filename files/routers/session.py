"""
routers/session.py
Session management: retrieve reports, list sessions, delete sessions.
"""

from fastapi import APIRouter, HTTPException
from services.session_store import session_store

router = APIRouter()


@router.get("/{session_id}")
async def get_session(session_id: str):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    # Return session metadata (not full messages for brevity)
    return {
        "session_id":    sess["id"],
        "created_at":    sess["created_at"],
        "persona":       sess["persona"],
        "role":          sess["role"],
        "message_count": len(sess["messages"]),
        "snapshot_count": len(sess["snapshots"]),
        "has_report":    sess["report"] is not None,
    }


@router.get("/{session_id}/report")
async def get_report(session_id: str):
    report = session_store.get_report(session_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not ready. End the interview first.")
    return report


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    session_store.delete(session_id)
    return {"deleted": session_id}


@router.get("/{session_id}/transcript")
async def get_transcript(session_id: str):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "messages":   sess.get("messages", []),
    }

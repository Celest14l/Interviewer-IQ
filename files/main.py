"""
InterviewIQ — AI Interview Coach Backend
FastAPI server: handles resume upload, AI interview conversation,
face snapshot analysis (every 3-5s), scoring, and SWOT report.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
import asyncio

from routers import interview, analysis, session
from services.session_store import session_store

app = FastAPI(
    title="InterviewIQ API",
    description="AI-powered mock interview backend with face snapshot analysis",
    version="1.0.0"
)

# ── CORS (allow your HTML frontend origin) ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(interview.router, prefix="/api/interview", tags=["Interview"])
app.include_router(analysis.router,  prefix="/api/analysis",  tags=["Analysis"])
app.include_router(session.router,   prefix="/api/session",   tags=["Session"])


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "InterviewIQ"}


# ── WebSocket: live interview conversation ─────────────────────────────────
@app.websocket("/ws/interview/{session_id}")
async def interview_ws(websocket: WebSocket, session_id: str):
    """
    Bidirectional WebSocket for the live interview.
    Client sends:
      { "type": "audio_chunk", "data": "<base64-wav>" }   – speech input
      { "type": "snapshot",    "data": "<base64-jpeg>" }  – face snapshot (every 3-5 s)
      { "type": "end_interview" }
    Server sends:
      { "type": "ai_reply",  "text": "...", "audio_url": "..." }
      { "type": "snapshot_result", "emotion": "...", "gaze": "...", "posture": "..." }
      { "type": "session_ended", "report_url": "/api/session/{id}/report" }
    """
    await websocket.accept()
    sess = session_store.get(session_id)
    if not sess:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    from services.interview_engine import InterviewEngine
    from services.face_analyzer import analyze_snapshot

    engine = InterviewEngine(session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            kind = msg.get("type")

            # ── Face snapshot (every 3-5 s from frontend) ──────────────────
            if kind == "snapshot":
                result = await analyze_snapshot(msg["data"], session_id)
                await websocket.send_json({"type": "snapshot_result", **result})

            # ── Transcribed text / audio chunk ─────────────────────────────
            elif kind == "user_message":
                text = msg.get("text", "")
                reply = await engine.respond(text)
                await websocket.send_json({
                    "type": "ai_reply",
                    "text": reply["text"],
                    "question_index": reply.get("question_index", 0),
                    "done": reply.get("done", False)
                })

            # ── End interview ───────────────────────────────────────────────
            elif kind == "end_interview":
                report = await engine.finalize()
                session_store.save_report(session_id, report)
                await websocket.send_json({
                    "type": "session_ended",
                    "report_url": f"/api/session/{session_id}/report"
                })
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

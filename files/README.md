# InterviewIQ — Backend

FastAPI backend for the InterviewIQ AI Interview Coach.
Provides REST + WebSocket APIs for your HTML frontend.

---

## Architecture Overview

```
Frontend (HTML/JS)
      │
      ├── POST /api/interview/start          → Upload resume, get session_id + questions
      ├── GET  /api/interview/{id}/open      → Get opening AI message
      │
      ├── POST /api/interview/{id}/message   → Send candidate text, get AI reply (HTTP mode)
      │   OR
      ├── WS   /ws/interview/{id}            → Live bidirectional interview (WebSocket mode)
      │
      ├── POST /api/analysis/snapshot        → Face snapshot every 3-5 s (base64 JPEG)
      ├── POST /api/analysis/audio-score     → Submit WPM / filler word stats
      │
      ├── POST /api/interview/{id}/end       → Finalize, generate SWOT report
      └── GET  /api/session/{id}/report      → Retrieve full report
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variable

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run

```bash
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## Key Design Decisions

### ✅ Snapshots instead of video streaming

The frontend captures a JPEG snapshot from the webcam **every 3-5 seconds** and POSTs it to:

```
POST /api/analysis/snapshot
{ "session_id": "...", "image": "<base64-jpeg>" }
```

This is far more bandwidth-efficient than streaming video and works over any connection.
Each snapshot is analysed for:
- **Emotion** (DeepFace: 7 classes — happy, sad, fear, angry, disgust, surprise, neutral)
- **Gaze** (MediaPipe iris landmarks — how centred the eyes are = eye contact proxy)
- **Posture** (MediaPipe pose — shoulder tilt + slouch detection)

Results are persisted per session and aggregated into the final report.

### ✅ Graceful degradation

If `deepface` or `mediapipe` are not installed, the analyzer returns neutral heuristic scores (5/10) instead of crashing. The interview still works — you just won't have face analytics.

### ✅ HTTP + WebSocket modes

- **Simple HTML frontend**: Use the REST endpoints (`/message`, `/snapshot`)
- **Advanced frontend**: Use the WebSocket at `/ws/interview/{session_id}` for lower latency

---

## API Reference

### POST `/api/interview/start`

| Field    | Type   | Description                                    |
|----------|--------|------------------------------------------------|
| resume   | File   | PDF resume (max 5 MB)                          |
| persona  | string | `friendly_hr` / `strict_technical` / `stress_interviewer` / `placement_panel` |
| role     | string | Job role (e.g., "Software Engineer")           |

**Response:**
```json
{
  "session_id": "uuid",
  "parsed_resume": { "name": "...", "skills": [...], ... },
  "questions": ["Q1", "Q2", ...],
  "ws_url": "/ws/interview/uuid"
}
```

---

### POST `/api/interview/{session_id}/message`

```json
{ "text": "I worked on a distributed caching system using Redis..." }
```

**Response:**
```json
{ "text": "Interesting! How did you handle cache invalidation?", "question_index": 2, "done": false }
```

---

### POST `/api/analysis/snapshot`

```json
{
  "session_id": "uuid",
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**
```json
{
  "emotion": "confident",
  "emotion_confidence": 7.2,
  "gaze_score": 8.1,
  "posture_score": 6.5,
  "feedback": "Looking great — keep it up!"
}
```

Call this every **3-5 seconds** from your frontend JavaScript:

```js
async function sendSnapshot() {
  const canvas = document.createElement('canvas');
  canvas.width = 320; canvas.height = 240;
  canvas.getContext('2d').drawImage(videoElement, 0, 0, 320, 240);
  const b64 = canvas.toDataURL('image/jpeg', 0.7);

  await fetch('http://localhost:8000/api/analysis/snapshot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, image: b64 })
  });
}

setInterval(sendSnapshot, 4000);   // every 4 seconds
```

---

### POST `/api/interview/{session_id}/end`

Finalizes the interview and generates the SWOT report.

---

### GET `/api/session/{session_id}/report`

Returns the full report:
```json
{
  "scores": {
    "content": 7.2,
    "clarity": 6.8,
    "structure": 7.0,
    "eye_contact": 8.1,
    "emotion": 7.5,
    "posture": 6.5,
    "final": 7.2
  },
  "dominant_emotion": "neutral",
  "swot": {
    "strengths": [...],
    "weaknesses": [...],
    "opportunities": [...],
    "threats": [...],
    "coaching_tips": [...]
  },
  "qa_breakdown": [...]
}
```

---

## WebSocket Protocol

Connect to `ws://localhost:8000/ws/interview/{session_id}`

**Client → Server:**
```json
{ "type": "user_message", "text": "My answer here" }
{ "type": "snapshot", "data": "<base64-jpeg>" }
{ "type": "end_interview" }
```

**Server → Client:**
```json
{ "type": "ai_reply", "text": "...", "question_index": 2, "done": false }
{ "type": "snapshot_result", "emotion": "...", "gaze_score": 8.1, "posture_score": 7.2, "feedback": "..." }
{ "type": "session_ended", "report_url": "/api/session/{id}/report" }
```

---

## Scoring Weights

| Dimension    | Weight |
|-------------|--------|
| Content      | 30%    |
| Clarity      | 15%    |
| Structure    | 15%    |
| Pace/Vocal   | 10%    |
| Eye Contact  | 10%    |
| Emotion      | 10%    |
| Posture      | 10%    |

---

## Environment Variables

| Variable           | Description                      |
|--------------------|----------------------------------|
| `ANTHROPIC_API_KEY` | Your Claude API key (required)  |

---

## Production Notes

- Replace in-memory `session_store` with Firebase / Redis
- Add rate limiting (slowapi)
- Add JWT auth
- Use Librosa for full audio analysis (speech pace, pitch)
- Consider ElevenLabs TTS for AI voice responses

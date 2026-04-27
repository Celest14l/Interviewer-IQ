# InterviewIQ — AI Interview Coach

InterviewIQ is a production-oriented AI interview coaching platform. Candidates upload their resume, conduct a live voice interview with an adaptive AI interviewer, and receive a detailed performance report with scores and coaching tips.

---

## Features

- **Real user accounts** — sign up, log in, and own your interviews
- **Resume-grounded interviews** — Gemini analyses your resume and builds a structured interview blueprint with primary questions, follow-up probes, and challenge probes
- **Adaptive live interview** — the AI interviewer decides whether to follow up, challenge, clarify, switch topics, or wrap based on your answers
- **Voice-first** — speak your answers; Groq Whisper transcribes them in real time and Cartesia voices the interviewer
- **Non-verbal analytics** — optional face-snapshot analysis (emotion, gaze, posture) via DeepFace and MediaPipe
- **SWOT report** — final report with per-answer scores, dimension breakdowns, strengths, weaknesses, opportunities, threats, and coaching tips

---

## Architecture Overview

```
Browser (frontend/index.html, frontend/interview.html)
      │
      ├── POST /api/auth/signup|login|logout   → User authentication
      ├── POST /api/interviews/bootstrap       → Upload resume → persisted interview + blueprint
      │
      ├── WS   /api/realtime/{interview_id}    → Live bidirectional interview
      │         ├── user_text / user_audio     →  candidate turn (text or base64 audio)
      │         ├── ai_reply                   ← interviewer text + audio URL
      │         └── end_interview              → triggers report synthesis
      │
      ├── POST /api/analysis/snapshot          → Face snapshot (base64 JPEG) every ~4 s
      ├── POST /api/audio/transcribe           → STT (Groq Whisper)
      ├── GET  /api/audio/synthesize           → TTS audio download (Cartesia)
      │
      ├── GET  /api/interviews/{id}/scores     → Per-answer score breakdown
      └── GET  /api/interviews/{id}/report     → Full SWOT report
```

---

## Tech Stack

### Backend

| Layer | Technology |
|---|---|
| Framework | FastAPI 0.111 |
| Database | PostgreSQL (SQLAlchemy + Alembic) |
| Cache / Realtime coordination | Redis |
| Auth | JWT (HS256) + bcrypt |

### AI / Speech Providers

| Role | Provider / Model |
|---|---|
| Adaptive reasoning (blueprint, turn analysis, scoring) | Gemini `gemini-2.5-flash` |
| High-trust async evaluation (final report) | Gemini `gemini-2.5-pro` |
| Live interviewer surface | NVIDIA NIM `nvidia/nemotron-mini-4b-instruct` |
| Speech-to-Text | Groq `whisper-large-v3-turbo` |
| Text-to-Speech | Cartesia `sonic-3` |
| Content moderation | NVIDIA NIM `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` |

### Face Analytics (optional)

- **Emotion** — DeepFace (7 classes: happy, sad, fear, angry, disgust, surprise, neutral)
- **Gaze** — MediaPipe iris landmarks (eye-contact proxy)
- **Posture** — MediaPipe pose (shoulder tilt + slouch detection)

If DeepFace or MediaPipe are not installed the analyzer returns neutral fallback scores (5/10) and the interview continues normally.

### Frontend

- Plain HTML + JavaScript (`frontend/index.html`, `frontend/interview.html`)
- No build step required

---

## Quick Start

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Celest14l/Interviewer-IQ.git
cd Interviewer-IQ/files
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create a `.env` file

```bash
cp .env.example .env   # or create .env manually
```

Fill in the required values (see [Environment Variables](#environment-variables) below).

### 3. Run database migrations

```bash
alembic upgrade head
```

### 4. Start the backend

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

### 5. Open the frontend

Serve `frontend/` with any static file server, for example:

```bash
cd ../frontend
python -m http.server 5500
```

Then open http://localhost:5500 in your browser.

---

## Repository Structure

```
Interviewer-IQ/
├── frontend/
│   ├── index.html          # Landing / setup screen
│   └── interview.html      # Live interview page
├── files/
│   ├── app/
│   │   ├── main.py         # FastAPI application factory
│   │   ├── core/           # Settings, security helpers
│   │   ├── db/             # SQLAlchemy engine, session, ORM models, Alembic
│   │   ├── api/
│   │   │   └── routers/    # auth, interviews, realtime, analysis, audio, health
│   │   ├── services/       # Deterministic business logic
│   │   ├── agents/         # Bounded agentic modules (blueprint, turn analysis, scoring)
│   │   ├── providers/      # Gemini, NVIDIA NIM, Groq, Cartesia adapters
│   │   └── schemas/        # Pydantic request / response schemas
│   ├── migrations/         # Alembic migration scripts
│   ├── requirements.txt
│   └── main.py             # Top-level entrypoint (delegates to app/)
└── docs/
    ├── IMPLEMENTATION_TRACKER.md
    └── IMPLEMENTATION_PROGRESS.md
```

---

## API Reference

### Auth

| Method | Path | Description |
|---|---|---|
| POST | `/api/auth/signup` | Create a new account |
| POST | `/api/auth/login` | Log in, receive JWT |
| POST | `/api/auth/logout` | Invalidate session |

### Interviews

| Method | Path | Description |
|---|---|---|
| POST | `/api/interviews/bootstrap` | Upload resume PDF → create interview + blueprint |
| GET | `/api/interviews/` | List current user's interviews |
| GET | `/api/interviews/{id}/scores` | Per-answer score breakdown |
| GET | `/api/interviews/{id}/report` | Full SWOT report |

**Bootstrap request (multipart/form-data):**

| Field | Type | Description |
|---|---|---|
| `resume` | File | PDF resume (max 5 MB) |
| `role` | string | Target job role (e.g. `"Software Engineer"`) |
| `persona` | string | `friendly_hr` / `strict_technical` / `stress_interviewer` / `placement_panel` |

**Bootstrap response:**
```json
{
  "interview_id": "uuid",
  "ws_url": "/api/realtime/uuid"
}
```

### Realtime WebSocket

Connect to `ws://localhost:8000/api/realtime/{interview_id}?token=<jwt>`

**Client → Server:**
```json
{ "type": "user_text",   "text": "My answer here" }
{ "type": "user_audio",  "data": "<base64-wav>" }
{ "type": "snapshot",    "data": "<base64-jpeg>" }
{ "type": "end_interview" }
```

**Server → Client:**
```json
{ "type": "ai_reply",        "text": "...", "audio_url": "/api/audio/synthesize?..." }
{ "type": "snapshot_result", "emotion": "neutral", "gaze_score": 8.1, "posture_score": 7.2, "feedback": "..." }
{ "type": "session_ended",   "report_url": "/api/interviews/{id}/report" }
```

### Audio

| Method | Path | Description |
|---|---|---|
| POST | `/api/audio/transcribe` | Transcribe audio with Groq Whisper (protected) |
| GET | `/api/audio/synthesize` | Download interviewer voice audio from Cartesia (protected) |

### Analysis

| Method | Path | Description |
|---|---|---|
| POST | `/api/analysis/snapshot` | Submit a webcam snapshot for non-verbal analysis (protected) |

### Health

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |

---

## WebSocket Live Interview Loop

1. Candidate speaks → browser records audio
2. `user_audio` frame sent over WebSocket (base64 WAV)
3. Server transcribes with **Groq Whisper**
4. Transcript is moderated and persisted
5. **Gemini 2.5 Flash** (turn analyzer) decides: follow up / challenge / clarify / switch topic / wrap
6. **NVIDIA Nemotron** renders the next interviewer utterance naturally
7. Response is moderated
8. **Cartesia** generates interviewer voice audio
9. `ai_reply` frame returned to client with text + audio URL

Background (off the hot path): answer scoring, snapshot analytics, vocal analytics, final report synthesis.

---

## Scoring Weights

| Dimension | Weight |
|---|---|
| Content | 30% |
| Clarity | 15% |
| Structure | 15% |
| Pace / Vocal | 10% |
| Eye Contact | 10% |
| Emotion | 10% |
| Posture | 10% |

---

## Environment Variables

Create a `.env` file in the `files/` directory:

| Variable | Required | Description |
|---|---|---|
| `SECRET_KEY` | ✅ | Secret for signing JWT tokens |
| `DATABASE_URL` | ✅ | PostgreSQL connection string (e.g. `postgresql+psycopg://user:pass@host/db`) |
| `REDIS_URL` | ✅ | Redis connection string (e.g. `redis://localhost:6379/0`) |
| `GEMINI_API_KEY` | ✅ | Google AI / Gemini API key |
| `GROQ_API_KEY` | ✅ | Groq API key (STT) |
| `CARTESIA_API_KEY` | ✅ | Cartesia API key (TTS) |
| `NVIDIA_API_KEY` | ✅ | NVIDIA NIM API key (live interviewer + moderation) |
| `APP_ENV` | | `development` / `staging` / `production` (default: `development`) |
| `GEMINI_REASONING_MODEL` | | Override Gemini model (default: `gemini-2.5-flash`) |
| `NVIDIA_LIVE_MODEL` | | Override NIM model (default: `nvidia/nemotron-mini-4b-instruct`) |
| `GROQ_STT_MODEL` | | Override Whisper model (default: `whisper-large-v3-turbo`) |
| `CARTESIA_MODEL_ID` | | Override Cartesia model (default: `sonic-3`) |
| `UPLOAD_DIR` | | Resume upload storage path (default: `storage/uploads`) |

---

## Production Checklist

- [ ] Set `APP_ENV=production` and a strong `SECRET_KEY`
- [ ] Use a managed PostgreSQL service (e.g. Supabase, RDS, Neon)
- [ ] Use a managed Redis service (e.g. Upstash, ElastiCache)
- [ ] Run behind a reverse proxy (nginx / Caddy) with TLS
- [ ] Add rate limiting (`slowapi`)
- [ ] Enable structured logging and an observability backend (e.g. Grafana, Datadog)
- [ ] Set up Alembic auto-migrations in your CI/CD pipeline


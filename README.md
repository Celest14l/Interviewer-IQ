# InterviewIQ — AI-Powered Multimodal Interview Coaching Platform

InterviewIQ is a production-grade AI interview coaching platform that combines Large Language Models, Speech AI, and Computer Vision to simulate realistic interviews and provide actionable feedback. Candidates upload their resume, participate in adaptive voice interviews, and receive detailed SWOT-based performance reports with personalized coaching recommendations.

---

# Features

- Resume-grounded interview generation using Gemini
- Adaptive AI interviewer with contextual follow-up questions
- Real-time voice interviews with speech-to-text and text-to-speech
- Snapshot-based behavioral analysis every ~5 seconds
- Emotion detection, eye-contact tracking, gaze estimation, and posture analysis
- JWT-based authentication and interview history management
- Real-time WebSocket communication
- SWOT-based performance reports
- Personalized coaching roadmap

---

# Architecture Overview

```text
Frontend (HTML + JavaScript)
            │
            ▼
      FastAPI Backend
            │
 ┌──────────┼──────────┐
 ▼          ▼          ▼
PostgreSQL  JWT Auth  WebSockets
            │
            ▼
      Interview Engine
            │
 ┌──────────┼──────────┬──────────┐
 ▼          ▼          ▼          ▼
Gemini   Whisper   DeepFace   MediaPipe
 (LLM)      STT     Emotion      CV
            │
            ▼
      Evaluation Engine
            │
            ▼
      SWOT Report Engine
            │
            ▼
      Coaching Dashboard
```

---

# Tech Stack

## Backend
- FastAPI
- Supabase
- SQLAlchemy
- Alembic
- JWT Authentication
- WebSockets

## AI & Computer Vision
- Gemini 2.5 Flash
- Gemini 2.5 Pro
- DeepFace
- MediaPipe
- NVIDIA Nemotron

## Frontend
- HTML
- JavaScript

---

# AI / Speech Providers

| Purpose | Provider |
|----------|----------|
| Interview Blueprint Generation | Gemini 2.5 Flash |
| Adaptive Questioning | Gemini 2.5 Flash |
| Final Evaluation & SWOT Report | Gemini 2.5 Pro |
| Live Interview Agent | NVIDIA Nemotron |
| Speech-to-Text | Groq Whisper Large V3 Turbo |
| Text-to-Speech | Cartesia Sonic-3 |
| Content Moderation | NVIDIA Safety Guard |

---

# Behavioral Analytics

During an interview session, InterviewIQ captures snapshots approximately every **5 seconds** and performs:

- Emotion Recognition (7 emotion classes)
- Eye Contact Analysis
- Gaze Estimation
- Posture Stability Analysis

These metrics are combined with verbal communication and answer quality scores to generate comprehensive interview feedback.

---

# Scoring Framework

| Dimension | Weight |
|------------|----------|
| Content Quality | 30% |
| Communication Clarity | 15% |
| Answer Structure | 15% |
| Speech Pace | 10% |
| Emotional Composure | 10% |
| Eye Contact | 10% |
| Posture Stability | 10% |

---

# Results

The platform was evaluated using **30 participants** across multiple interview sessions.

### Improvements Achieved

- 32% improvement in overall interview scores
- 38.8% improvement in answer structure
- 38.3% improvement in eye-contact metrics
- Significant gains in confidence and communication clarity

---

# Core APIs

### Authentication
- POST `/api/auth/signup`
- POST `/api/auth/login`
- POST `/api/auth/logout`

### Interviews
- POST `/api/interviews/bootstrap`
- GET `/api/interviews/{id}/scores`
- GET `/api/interviews/{id}/report`

### Audio
- POST `/api/audio/transcribe`
- GET `/api/audio/synthesize`

### Analysis
- POST `/api/analysis/snapshot`

### Health
- GET `/health`

---

# Quick Start

## Clone Repository

```bash
git clone https://github.com/Celest14l/Interviewer-IQ.git
cd Interviewer-IQ/files
```

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

```bash
cp .env.example .env
```

Fill in:

```env
SECRET_KEY=
DATABASE_URL=
GEMINI_API_KEY=
GROQ_API_KEY=
CARTESIA_API_KEY=
NVIDIA_API_KEY=
```

## Run Database Migrations

```bash
alembic upgrade head
```

## Start Backend

```bash
uvicorn app.main:app --reload --port 8000
```

## Open API Documentation

```text
http://localhost:8000/docs
```

---

# Future Enhancements

- Multi-language interview support
- Group discussion simulation
- Mobile application
- Institution/TPO integration
- Advanced analytics dashboard
- CI/CD deployment pipeline
- Monitoring and observability integration

---

# Impact

InterviewIQ bridges the gap between traditional interview preparation and personalized coaching by combining cloud infrastructure, speech AI, computer vision, and large language models into a unified platform that delivers measurable improvements in interview performance.

# InterviewIQ — AI-Powered Multimodal Interview Coaching Platform

InterviewIQ is a cloud-hosted AI interview coaching platform that conducts adaptive voice interviews, evaluates verbal and non-verbal communication, and generates detailed SWOT-based performance reports.

---

# Features

- Resume-grounded interview generation using Gemini
- Adaptive AI interviewer with dynamic follow-up questioning
- Real-time voice interviews with STT and TTS integration
- Snapshot-based behavioral analysis every ~5 seconds
- Emotion, eye-contact, gaze, and posture tracking
- JWT authentication and interview history management
- SWOT reports with personalized coaching recommendations
- Real-time WebSocket communication
- Cloud deployment on AWS EC2

---

# Architecture Overview

```text
Frontend (HTML/JavaScript)
            │
            ▼
      FastAPI Backend
            │
 ┌──────────┼──────────┐
 ▼          ▼          ▼
PostgreSQL  WebSocket  JWT Auth
            │
            ▼
     Interview Engine
            │
 ┌──────────┼──────────┬──────────┐
 ▼          ▼          ▼          ▼
Gemini   Whisper   DeepFace   MediaPipe
(LLM)      STT     Emotion     CV Analytics
            │
            ▼
      SWOT Report Engine
            │
            ▼
      Coaching Dashboard

Tech Stack
Cloud & Infrastructure
AWS EC2
Docker
Linux
Git
Backend
FastAPI
PostgreSQL
SQLAlchemy
Alembic
JWT Authentication
WebSockets
AI & Computer Vision
Gemini 2.5 Flash / Pro
DeepFace
MediaPipe
NVIDIA Nemotron
Frontend
HTML
JavaScript
AI / Speech Providers
Purpose	Provider
Interview Generation	Gemini 2.5 Flash
Final Evaluation	Gemini 2.5 Pro
Live Interview Agent	NVIDIA Nemotron
Speech-to-Text	Groq Whisper Large V3 Turbo
Text-to-Speech	Cartesia Sonic-3
Content Moderation	NVIDIA Safety Guard
Results
32% improvement in overall interview scores
38.8% improvement in answer structure
38.3% improvement in eye-contact metrics
Tested on 30 participants across multiple interview sessions
Quick Start
git clone https://github.com/Celest14l/Interviewer-IQ.git
cd Interviewer-IQ/files

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

alembic upgrade head

uvicorn app.main:app --reload --port 8000

Open:

http://localhost:8000/docs

for API documentation.


This structure is what recruiters and judges typically look for:
**Features → Architecture → Tech Stack → AI Providers → Results → Quick Start**. It highlights both the cloud engineering 

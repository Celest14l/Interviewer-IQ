# InterviewIQ Implementation Tracker

Last updated: 2026-04-25

This document is the living source of truth for what we are building, what has been decided, and what changes as implementation evolves.

## Product Goal

Turn the current InterviewIQ prototype into a production-functional interview platform with:

- real authentication
- persistent interview state
- real-time interview orchestration
- real STT/TTS
- trustworthy scoring and reporting
- no mock, demo, or simulated production paths

## Chosen Stack

### Core app

- Frontend: existing `frontend/` client, to be refactored away from demo logic into a real backend-driven client
- Backend: FastAPI
- Primary database: hosted PostgreSQL
- Realtime coordination: Redis

### AI / speech providers

- STT: Groq `whisper-large-v3-turbo`
- TTS: Cartesia
- Live interviewer surface model: NVIDIA NIM `nvidia/nemotron-mini-4b-instruct`
- Adaptive reasoning model: Gemini `gemini-2.5-flash`
- High-trust async reasoning upgrade path: Gemini `gemini-2.5-pro`
- Safety / moderation model: NVIDIA NIM `nvidia/llama-3.1-nemotron-safety-guard-8b-v3`

## Model Routing Decisions

### Live interviewer surface

We are currently using:

- `nvidia/nemotron-mini-4b-instruct`

Role:

- render the interviewer in a natural, low-latency conversational voice
- phrase follow-ups naturally
- maintain interviewer persona during the live session

Important:

- this model is no longer the main reasoning brain
- it is the conversational surface for the interviewer

### Adaptive reasoning brain

We are currently using:

- `gemini-2.5-flash`

Role:

- resume grounding refinement
- interview blueprint generation
- turn analysis
- adaptive follow-up decisioning
- primary answer scoring

Reason:

- this is the main low-latency reasoning model in the system
- it is better suited than the small interviewer model for connecting resume claims, deciding when to probe deeper, and keeping the interview adaptive

### High-trust async upgrade path

We may use:

- `gemini-2.5-pro`

Role:

- final report generation
- verification of borderline or inconsistent answer scoring
- difficult contradiction-resolution cases

Important:

- this model is not intended to sit on the hot realtime path by default
- it is reserved for higher-trust asynchronous evaluation work where needed

### Moderation

- `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` checks candidate input and assistant output

## Architectural Direction

### Source of truth

PostgreSQL is the canonical store for:

- users
- interviews
- transcripts
- scores
- reports
- uploads
- analytics metadata

Redis is only for:

- WebSocket presence
- active interview locks
- heartbeats
- rate limiting
- short-lived ephemeral coordination

### Realtime transport

We are standardizing on:

- WebSocket for live interview flow

REST remains for:

- auth
- resume upload
- session bootstrap
- report retrieval
- admin/support endpoints

### Architecture style

We are building:

- a deterministic realtime platform
- with a bounded adaptive reasoning layer

This means:

- infrastructure, persistence, transport, auth, and provider adapters stay deterministic
- only the reasoning-heavy parts of the interview process are agentic

We are not building:

- a free-roaming multi-agent runtime
- a fully autonomous swarm architecture

## Current Problems To Eliminate

- frontend demo mode
- simulated face results
- fake live scoring in the client
- fake report fallback
- in-memory session store
- loosely coupled turn state
- silent fallback behavior that hides provider failure

## Final Functional Architecture

### Deterministic modules

- auth and session management
- resume upload and file management
- PostgreSQL persistence layer
- Redis presence and locks
- WebSocket gateway
- Groq STT adapter
- Cartesia TTS adapter
- moderation adapter
- analytics ingestion
- report storage and retrieval

### Agentic modules

These are the only places where agentic reasoning is intentionally used:

- interview blueprint builder
- turn analyzer
- answer scorer
- score verifier
- report synthesizer

### Interview blueprint concept

The system will not operate on a flat list of static questions.

Instead, it will build an interview blueprint containing:

- resume-anchored topics
- primary questions
- follow-up probes
- challenge probes
- ownership probes
- contradiction probes
- completion criteria

This blueprint is the structured plan that drives the adaptive interview.

### Live interview loop

1. user speaks
2. Groq STT finalizes transcript
3. transcript is moderated
4. transcript is persisted
5. turn analyzer decides whether to follow up, challenge, clarify, switch topic, or wrap topic
6. `nvidia/nemotron-mini-4b-instruct` renders the next interviewer utterance
7. response is moderated
8. Cartesia generates interviewer audio
9. response is sent to client

### Background evaluation loop

These tasks should not block the realtime interview path:

- rubric-based answer scoring
- vocal analytics
- snapshot analytics
- score verification
- final report synthesis

### Latency strategy

To keep the interview feeling responsive:

- keep the realtime path small
- use `gemini-2.5-flash` for reasoning in the live control loop
- use `nvidia/nemotron-mini-4b-instruct` only for conversational rendering
- push heavy verification and reporting work off the hot path

## Implementation Phases

### Phase 1: Foundation

- add config/settings layer
- add hosted Postgres integration
- add Redis integration
- add DB models and migrations
- remove in-memory session storage
- add auth and per-user ownership

### Phase 2: Realtime Interview Core

- define WebSocket event protocol
- add authenticated socket connections
- persist transcript turns
- store interview state in DB
- add Redis presence and interview locks

### Phase 3: Provider Integrations

- Groq STT adapter
- Cartesia TTS adapter
- NVIDIA live interviewer adapter
- Gemini reasoning adapter
- NVIDIA moderation adapter

### Phase 4: Scoring and Reporting

- rubric-based answer scoring
- answer-level score persistence
- final report aggregation
- trustworthy report generation from structured inputs
- optional Gemini Pro verifier path for high-trust async evaluation

### Phase 5: Analytics

- vocal analytics pipeline
- snapshot analytics cleanup
- coaching cues based on real data

### Phase 6: Hardening

- retries and timeouts
- observability
- rate limiting
- error handling
- end-to-end tests

## Implementation Workstreams

### Workstream 1: Backend foundation

Goal:

- convert the current prototype backend into a layered production backend

Deliverables:

- centralized settings/config module
- application startup wiring
- database engine and session management
- Redis client wiring
- dependency injection helpers
- health and readiness endpoints

Planned repo shape:

- `files/app/core/` for settings, security, logging, config
- `files/app/db/` for engine, sessions, models, migrations wiring
- `files/app/api/` for routers
- `files/app/services/` for deterministic services
- `files/app/agents/` for bounded reasoning modules
- `files/app/providers/` for Groq, Cartesia, Gemini, NIM adapters
- `files/app/schemas/` for Pydantic request/response and internal schemas

### Workstream 2: Persistence and data model

Goal:

- replace in-memory session state with durable relational state

Deliverables:

- SQLAlchemy models
- Alembic migrations
- repositories or service-layer persistence helpers

Initial entities:

- `users`
- `auth_sessions`
- `uploaded_files`
- `interviews`
- `interview_blueprints`
- `transcript_turns`
- `answer_scores`
- `final_reports`
- `snapshot_events`
- `vocal_analytics`
- `moderation_events`

### Workstream 3: Authentication and access control

Goal:

- make every interview session user-owned and authenticated

Deliverables:

- signup/login/logout
- password hashing
- browser auth session
- protected API routes
- protected WebSocket auth

### Workstream 4: Realtime interview core

Goal:

- make WebSocket the single live interview path

Deliverables:

- event protocol
- socket auth
- interview heartbeat
- Redis presence
- active interview locking
- transcript turn persistence

### Workstream 5: Provider integration layer

Goal:

- isolate all external AI/audio providers behind stable interfaces

Deliverables:

- Groq STT adapter
- Cartesia TTS adapter
- Gemini reasoning adapter
- NIM interviewer adapter
- NIM safety adapter

Rule:

- routers and controllers should never call provider APIs directly

### Workstream 6: Adaptive reasoning layer

Goal:

- implement the bounded agentic modules that actually drive adaptiveness

Deliverables:

- interview blueprint builder
- turn analyzer
- answer scorer
- score verifier
- report synthesizer

### Workstream 7: Frontend rewrite

Goal:

- convert `frontend/interview.html` from a prototype page into a real backend-driven client

Deliverables:

- remove demo mode
- remove simulated face results
- remove fake score calculation as truth source
- add real WebSocket client
- add real upload/bootstrap flow
- add explicit loading, error, retry, and reconnect states

### Workstream 8: Evaluation and hardening

Goal:

- make the system trustworthy and testable

Deliverables:

- prompt versioning
- replay fixtures
- scoring consistency tests
- provider failure tests
- end-to-end interview tests

## Repository Refactor Plan

### Current prototype files to retire or replace

- `files/services/session_store.py`
- `files/services/interview_engine.py`
- `files/services/resume_parser.py`
- `files/routers/interview.py`
- `files/routers/analysis.py`
- `files/routers/session.py`

These prototype files are useful as reference, but they should not remain the long-term architecture.

### Transitional approach

We should not rewrite everything in one move.

Instead:

1. create the new application structure alongside the existing prototype files
2. migrate one concern at a time
3. switch routers to the new services incrementally
4. remove prototype files only after equivalent production paths exist

## Concrete Build Order

### Sprint 1: Foundation and persistence

Build first:

- `app/core/settings.py`
- `app/db/base.py`
- `app/db/session.py`
- SQLAlchemy models for users, interviews, transcripts, reports
- Alembic setup
- Postgres wiring
- Redis wiring

Outcome:

- backend can boot with real config and real persistence

### Sprint 2: Auth and protected sessions

Build next:

- auth schemas
- auth service
- auth router
- browser session handling
- current-user dependency
- protected interview ownership rules

Outcome:

- interviews become real user-owned resources

### Sprint 3: Interview bootstrap path

Build next:

- upload/file abstraction
- resume ingestion service
- Gemini grounding adapter
- blueprint schema
- blueprint builder
- interview creation API

Outcome:

- uploading a resume creates a real persisted interview and blueprint

### Sprint 4: WebSocket interview loop

Build next:

- authenticated WebSocket gateway
- client event schema
- Redis presence and interview lock
- transcript turn persistence
- turn analyzer integration
- Nemotron response rendering

Outcome:

- live interview works without fake frontend logic

### Sprint 5: STT and TTS

Build next:

- Groq STT adapter
- chunk handling
- transcript finalization logic
- Cartesia TTS adapter
- audio response flow

Outcome:

- voice interview loop becomes functional end to end

### Sprint 6: Scoring and reporting

Build next:

- rubric schema
- answer scorer
- score verifier
- final report synthesizer
- report retrieval endpoints

Outcome:

- interviews produce auditable scores and reports

### Sprint 7: Frontend productionization

Build next:

- remove demo mode from `frontend/interview.html`
- move to real socket state
- wire upload/auth/bootstrap
- render transcript, reconnect, failure states, and final report

Outcome:

- frontend becomes an actual product surface rather than a prototype shell

### Sprint 8: Hardening

Build next:

- structured logs
- timeout policy
- retry policy
- rate limiting
- observability
- end-to-end tests

Outcome:

- system becomes production-shaped and testable

## Immediate Implementation Plan

This is the recommended order for the very next implementation session:

1. create the new backend package layout under `files/app/`
2. add settings and environment loading
3. add SQLAlchemy and Alembic wiring
4. define first-pass Postgres models
5. wire FastAPI app to the new package structure
6. leave old routers in place temporarily while the new stack comes online

## Definition of Done for Phase 1

Phase 1 is done when:

- the app starts using centralized settings
- Postgres is connected
- Redis is connected
- the first migration runs
- users and interviews exist in real tables
- the in-memory session store is no longer required for new flows

## First Files To Create

- `files/app/__init__.py`
- `files/app/main.py`
- `files/app/core/settings.py`
- `files/app/core/security.py`
- `files/app/db/base.py`
- `files/app/db/session.py`
- `files/app/db/models/`
- `files/app/api/routers/`
- `files/app/schemas/`
- `files/app/providers/`
- `files/app/services/`
- `files/app/agents/`

## Implementation Risks

- trying to migrate the frontend before the backend contracts are stable
- putting provider logic directly in routers
- letting the adaptive controller remain underspecified
- skipping schema/version fields for model outputs
- mixing prototype and production state models for too long

## Near-Term Priority

The top priority is:

- establish the new backend foundation before touching the adaptive intelligence layer in depth

Reason:

- if the persistence, auth, and provider boundaries are weak, the smarter interview logic will sit on unstable ground

## Working Rules

- no silent fallback to mock data in production paths
- every provider call should have explicit timeout and error handling
- every generated artifact should store provider, model, latency, and prompt version when applicable
- every major architecture or model decision should be updated in this file

## Change Log

### 2026-04-25

- created implementation tracker
- locked primary provider choices for STT, TTS, live interview surface, adaptive reasoning, and moderation
- replaced the earlier flat question-generation assumption with a structured interview blueprint architecture
- moved main reasoning responsibilities to `gemini-2.5-flash`
- reserved `gemini-2.5-pro` for higher-trust asynchronous evaluation work
- narrowed agentic behavior to blueprint building, turn analysis, scoring, verification, and report synthesis
- decided on hosted PostgreSQL as system of record
- decided on Redis for WebSocket presence and ephemeral coordination only

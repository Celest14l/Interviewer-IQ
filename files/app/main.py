"""Production-oriented FastAPI entrypoint for the new InterviewIQ app."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.analysis import router as analysis_router
from app.api.routers.auth import router as auth_router
from app.api.routers.audio import router as audio_router
from app.api.routers.health import router as health_router
from app.api.routers.interviews import router as interview_router
from app.api.routers.realtime import router as realtime_router
from app.core.settings import get_settings
from app.db.init_db import init_db


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Production-oriented InterviewIQ backend",
        version=settings.app_version,
    )

    @app.on_event("startup")
    def startup() -> None:
        init_db()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(analysis_router, prefix=settings.api_prefix)
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(audio_router, prefix=settings.api_prefix)
    app.include_router(interview_router, prefix=settings.api_prefix)
    app.include_router(realtime_router, prefix=settings.api_prefix)
    return app


app = create_app()

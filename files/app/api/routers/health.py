"""Health and readiness endpoints."""

from fastapi import APIRouter

from app.core.settings import get_settings
from app.db.session import db_status, redis_status

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }


@router.get("/ready")
async def ready() -> dict:
    return {
        "database": db_status(),
        "redis": redis_status(),
    }

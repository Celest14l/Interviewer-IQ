"""Database and Redis session wiring.

This project currently connects to Supabase through the pooler endpoint.
Pgbouncer-style poolers do not play well with psycopg3 prepared statements,
so we explicitly disable automatic statement preparation here.
"""

from collections.abc import Generator

from redis import Redis
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    future=True,
    pool_pre_ping=True,
    connect_args={"prepare_threshold": None},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)

redis_client = Redis.from_url(settings.redis_url, decode_responses=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_status() -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "url": settings.database_url}
    except SQLAlchemyError as exc:
        return {"ok": False, "error": str(exc)}


def redis_status() -> dict:
    try:
        redis_client.ping()
        return {"ok": True, "url": settings.redis_url}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

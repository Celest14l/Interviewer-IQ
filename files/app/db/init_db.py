"""Database initialization helpers."""

from app.db.base import Base
from app.db.models import final_report, interview, interview_blueprint, transcript_turn, uploaded_file, user  # noqa: F401
from app.db.session import engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

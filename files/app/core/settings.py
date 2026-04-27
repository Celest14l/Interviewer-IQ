"""Application settings and environment loading."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "InterviewIQ API"
    app_version: str = "2.0.0"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    secret_key: str = "change-me"
    access_token_ttl_minutes: int = 60 * 24

    api_prefix: str = "/api"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "null",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://localhost:8091",
            "http://127.0.0.1:8091",
        ]
    )

    database_url: str = "sqlite+pysqlite:///./interviewer_iq.db"
    redis_url: str = "redis://localhost:6379/0"

    gemini_api_key: str = ""
    groq_api_key: str = ""
    cartesia_api_key: str = ""
    nvidia_api_key: str = ""
    gemini_reasoning_model: str = "gemini-2.5-flash"
    nvidia_live_model: str = "nvidia/nemotron-mini-4b-instruct"
    groq_stt_model: str = "whisper-large-v3-turbo"
    cartesia_model_id: str = "sonic-3"
    cartesia_voice_id: str = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
    cartesia_language: str = "en"
    cartesia_version: str = "2026-03-01"
    realtime_presence_ttl_seconds: int = 90
    realtime_interview_lock_seconds: int = 45
    cv_python_path: str = "../venv/Scripts/python.exe"

    upload_dir: str = "storage/uploads"

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

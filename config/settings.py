from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.constants import (
    DEFAULT_CODER_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REVIEWER_MODEL,
    UE5_REMOTE_CONTROL_PORT,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM backends
    ollama_base_url: str = "http://localhost:11434"
    groq_api_key: str = ""

    # Model assignments
    planner_model: str = DEFAULT_PLANNER_MODEL
    coder_model: str = DEFAULT_CODER_MODEL
    reviewer_model: str = DEFAULT_REVIEWER_MODEL

    # UE5 connection
    ue5_remote_control_host: str = "localhost"
    ue5_remote_control_port: int = UE5_REMOTE_CONTROL_PORT

    # State backend
    redis_url: str = "redis://localhost:6379/0"

    # Behavior
    dry_run: bool = False
    log_level: str = "INFO"

    @property
    def ue5_base_url(self) -> str:
        return f"http://{self.ue5_remote_control_host}:{self.ue5_remote_control_port}"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

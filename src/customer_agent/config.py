"""Application settings, loaded from environment variables (prefix ``CA_``) and ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CA_", env_file=".env", extra="ignore")

    app_env: Literal["local", "test", "ci", "prod"] = "local"
    log_level: str = "INFO"
    database_url: str = "postgresql://agent:agent@localhost:5432/customer_agent"
    knowledge_dir: Path = Path("data/knowledge")


@lru_cache
def get_settings() -> Settings:
    return Settings()

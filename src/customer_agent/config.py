"""Application settings, loaded from environment variables (prefix ``CA_``) and ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CA_", env_file=".env", extra="ignore")

    app_env: Literal["local", "test", "ci", "prod"] = "local"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://agent:agent@localhost:5432/customer_agent"
    sql_echo: bool = False  # log every SQL statement (debugging only)
    knowledge_dir: Path = Path("data/knowledge")
    tenant_id: str = "nordelektro"

    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("CA_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    @field_validator("openai_api_key", "openai_base_url", mode="before")
    @classmethod
    def _empty_as_none(cls, value: Any) -> Any:
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()

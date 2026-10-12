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
    # The SDK default is 600 s with 2 retries; a hung call would block a request for minutes.
    embedding_timeout_seconds: float = 15.0

    # Chat model (ADR-0008): any OpenAI-compatible endpoint
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = "https://opencode.ai/zen/go/v1"
    llm_model: str = "deepseek-v4-flash"
    llm_timeout_seconds: float = 30.0
    # Honest client identification; OpenCode Go requires both (ADR-0008).
    llm_user_agent: str = "nordelektro-customer-agent/0.1.0"
    llm_session_header: str | None = None  # e.g. "x-opencode-session"
    llm_reasoning_effort: str | None = None  # e.g. "none" for OpenAI GPT-5 models
    # Per-role overrides (None = use llm_model / llm_reasoning_effort). Routing must be fast and
    # stable; answers may afford more reasoning.
    router_llm_model: str | None = None
    router_reasoning_effort: str | None = None
    answer_llm_model: str | None = None
    answer_reasoning_effort: str | None = None
    llm_cache_dir: Path | None = None  # evaluation only, see infrastructure/llm_cache.py

    # Retrieval: RRF weight of the full-text list relative to the vector list (1.0 = equal).
    # 0.25: chosen on core.jsonl, confirmed on retrieval_holdout.jsonl (Day 5, architecture §6.4).
    retrieval_fulltext_weight: float = 0.25

    # Answering (Day 3)
    answer_prompt_version: int = 2
    router_prompt_version: int = 2
    # Below this best cosine similarity the question is treated as unrelated and answered with
    # "no answer" without an LLM call. Between this floor and clearly relevant scores the LLM
    # decides (it may reply KEINE_ANTWORT). Measured on Day 2/3, see architecture §6.4.
    min_answer_similarity: float = 0.35
    answer_top_k: int = 4  # documents given to the answer prompt
    answer_max_tokens: int = 800

    @field_validator(
        "openai_api_key",
        "openai_base_url",
        "llm_api_key",
        "llm_base_url",
        "llm_cache_dir",
        "llm_reasoning_effort",
        "router_llm_model",
        "router_reasoning_effort",
        "answer_llm_model",
        "answer_reasoning_effort",
        mode="before",
    )
    @classmethod
    def _empty_as_none(cls, value: Any) -> Any:
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()

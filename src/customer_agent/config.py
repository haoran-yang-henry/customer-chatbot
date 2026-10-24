"""Application settings, loaded from environment variables (prefix ``CA_``) and ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CA_", env_file=".env", extra="ignore")

    app_env: Literal["local", "test", "ci", "prod"] = "local"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"  # json in containers (Day 7)
    # OTLP/HTTP receiver for traces, e.g. http://localhost:4318 (Jaeger); unset = not exported.
    otel_endpoint: str | None = None
    # Langfuse (rebuild plan phase 1): model calls and turns as traces, evaluations as experiments.
    # The SDK's own variable names are accepted, so one .env serves the app and Langfuse's tools.
    langfuse_public_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CA_LANGFUSE_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY"),
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CA_LANGFUSE_SECRET_KEY", "LANGFUSE_SECRET_KEY"),
    )
    langfuse_base_url: str | None = Field(  # unset = Langfuse Cloud EU
        default=None, validation_alias=AliasChoices("CA_LANGFUSE_BASE_URL", "LANGFUSE_BASE_URL")
    )
    database_url: str = "postgresql+psycopg://agent:agent@localhost:5432/customer_agent"
    sql_echo: bool = False  # log every SQL statement (debugging only)
    # Connection pool per process. One turn can hold three connections at once (session, the two
    # concurrent searches); with the SQLAlchemy defaults (5 + 10) turns queued for connections in
    # the load test (p99 284 ms vs 76 ms with 20 + 10, Day 7). 30 per worker: several workers or
    # instances need a pooler (PgBouncer) below Postgres' default limit of 100 connections.
    db_pool_size: int = 20
    db_max_overflow: int = 10
    knowledge_dir: Path = Path("data/knowledge")
    tenant_id: str = "nordelektro"

    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("CA_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    # The SDK default is 600 s with 2 retries; a hung call would block a request for minutes.
    embedding_timeout_seconds: float = 15.0  # ingestion batches
    embedding_query_timeout_seconds: float = 5.0  # one query embedding on the request path

    # Chat model (ADR-0008): any OpenAI-compatible endpoint
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = "https://opencode.ai/zen/go/v1"
    llm_model: str = "deepseek-v4-flash"
    llm_timeout_seconds: float = 30.0  # answer generation (read timeout per streamed chunk)
    # Router and classifier: 4 s and one retry stay below NFR-11's 10 s to the fallback (Day 7).
    router_timeout_seconds: float = 4.0
    llm_max_retries: int = 1  # SDK retries with exponential backoff (connection errors, 429, 5xx)
    # Circuit breaker per endpoint: open after this many consecutive failures, for this long.
    breaker_failure_threshold: int = 3
    breaker_reset_seconds: float = 30.0
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
    # Evaluation only: replay (or record) model responses, see infrastructure/cassette.py.
    eval_cassette: Path | None = None
    eval_cassette_record: bool = False

    # Authentication (ADR-0010). Unset issuer = no authentication: every request is anonymous.
    auth_issuer: str | None = None  # e.g. http://localhost:8080/realms/nordelektro
    auth_audience: str = "customer-agent-api"
    auth_jwks_url: str | None = None  # default: <issuer>/protocol/openid-connect/certs
    # Reject anonymous chat requests (public demo: every message costs model calls).
    auth_required: bool = False
    # Browser origins allowed to call the API (comma-separated), e.g. the GitHub Pages demo.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Retrieval: RRF weight of the full-text list relative to the vector list (1.0 = equal).
    # 0.25: chosen on core.jsonl, confirmed on retrieval_holdout.jsonl (Day 5, architecture §6.4).
    retrieval_fulltext_weight: float = 0.25

    # Answering (Day 3)
    answer_prompt_version: int = 4
    router_prompt_version: int = 5
    # Below this best cosine similarity the question is treated as unrelated and answered with
    # "no answer" without an LLM call. Between this floor and clearly relevant scores the LLM
    # decides (it may reply KEINE_ANTWORT). Measured on Day 2/3, see architecture §6.4.
    min_answer_similarity: float = 0.35
    answer_top_k: int = 4  # documents given to the answer prompt
    answer_max_tokens: int = 800

    # Guardrails (Day 6, ADR-0011). The classifier uses the router's model and runs concurrently.
    guard_classifier_enabled: bool = True
    guard_prompt_version: int = 1
    # Attack probability at which the classifier blocks (evaluation plan §5.5).
    guard_classifier_threshold: float = 0.7
    max_tickets_per_day: int = 3  # per customer, rolling 24 hours (T12)
    # Chat messages per minute per customer (logged in) or client address (anonymous); 0 = off.
    rate_limit_per_minute: int = 20
    # Customer feedback (rebuild phase 3): this share of sessions is asked once, from this many
    # rounds on, whether the conversation helped.
    feedback_sample_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    feedback_min_rounds: int = Field(default=3, ge=1)
    # Conversations without a turn for this long are deleted (policy-privacy;
    # customer-agent-retention, run daily).
    retention_days: int = Field(default=90, ge=1)

    @field_validator(
        "openai_api_key",
        "openai_base_url",
        "llm_api_key",
        "llm_base_url",
        "llm_cache_dir",
        "eval_cassette",
        "auth_issuer",
        "auth_jwks_url",
        "otel_endpoint",
        "langfuse_public_key",
        "langfuse_secret_key",
        "langfuse_base_url",
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

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _comma_separated(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]
        return value

    @property
    def langfuse_enabled(self) -> bool:
        """Both keys set, and never in tests or CI: a developer's .env must not send test runs."""
        return (
            self.langfuse_public_key is not None
            and self.langfuse_secret_key is not None
            and self.app_env not in ("test", "ci")
        )

    @model_validator(mode="after")
    def _no_evaluation_shortcuts_in_prod(self) -> "Settings":
        # A cassette answers only recorded questions; the cache would replay stale answers.
        if self.app_env == "prod" and (self.eval_cassette or self.llm_cache_dir):
            raise ValueError("CA_EVAL_CASSETTE and CA_LLM_CACHE_DIR are for evaluation runs only")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

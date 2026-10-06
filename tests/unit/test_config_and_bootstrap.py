import pytest

from customer_agent.bootstrap import (
    ConfigurationError,
    check_embedding_dimensions,
    create_embedding_provider,
)
from customer_agent.config import Settings
from customer_agent.infrastructure.embeddings import HashingEmbeddingProvider


def test_empty_optional_values_become_none() -> None:
    settings = Settings(_env_file=None, openai_api_key="", openai_base_url="")

    assert settings.openai_api_key is None
    assert settings.openai_base_url is None


def test_standard_openai_variable_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CA_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(settings)


def test_missing_api_key_fails_with_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CA_OPENAI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigurationError, match="CA_OPENAI_API_KEY"):
        create_embedding_provider(Settings(_env_file=None))


def test_dimension_mismatch_with_database_column_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="dimensions"):
        check_embedding_dimensions(HashingEmbeddingProvider(dimensions=3072))

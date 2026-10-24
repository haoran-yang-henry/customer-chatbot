import gzip
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from customer_agent.bootstrap import (
    ConfigurationError,
    check_embedding_dimensions,
    create_chat_models,
    create_embedding_provider,
)
from customer_agent.config import Settings
from customer_agent.infrastructure.cassette import CassetteEmbeddings
from customer_agent.infrastructure.lifecycle import aclose_all
from customer_agent.infrastructure.models import BreakerChatModel, ProviderChatModel
from customer_agent.infrastructure.recorded import ReplayChatModel


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


def test_replaying_a_cassette_needs_no_api_key(tmp_path: Path) -> None:
    cassette = tmp_path / "smoke.json.gz"
    with gzip.open(cassette, "wt", encoding="utf-8") as file:
        json.dump({"chat": {}, "embeddings": {}}, file)
    settings = Settings(_env_file=None, eval_cassette=cassette)

    models = create_chat_models(settings)

    assert all(isinstance(model, ReplayChatModel) for model in models.values())
    assert isinstance(create_embedding_provider(settings), CassetteEmbeddings)


def test_evaluation_shortcuts_are_refused_in_prod(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="evaluation runs only"):
        Settings(_env_file=None, app_env="prod", eval_cassette=tmp_path / "smoke.json.gz")
    with pytest.raises(ValidationError, match="evaluation runs only"):
        Settings(_env_file=None, app_env="prod", llm_cache_dir=tmp_path)


def test_dimension_mismatch_with_database_column_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="dimensions"):
        check_embedding_dimensions(3072)


async def test_embedding_clients_have_short_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CA_OPENAI_API_KEY", "sk-test")

    provider = create_embedding_provider(Settings(_env_file=None))._inner  # type: ignore[attr-defined]

    model = provider
    assert model.request_timeout == 15.0 and model.max_retries == 1
    assert model.query_timeout == 5.0  # the request path waits less
    await provider.aclose()


async def test_router_timeout_and_retries_stay_below_the_fallback_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NFR-11: an LLM outage reaches the fallback within 10 s (timeout × attempts + backoff)."""
    monkeypatch.setenv("CA_LLM_API_KEY", "sk-test")
    settings = Settings(_env_file=None)

    models = create_chat_models(settings)
    router = models["router"]
    assert isinstance(router, BreakerChatModel)
    model = router.inner
    assert isinstance(model, ProviderChatModel)

    attempts = settings.llm_max_retries + 1
    assert isinstance(model.request_timeout, float)
    assert model.request_timeout * attempts + 1.0 < 10.0
    await aclose_all(*models.values())


@pytest.mark.parametrize(
    ("header", "kept"),
    [
        ("req-123", True),
        ("a" * 64, True),
        ("a" * 65, False),
        ("evil\r\nX-Injected: 1", False),
        ("", False),
    ],
)
def test_request_id_header_is_validated(header: str, kept: bool) -> None:
    from customer_agent.api.middleware import request_id_from

    assert (request_id_from(header) == header) is kept

from fastapi.testclient import TestClient

from customer_agent import __version__
from customer_agent.config import Settings
from customer_agent.main import create_app


def test_health_returns_ok_and_version() -> None:
    client = TestClient(create_app(Settings(app_env="test")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_docs_are_disabled_in_production() -> None:
    client = TestClient(create_app(Settings(app_env="prod")))

    assert client.get("/docs").status_code == 404


def test_liveness_alias_and_readiness_without_dependencies() -> None:
    app = create_app(Settings(app_env="test"), chat=object())  # type: ignore[arg-type]
    with TestClient(app) as client:
        assert client.get("/health/live").json()["status"] == "ok"
        ready = client.get("/health/ready")

    assert ready.status_code == 200 and ready.json()["status"] == "ready"


def test_readiness_fails_on_the_database_but_not_on_an_open_breaker() -> None:
    app = create_app(Settings(app_env="test"), chat=object())  # type: ignore[arg-type]
    with TestClient(app) as client:

        async def llm_down() -> dict[str, str]:
            return {"database": "ok", "llm": "open"}

        async def database_down() -> dict[str, str]:
            return {"database": "unavailable", "llm": "closed"}

        app.state.readiness = llm_down
        degraded = client.get("/health/ready")
        app.state.readiness = database_down
        unavailable = client.get("/health/ready")

    assert degraded.status_code == 200 and degraded.json()["checks"]["llm"] == "open"
    assert unavailable.status_code == 503 and unavailable.json()["status"] == "unavailable"


def test_readiness_waits_for_the_checkpoint_tables() -> None:
    app = create_app(Settings(app_env="test"), chat=object())  # type: ignore[arg-type]
    with TestClient(app) as client:

        async def tables_pending() -> dict[str, str]:
            return {"database": "ok", "checkpoints": "pending"}

        app.state.readiness = tables_pending
        response = client.get("/health/ready")

    assert response.status_code == 503 and response.json()["checks"]["checkpoints"] == "pending"

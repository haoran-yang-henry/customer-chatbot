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

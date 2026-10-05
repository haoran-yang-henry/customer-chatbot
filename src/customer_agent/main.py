"""Composition root: builds the FastAPI application and wires adapters into use cases."""

from fastapi import FastAPI

from customer_agent import __version__
from customer_agent.api import health
from customer_agent.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="NordElektro Customer Service Agent",
        version=__version__,
        docs_url="/docs" if settings.app_env != "prod" else None,
    )
    app.include_router(health.router)
    return app

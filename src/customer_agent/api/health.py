from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from customer_agent import __version__

router = APIRouter(tags=["operations"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


@router.get("/health")
async def health() -> HealthResponse:
    """Liveness probe. Readiness (database, providers) is added on Day 2."""
    return HealthResponse(status="ok", version=__version__)

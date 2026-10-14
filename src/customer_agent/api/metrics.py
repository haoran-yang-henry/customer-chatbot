"""``GET /metrics``: Prometheus text format (Day 7). Expose it to the scraper only, not to the
internet: it reveals traffic and error patterns."""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["operations"])


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

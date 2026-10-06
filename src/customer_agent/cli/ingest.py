"""``customer-agent-ingest``: synchronise the knowledge index with ``data/knowledge``."""

import argparse
import asyncio

from customer_agent.application.ingestion import IngestionReport, IngestionService
from customer_agent.bootstrap import (
    create_embedding_provider,
    create_engine,
    create_knowledge_repository,
)
from customer_agent.config import get_settings
from customer_agent.infrastructure.knowledge_files import load_knowledge_directory


def _print_report(report: IngestionReport) -> None:
    for label, ids in [
        ("added", report.added),
        ("updated", report.updated),
        ("removed", report.removed),
        ("skipped (not approved)", report.skipped_not_approved),
    ]:
        print(f"{label:<24} {len(ids):>3}  {', '.join(ids)}")
    print(f"{'unchanged':<24} {len(report.unchanged):>3}")
    print(f"{'chunks embedded':<24} {report.chunks_embedded:>3}")


async def _run() -> IngestionReport:
    settings = get_settings()
    documents = load_knowledge_directory(settings.knowledge_dir)
    engine = create_engine(settings)
    try:
        service = IngestionService(
            create_knowledge_repository(engine), create_embedding_provider(settings)
        )
        return await service.run(documents)
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    _print_report(asyncio.run(_run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""``customer-agent-search "question"``: inspect retrieval for one query, mode by mode.

Shows rank, RRF score, cosine similarity and the matching chunk, which is usually enough to see
*why* a document was (not) found. Uses the real database and embedding model from ``.env``.
"""

import argparse
import asyncio
from datetime import date

from customer_agent.application.retrieval import RetrievalService
from customer_agent.bootstrap import (
    create_embedding_provider,
    create_engine,
    create_knowledge_repository,
)
from customer_agent.config import get_settings
from customer_agent.domain.retrieval import RetrievedDocument, SearchMode, SearchScope
from customer_agent.evaluation.schema import DATASET_REFERENCE_DATE


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=[*SearchMode, "all"], default="all")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=DATASET_REFERENCE_DATE,
        help="'Today' for validity filters (default: dataset reference date).",
    )
    return parser.parse_args(argv)


def _print(mode: SearchMode, documents: list[RetrievedDocument]) -> None:
    print(f"\n== {mode} ==")
    if not documents:
        print("  (no results)")
    for rank, doc in enumerate(documents, start=1):
        similarity = f"{doc.similarity:.3f}" if doc.similarity is not None else "  -  "
        flag = "  [UNTRUSTED]" if doc.trust_level == "untrusted" else ""
        print(f"  {rank}. {doc.document_id:<36} rrf={doc.score:.4f} sim={similarity}{flag}")
        best = doc.chunks[0].content.replace("\n", " ")
        print(f"     {best[:110]}…")


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    engine = create_engine(settings)
    try:
        service = RetrievalService(
            create_knowledge_repository(engine), create_embedding_provider(settings)
        )
        scope = SearchScope(today=args.date, tenant_id=settings.tenant_id)
        modes = list(SearchMode) if args.mode == "all" else [SearchMode(args.mode)]
        for mode in modes:
            _print(mode, await service.retrieve(args.query, scope, top_k=args.k, mode=mode))
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    asyncio.run(_run(_parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

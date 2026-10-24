"""``customer-agent-search "question"``: inspect retrieval for one query, mode by mode.

Shows rank, RRF score, cosine similarity and the matching chunk, which is usually enough to see
*why* a document was (not) found. Uses the real database and embedding model from ``.env``.
"""

import argparse
import asyncio
from datetime import date

from langchain_core.documents import Document

from customer_agent.application.retrieval import HybridRetriever
from customer_agent.bootstrap import (
    create_embedding_provider,
    create_engine,
    create_knowledge_repository,
)
from customer_agent.config import get_settings
from customer_agent.domain.retrieval import SearchMode
from customer_agent.evaluation.schema import DATASET_REFERENCE_DATE
from customer_agent.infrastructure.clock import FixedClock
from customer_agent.infrastructure.lifecycle import aclose_all


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


def _print(mode: SearchMode, documents: list[Document]) -> None:
    print(f"\n== {mode} ==")
    if not documents:
        print("  (no results)")
    for rank, doc in enumerate(documents, start=1):
        meta = doc.metadata
        similarity = f"{meta['similarity']:.3f}" if meta["similarity"] is not None else "  -  "
        flag = "  [UNTRUSTED]" if meta["trust_level"] == "untrusted" else ""
        print(f"  {rank}. {meta['document_id']:<36} rrf={meta['score']:.4f} sim={similarity}{flag}")
        best = doc.page_content.replace("\n", " ")  # the best chunk comes first
        print(f"     {best[:110]}…")


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    engine = create_engine(settings)
    embeddings = None
    try:
        embeddings = create_embedding_provider(settings)
        retriever = HybridRetriever(
            search=create_knowledge_repository(engine),
            embeddings=embeddings,
            clock=FixedClock(args.date),
            tenant_id=settings.tenant_id,
            k=args.k,
        )
        modes = list(SearchMode) if args.mode == "all" else [SearchMode(args.mode)]
        for mode in modes:
            _print(mode, await retriever.model_copy(update={"mode": mode}).ainvoke(args.query))
    finally:
        await aclose_all(embeddings)
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    asyncio.run(_run(_parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

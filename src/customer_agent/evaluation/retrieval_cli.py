"""``customer-agent-eval-retrieval``: Recall@k / MRR for one or all search modes."""

import argparse
import asyncio
from pathlib import Path

from customer_agent.application.retrieval import RetrievalService
from customer_agent.bootstrap import (
    create_embedding_provider,
    create_engine,
    create_knowledge_repository,
)
from customer_agent.config import get_settings
from customer_agent.domain.retrieval import SearchMode, SearchScope
from customer_agent.evaluation.retrieval import RetrievalReport, run_retrieval_evaluation
from customer_agent.evaluation.schema import DATASET_REFERENCE_DATE, load_dataset
from customer_agent.infrastructure.lifecycle import aclose_all


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=[*SearchMode, "all"], default="all")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--fulltext-weight",
        type=float,
        help="RRF weight of the full-text list (vector = 1.0); default from settings",
    )
    parser.add_argument("--out", type=Path, help="Write the hybrid (or single-mode) report here.")
    return parser.parse_args(argv)


def _print(report: RetrievalReport) -> None:
    print(f"\n== {report.mode} ({report.embedding_model}), {report.cases} cases ==")
    print(f"Recall@{report.k}: {report.recall_at_k:.3f}   MRR: {report.mrr:.3f}")
    for name, value in report.by_category.items():
        print(f"  {name:<22} recall {value:.2f}")
    for miss in report.misses:
        print(f"  MISS {miss.case_id}: expected {miss.expected}, got {miss.retrieved[: report.k]}")


async def _run(args: argparse.Namespace) -> list[RetrievalReport]:
    settings = get_settings()
    cases = load_dataset(args.dataset)
    scope = SearchScope(today=DATASET_REFERENCE_DATE, tenant_id=settings.tenant_id)
    modes = list(SearchMode) if args.mode == "all" else [SearchMode(args.mode)]
    engine = create_engine(settings)
    embeddings = None
    try:
        embeddings = create_embedding_provider(settings)
        weight = (
            args.fulltext_weight
            if args.fulltext_weight is not None
            else settings.retrieval_fulltext_weight
        )
        service = RetrievalService(
            create_knowledge_repository(engine), embeddings, fulltext_weight=weight
        )
        return [
            await run_retrieval_evaluation(
                cases, service, scope, mode=m, embedding_model=settings.embedding_model, k=args.k
            )
            for m in modes
        ]
    finally:
        await aclose_all(embeddings)
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reports = asyncio.run(_run(args))
    for report in reports:
        _print(report)
    if args.out:
        main_report = next((r for r in reports if r.mode is SearchMode.HYBRID), reports[0])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(main_report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

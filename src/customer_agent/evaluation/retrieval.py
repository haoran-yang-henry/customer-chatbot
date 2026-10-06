"""Retrieval evaluation: Recall@k and MRR on the dataset's ``expected_sources``, without an LLM."""

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from customer_agent.application.retrieval import RetrievalService
from customer_agent.domain.conversation import Role
from customer_agent.domain.retrieval import SearchMode, SearchScope
from customer_agent.evaluation.schema import EvalCase

RANKING_DEPTH = 10  # MRR looks this deep; Recall@k uses the first k


def retrieval_query(case: EvalCase) -> str:
    """All user turns joined: a stand-in for conversation-aware query rewriting (Day 4)."""
    return " ".join(turn.content for turn in case.turns if turn.role is Role.USER)


def recall_at_k(expected: Sequence[str], ranked: Sequence[str], k: int) -> float:
    return len(set(expected) & set(ranked[:k])) / len(set(expected))


def reciprocal_rank(expected: Sequence[str], ranked: Sequence[str]) -> float:
    for rank, document_id in enumerate(ranked, start=1):
        if document_id in expected:
            return 1.0 / rank
    return 0.0


class RetrievalCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    category: str
    query: str
    expected: list[str]
    retrieved: list[str]
    recall: float
    reciprocal_rank: float


class RetrievalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: SearchMode
    embedding_model: str
    k: int
    cases: int
    recall_at_k: float
    mrr: float
    by_category: dict[str, float]
    results: list[RetrievalCaseResult]

    @property
    def misses(self) -> list[RetrievalCaseResult]:
        return [r for r in self.results if r.recall < 1.0]


async def run_retrieval_evaluation(
    cases: Sequence[EvalCase],
    service: RetrievalService,
    scope: SearchScope,
    *,
    mode: SearchMode,
    embedding_model: str,
    k: int = 5,
) -> RetrievalReport:
    results = []
    for case in (c for c in cases if c.expected_sources):
        query = retrieval_query(case)
        documents = await service.retrieve(query, scope, top_k=RANKING_DEPTH, mode=mode)
        ranked = [d.document_id for d in documents]
        results.append(
            RetrievalCaseResult(
                case_id=case.id,
                category=case.category,
                query=query,
                expected=case.expected_sources,
                retrieved=ranked,
                recall=recall_at_k(case.expected_sources, ranked, k),
                reciprocal_rank=reciprocal_rank(case.expected_sources, ranked),
            )
        )

    by_category: defaultdict[str, list[float]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result.recall)

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return RetrievalReport(
        mode=mode,
        embedding_model=embedding_model,
        k=k,
        cases=len(results),
        recall_at_k=mean([r.recall for r in results]),
        mrr=mean([r.reciprocal_rank for r in results]),
        by_category={name: mean(values) for name, values in sorted(by_category.items())},
        results=results,
    )

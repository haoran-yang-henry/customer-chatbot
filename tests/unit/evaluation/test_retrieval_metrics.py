import pytest

from customer_agent.evaluation.retrieval import recall_at_k, reciprocal_rank, retrieval_query
from tests.unit.evaluation.test_scoring import make_case


def test_recall_counts_expected_documents_within_k() -> None:
    assert recall_at_k(["a", "b"], ["x", "a", "y", "b"], k=2) == pytest.approx(0.5)


def test_reciprocal_rank_uses_first_relevant_position() -> None:
    assert reciprocal_rank(["b"], ["x", "b", "b"]) == pytest.approx(0.5)
    assert reciprocal_rank(["b"], ["x", "y"]) == 0.0


def test_retrieval_query_joins_user_turns_only() -> None:
    case = make_case(
        turns=[
            {"role": "user", "content": "Versand nach Österreich?"},
            {"role": "assistant", "content": "7,95 €"},
            {"role": "user", "content": "Und in die Schweiz?"},
        ]
    )

    assert retrieval_query(case) == "Versand nach Österreich? Und in die Schweiz?"

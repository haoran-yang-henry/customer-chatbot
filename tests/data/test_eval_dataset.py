"""Consistency between the evaluation dataset, the knowledge base and the seed data."""

import json
from collections import Counter

import pytest

from customer_agent.domain.knowledge import KnowledgeDocument, TrustLevel
from customer_agent.evaluation.schema import Category, EvalCase, load_dataset
from customer_agent.infrastructure.knowledge_files import load_knowledge_directory
from tests.conftest import CORE_DATASET, KNOWLEDGE_DIR, REFERENCE_DATE, SEED_DIR, TENANT_ID


@pytest.fixture(scope="module", params=["core.jsonl", "retrieval_holdout.jsonl", "redteam.jsonl"])
def cases(request: pytest.FixtureRequest) -> list[EvalCase]:
    return load_dataset(CORE_DATASET.parent / request.param)


@pytest.fixture(scope="module")
def documents_by_id() -> dict[str, KnowledgeDocument]:
    return {doc.meta.id: doc for doc in load_knowledge_directory(KNOWLEDGE_DIR)}


def test_every_category_is_covered() -> None:
    counts = Counter(case.category for case in load_dataset(CORE_DATASET))

    assert set(counts) == set(Category)


def test_holdout_shares_no_question_with_core() -> None:
    core = {c.turns[-1].content for c in load_dataset(CORE_DATASET)}
    holdout = load_dataset(CORE_DATASET.parent / "retrieval_holdout.jsonl")

    assert not core & {c.turns[-1].content for c in holdout}


def test_expected_sources_are_retrievable_and_trusted(
    cases: list[EvalCase], documents_by_id: dict[str, KnowledgeDocument]
) -> None:
    """A case may only expect citations of documents a customer is allowed to see."""
    for case in cases:
        for source in case.expected_sources:
            assert source in documents_by_id, f"{case.id}: unknown source {source}"
            meta = documents_by_id[source].meta
            assert meta.is_retrievable_for_customer(REFERENCE_DATE, TENANT_ID), (
                f"{case.id}: {source} is not retrievable on {REFERENCE_DATE}"
            )
            assert meta.trust_level is not TrustLevel.UNTRUSTED, f"{case.id}: cites {source}"


def test_customers_exist(cases: list[EvalCase]) -> None:
    customers = json.loads((SEED_DIR / "customers.json").read_text(encoding="utf-8"))
    known = {customer["customer_id"] for customer in customers}

    for case in cases:
        assert case.customer_id is None or case.customer_id in known, case.id


def test_transactional_actions_have_a_customer(cases: list[EvalCase]) -> None:
    """Looking up an order requires an authenticated customer."""
    order_tools = {"get_order_status"}
    for case in cases:
        if case.expected_tool in order_tools:
            assert case.customer_id is not None, case.id

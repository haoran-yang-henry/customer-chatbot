"""Governance rules for the knowledge base in data/knowledge (threat model T03, T04)."""

from collections import defaultdict

import pytest

from customer_agent.domain.knowledge import (
    AccessLevel,
    KnowledgeDocument,
    SourceType,
    TrustLevel,
)
from customer_agent.infrastructure.knowledge_files import load_knowledge_directory
from tests.conftest import KNOWLEDGE_DIR, REFERENCE_DATE, TENANT_ID


@pytest.fixture(scope="module")
def documents() -> list[KnowledgeDocument]:
    return load_knowledge_directory(KNOWLEDGE_DIR)


def test_knowledge_base_is_not_empty(documents: list[KnowledgeDocument]) -> None:
    assert len(documents) >= 25


def test_at_most_one_active_policy_per_topic(documents: list[KnowledgeDocument]) -> None:
    active_by_topic: defaultdict[str, list[str]] = defaultdict(list)
    for doc in documents:
        if doc.meta.source_type is SourceType.POLICY and doc.meta.is_active_on(REFERENCE_DATE):
            active_by_topic[doc.meta.topic].append(doc.meta.id)

    conflicts = {topic: ids for topic, ids in active_by_topic.items() if len(ids) > 1}
    assert not conflicts, f"conflicting active policies: {conflicts}"


def test_versions_are_unique_per_topic(documents: list[KnowledgeDocument]) -> None:
    seen: set[tuple[str, int]] = set()
    for doc in documents:
        key = (doc.meta.topic, doc.meta.version)
        assert key not in seen, f"duplicate version {key}"
        seen.add(key)


def test_internal_documents_are_never_public(documents: list[KnowledgeDocument]) -> None:
    for doc in documents:
        if (
            doc.meta.source_type is SourceType.INTERNAL
            or doc.meta.trust_level is TrustLevel.INTERNAL
        ):
            assert AccessLevel.PUBLIC not in doc.meta.acl, doc.meta.id


def test_community_content_is_marked_untrusted(documents: list[KnowledgeDocument]) -> None:
    for doc in documents:
        if doc.meta.source_type is SourceType.COMMUNITY:
            assert doc.meta.trust_level is TrustLevel.UNTRUSTED, doc.meta.id


def test_all_documents_belong_to_the_tenant(documents: list[KnowledgeDocument]) -> None:
    assert {doc.meta.tenant_id for doc in documents} == {TENANT_ID}

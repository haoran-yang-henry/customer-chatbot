"""Governance rules for the knowledge base in data/knowledge (threat model T03, T04)."""

from collections import defaultdict
from datetime import date
from itertools import pairwise

import pytest

from customer_agent.domain.knowledge import (
    AccessLevel,
    DocumentStatus,
    KnowledgeDocument,
    SourceType,
    TrustLevel,
)
from customer_agent.infrastructure.knowledge_files import load_knowledge_directory
from tests.conftest import KNOWLEDGE_DIR, TENANT_ID


@pytest.fixture(scope="module")
def documents() -> list[KnowledgeDocument]:
    return load_knowledge_directory(KNOWLEDGE_DIR)


def test_knowledge_base_is_not_empty(documents: list[KnowledgeDocument]) -> None:
    assert len(documents) >= 25


def test_approved_policy_versions_never_overlap(documents: list[KnowledgeDocument]) -> None:
    """At most one approved policy per topic is valid on any date, past or future.

    Publishing a new version therefore requires closing the previous one (``valid_until``)
    in the same change; see docs/architecture.md, "Knowledge lifecycle".
    """
    windows: defaultdict[str, list[tuple[date, date, str]]] = defaultdict(list)
    for doc in documents:
        meta = doc.meta
        if meta.source_type is SourceType.POLICY and meta.status is DocumentStatus.APPROVED:
            windows[meta.topic].append((meta.valid_from, meta.valid_until or date.max, meta.id))

    for topic, spans in windows.items():
        spans.sort()
        for (_, end, earlier), (start, _, later) in pairwise(spans):
            assert start > end, f"{topic}: {earlier} and {later} are both valid on {start}"


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

"""Test data builders shared across test modules."""

from datetime import date
from typing import Any

from customer_agent.domain.knowledge import KnowledgeDocument, KnowledgeDocumentMeta


def make_meta(**overrides: Any) -> KnowledgeDocumentMeta:
    data: dict[str, Any] = {
        "id": "faq-test",
        "title": "Testdokument",
        "topic": "test",
        "intents": ["SHIPPING_INFO"],
        "source_type": "faq",
        "language": "de",
        "version": 1,
        "valid_from": date(2025, 1, 1),
        "valid_until": None,
        "status": "approved",
        "trust_level": "official",
        "tenant_id": "nordelektro",
        "acl": ["public"],
        "owner": "test",
    }
    data.update(overrides)
    return KnowledgeDocumentMeta.model_validate(data)


def make_document(body: str = "Ein kurzer Inhalt.", **meta: Any) -> KnowledgeDocument:
    return KnowledgeDocument(meta=make_meta(**meta), body=body)

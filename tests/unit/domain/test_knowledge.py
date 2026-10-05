from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from customer_agent.domain.knowledge import AccessLevel, DocumentStatus, KnowledgeDocumentMeta


def make_meta(**overrides: Any) -> KnowledgeDocumentMeta:
    data: dict[str, Any] = {
        "id": "policy-returns-v5",
        "title": "Rückgaberecht",
        "topic": "returns",
        "intents": ["RETURN_POLICY"],
        "source_type": "policy",
        "language": "de",
        "version": 5,
        "valid_from": date(2025, 3, 1),
        "valid_until": None,
        "status": "approved",
        "trust_level": "official",
        "tenant_id": "nordelektro",
        "acl": ["public"],
        "owner": "customer-service",
    }
    data.update(overrides)
    return KnowledgeDocumentMeta.model_validate(data)


class TestValidation:
    def test_rejects_validity_window_ending_before_it_starts(self) -> None:
        with pytest.raises(ValidationError, match="valid_until"):
            make_meta(valid_from=date(2025, 3, 1), valid_until=date(2025, 2, 1))

    def test_rejects_non_kebab_case_id(self) -> None:
        with pytest.raises(ValidationError):
            make_meta(id="Policy_Returns")

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            make_meta(priority="high")

    def test_rejects_non_german_language(self) -> None:
        with pytest.raises(ValidationError):
            make_meta(language="en")


class TestIsActiveOn:
    @pytest.mark.parametrize(
        ("day", "expected"),
        [
            (date(2025, 2, 28), False),  # before valid_from
            (date(2025, 3, 1), True),  # first day (inclusive)
            (date(2025, 6, 30), True),  # last day (inclusive)
            (date(2025, 7, 1), False),  # after valid_until
        ],
    )
    def test_validity_window_is_inclusive(self, day: date, expected: bool) -> None:
        meta = make_meta(valid_from=date(2025, 3, 1), valid_until=date(2025, 6, 30))

        assert meta.is_active_on(day) is expected

    def test_open_ended_document_stays_active(self) -> None:
        assert make_meta(valid_until=None).is_active_on(date(2030, 1, 1))

    @pytest.mark.parametrize("status", [DocumentStatus.DRAFT, DocumentStatus.ARCHIVED])
    def test_only_approved_documents_are_active(self, status: DocumentStatus) -> None:
        assert not make_meta(status=status).is_active_on(date(2026, 1, 1))


class TestIsRetrievableForCustomer:
    def test_public_active_document_of_same_tenant_is_retrievable(self) -> None:
        assert make_meta().is_retrievable_for_customer(date(2026, 1, 1), "nordelektro")

    def test_internal_document_is_not_retrievable(self) -> None:
        meta = make_meta(acl=[AccessLevel.INTERNAL])

        assert not meta.is_retrievable_for_customer(date(2026, 1, 1), "nordelektro")

    def test_other_tenant_is_not_retrievable(self) -> None:
        assert not make_meta().is_retrievable_for_customer(date(2026, 1, 1), "other-shop")

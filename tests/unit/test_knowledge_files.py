from pathlib import Path

import pytest

from customer_agent.infrastructure.knowledge_files import (
    KnowledgeFileError,
    load_knowledge_directory,
    parse_knowledge_markdown,
)

VALID = """---
id: faq-test
title: Test
topic: test
intents: [SHIPPING_INFO]
source_type: faq
language: de
version: 1
valid_from: 2025-01-01
valid_until: null
status: approved
trust_level: official
tenant_id: nordelektro
acl: [public]
owner: test
---
# Überschrift

Inhalt mit Umlauten: Rücksendung.
"""


def test_parses_front_matter_and_body() -> None:
    document = parse_knowledge_markdown(VALID)

    assert document.meta.id == "faq-test"
    assert document.body.startswith("# Überschrift")


def test_missing_front_matter_is_reported_with_source() -> None:
    with pytest.raises(KnowledgeFileError, match=r"x\.md: missing front matter start"):
        parse_knowledge_markdown("# no front matter", source="x.md")


def test_invalid_metadata_is_reported_with_source() -> None:
    with pytest.raises(KnowledgeFileError, match=r"x\.md"):
        parse_knowledge_markdown(VALID.replace("language: de", "language: en"), source="x.md")


def test_file_name_must_match_id(tmp_path: Path) -> None:
    (tmp_path / "wrong-name.md").write_text(VALID, encoding="utf-8")

    with pytest.raises(KnowledgeFileError, match="does not match file name"):
        load_knowledge_directory(tmp_path)

import pytest

from customer_agent.application.chunking import MAX_SECTION_CHARS, chunk_document
from tests.factories import make_document

LONG = "Dies ist ein ausreichend langer Absatz über Versand und Rücksendung. " * 3


def test_document_without_sections_is_one_chunk_with_title() -> None:
    chunks = chunk_document(make_document(body="# Titel\n\nNur ein Absatz.", title="Versand"))

    assert len(chunks) == 1
    assert chunks[0].heading is None
    assert chunks[0].content == "Versand\n\nNur ein Absatz."


def test_each_section_becomes_a_chunk_with_title_and_heading() -> None:
    body = f"# T\n\n{LONG}\n\n## Kosten\n\n{LONG}\n\n## Ablauf\n\n{LONG}"

    chunks = chunk_document(make_document(body=body, title="Rückgabe"))

    assert [c.heading for c in chunks] == [None, "Kosten", "Ablauf"]
    assert chunks[1].content.startswith("Rückgabe - Kosten\n\n")
    assert [c.position for c in chunks] == [0, 1, 2]


def test_tiny_section_is_merged_into_previous_chunk() -> None:
    body = f"{LONG}\n\n## Hinweis\n\nKurz."

    chunks = chunk_document(make_document(body=body))

    assert len(chunks) == 1
    assert chunks[0].content.endswith("Hinweis:\nKurz.")


def test_oversized_section_is_split_at_paragraphs() -> None:
    paragraph = "x" * 600
    body = "\n\n".join([paragraph] * 4)

    chunks = chunk_document(make_document(body=body))

    assert len(chunks) == 2
    assert all(len(c.content) <= MAX_SECTION_CHARS + 100 for c in chunks)


def test_document_with_only_a_title_is_rejected() -> None:
    with pytest.raises(ValueError, match="no content"):
        chunk_document(make_document(body="# Nur ein Titel"))

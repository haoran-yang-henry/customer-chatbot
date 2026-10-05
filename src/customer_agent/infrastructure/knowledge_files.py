"""Loads knowledge documents from Markdown files with YAML front matter."""

from pathlib import Path

import yaml

from customer_agent.domain.knowledge import KnowledgeDocument, KnowledgeDocumentMeta

_DELIMITER = "---"


class KnowledgeFileError(ValueError):
    """A knowledge file is malformed. The message names the file and the problem."""


def parse_knowledge_markdown(text: str, *, source: str = "<string>") -> KnowledgeDocument:
    lines = text.splitlines()
    if not lines or lines[0].strip() != _DELIMITER:
        raise KnowledgeFileError(f"{source}: missing front matter start '---'")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == _DELIMITER)
    except StopIteration:
        raise KnowledgeFileError(f"{source}: missing front matter end '---'") from None

    front_matter = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(front_matter, dict):
        raise KnowledgeFileError(f"{source}: front matter must be a mapping")
    try:
        meta = KnowledgeDocumentMeta.model_validate(front_matter)
        return KnowledgeDocument(meta=meta, body="\n".join(lines[end + 1 :]).strip())
    except ValueError as exc:
        raise KnowledgeFileError(f"{source}: {exc}") from exc


def load_knowledge_directory(directory: Path) -> list[KnowledgeDocument]:
    """Load all ``*.md`` files; the file name (without suffix) must equal the document id."""
    documents = []
    for path in sorted(directory.glob("*.md")):
        document = parse_knowledge_markdown(path.read_text(encoding="utf-8"), source=path.name)
        if document.meta.id != path.stem:
            raise KnowledgeFileError(
                f"{path.name}: id '{document.meta.id}' does not match file name"
            )
        documents.append(document)
    return documents

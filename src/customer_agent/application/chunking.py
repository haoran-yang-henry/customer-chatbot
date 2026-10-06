"""Structure-aware chunking of Markdown knowledge documents.

Each ``##`` section becomes one chunk. Tiny sections are merged into the previous chunk and
oversized sections are split at paragraph boundaries. Every chunk starts with the document title
(and section heading) so that it stays understandable when retrieved on its own.
"""

from customer_agent.domain.knowledge import KnowledgeChunk, KnowledgeDocument

# Bump when the algorithm changes: it is part of the ingestion fingerprint, so a new version
# re-chunks and re-embeds every document on the next ingestion.
CHUNKER_VERSION = "1"

MIN_SECTION_CHARS = 120
MAX_SECTION_CHARS = 1500


def _split_sections(body: str) -> list[tuple[str | None, str]]:
    """Return ``(heading, text)`` pairs; text before the first ``##`` has heading ``None``."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in body.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif line.startswith("# "):
            continue  # the document title is added to every chunk instead
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections]


def _merge_small(sections: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    merged: list[tuple[str | None, str]] = []
    for heading, text in sections:
        if not text:
            continue
        if merged and len(text) < MIN_SECTION_CHARS:
            prev_heading, prev_text = merged[-1]
            label = f"{heading}:\n" if heading else ""
            merged[-1] = (prev_heading, f"{prev_text}\n\n{label}{text}")
        else:
            merged.append((heading, text))
    return merged


def _split_large(text: str) -> list[str]:
    if len(text) <= MAX_SECTION_CHARS:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        if current and len(current) + len(paragraph) + 2 > MAX_SECTION_CHARS:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    parts.append(current)
    return parts


def chunk_document(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    title = document.meta.title
    chunks: list[KnowledgeChunk] = []
    for heading, text in _merge_small(_split_sections(document.body)):
        for part in _split_large(text):
            header = f"{title} - {heading}" if heading else title
            chunks.append(
                KnowledgeChunk(
                    document_id=document.meta.id,
                    position=len(chunks),
                    heading=heading,
                    content=f"{header}\n\n{part}",
                )
            )
    if not chunks:
        raise ValueError(f"{document.meta.id}: document has no content to chunk")
    return chunks

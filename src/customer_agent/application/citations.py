"""Inline citations: the model writes ``[document-id]`` after sentences; the customer sees clean
text and the ids travel separately as ``sources``."""

import re

NO_ANSWER_TOKEN = "KEINE_ANTWORT"  # noqa: S105 - not a secret
_CITATION = re.compile(r"\s*\[([a-z0-9]+(?:-[a-z0-9]+)+)\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?])")


def strip_citations(text: str) -> str:
    """Remove markers without trimming, so streamed fragments keep their spacing."""
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", _CITATION.sub("", text))


def extract_citations(text: str) -> tuple[str, list[str]]:
    """Return text without citation markers and the cited ids in order of first appearance."""
    cited = list(dict.fromkeys(_CITATION.findall(text)))
    return strip_citations(text).strip(), cited


def is_no_answer(text: str) -> bool:
    """The token anywhere counts: models sometimes explain first and append it (Day 4, K-004).
    Such text must never reach the customer; the streamed part is replaced by the final event."""
    return NO_ANSWER_TOKEN in text


class StreamingCitationFilter:
    """Removes ``[id]`` markers from a token stream without ever emitting a partial marker.

    Text after an opening ``[`` is held back until the bracket closes (or grows too long to be a
    citation). The beginning of the stream is held until it cannot be the no-answer token.
    ``raw`` keeps the unfiltered text for citation extraction at the end.
    """

    _MAX_MARKER = 80

    def __init__(self) -> None:
        self._pending = ""
        self._started = False
        self.raw = ""

    def feed(self, delta: str) -> str:
        self.raw += delta
        self._pending += delta
        if not self._started:
            head = self._pending.strip()
            if NO_ANSWER_TOKEN.startswith(head) or head.startswith(NO_ANSWER_TOKEN):
                return ""
            self._started = True
        cut = self._pending.rfind("[")
        if (
            cut != -1
            and "]" not in self._pending[cut:]
            and len(self._pending) - cut < self._MAX_MARKER
        ):
            emit, self._pending = self._pending[:cut], self._pending[cut:]
        else:
            emit, self._pending = self._pending, ""
        return strip_citations(emit)

    def flush(self) -> str:
        if is_no_answer(self.raw):
            return ""
        emit, self._pending = self._pending, ""
        return strip_citations(emit)

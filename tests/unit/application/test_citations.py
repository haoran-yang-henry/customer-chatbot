import pytest

from customer_agent.application.citations import (
    StreamingCitationFilter,
    extract_citations,
    is_no_answer,
)

ANSWER = (
    "Die Rücksendung ist kostenlos. [policy-returns-v5] Die Erstattung dauert bis zu 14 Tage "
    "[faq-refund-timeline]. Mehr dazu [policy-returns-v5]."
)


def test_extracts_unique_ids_in_order_and_cleans_text() -> None:
    text, cited = extract_citations(ANSWER)

    assert cited == ["policy-returns-v5", "faq-refund-timeline"]
    assert "[" not in text
    assert "14 Tage." in text  # no space left before punctuation


def test_square_brackets_that_are_not_ids_stay() -> None:
    text, cited = extract_citations("Größe [XL] ist verfügbar.")

    assert cited == [] and text == "Größe [XL] ist verfügbar."


@pytest.mark.parametrize(
    "text",
    [
        "KEINE_ANTWORT",
        "  KEINE_ANTWORT\n",
        "KEINE_ANTWORT.",
        "Die EVIDENCE enthält keine Informationen dazu.\n\nKEINE_ANTWORT",
    ],
)
def test_no_answer_token(text: str) -> None:
    assert is_no_answer(text)


@pytest.mark.parametrize("piece", [1, 3, 7, 50])
def test_stream_filter_never_emits_markers_and_keeps_raw(piece: int) -> None:
    stream_filter = StreamingCitationFilter()
    emitted = "".join(
        stream_filter.feed(ANSWER[i : i + piece]) for i in range(0, len(ANSWER), piece)
    )
    emitted += stream_filter.flush()

    assert "[" not in emitted and "]" not in emitted
    assert "kostenlos." in emitted and "14 Tage" in emitted
    assert stream_filter.raw == ANSWER


@pytest.mark.parametrize("piece", [1, 4, 20])
def test_stream_filter_suppresses_no_answer_token(piece: int) -> None:
    stream_filter = StreamingCitationFilter()
    raw = "KEINE_ANTWORT"
    emitted = "".join(stream_filter.feed(raw[i : i + piece]) for i in range(0, len(raw), piece))

    assert emitted + stream_filter.flush() == ""


def test_stream_filter_releases_unclosed_bracket_at_end() -> None:
    stream_filter = StreamingCitationFilter()

    emitted = stream_filter.feed("Preis: 49,99 € [inkl") + stream_filter.flush()

    assert emitted == "Preis: 49,99 € [inkl"

"""Which conversations are asked whether they helped (rebuild phase 3)."""

import uuid

import pytest

from customer_agent.application.feedback import FeedbackPolicy
from customer_agent.domain.conversation import Behavior

SESSIONS = [uuid.UUID(int=i).hex for i in range(4000)]


def test_about_the_configured_share_of_sessions_is_sampled() -> None:
    sampled = sum(FeedbackPolicy(sample_rate=0.3).sampled(s) for s in SESSIONS)

    assert 0.27 < sampled / len(SESSIONS) < 0.33


def test_a_session_is_in_the_sample_or_not_on_every_turn() -> None:
    policy = FeedbackPolicy(sample_rate=0.3)

    assert [policy.sampled(s) for s in SESSIONS[:50]] == [policy.sampled(s) for s in SESSIONS[:50]]


@pytest.mark.parametrize(("rate", "expected"), [(0.0, False), (1.0, True)])
def test_no_session_or_every_session(rate: float, expected: bool) -> None:
    assert {FeedbackPolicy(sample_rate=rate).sampled(s) for s in SESSIONS[:200]} == {expected}


@pytest.mark.parametrize(
    ("rounds", "behavior", "asks"),
    [
        (2, Behavior.ANSWER, False),  # too early
        (3, Behavior.ANSWER, True),
        (5, Behavior.ACTION, True),
        (3, Behavior.REFUSE, False),
        (3, Behavior.ESCALATE, False),  # e.g. an outage
        (3, Behavior.CLARIFY, False),
        (3, Behavior.NO_ANSWER, False),
    ],
)
def test_only_a_reply_worth_rating_from_the_third_round_asks(
    rounds: int, behavior: Behavior, asks: bool
) -> None:
    policy = FeedbackPolicy(sample_rate=1.0, min_rounds=3)

    assert policy.asks("s1", rounds, behavior) is asks

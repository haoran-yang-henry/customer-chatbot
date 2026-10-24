"""Customer feedback (rebuild phase 3): which conversations are asked whether they helped.

Only conversations of several rounds are worth a rating, and only a share of them is asked, so
customers are not nagged. A session is in the sample or not by a hash of its id: stable across
its turns and across replays, unlike a random draw. It is asked once, at the first round that
qualifies; the turn graph's ``finish`` node decides and keeps that it asked in the checkpoint.
"""

import hashlib
from dataclasses import dataclass

from customer_agent.domain.conversation import Behavior

# What the customer came for; a refusal, a question back or an outage is no moment to ask.
_RATED = frozenset({Behavior.ANSWER, Behavior.ACTION})


@dataclass(frozen=True)
class FeedbackPolicy:
    sample_rate: float = 0.3  # share of sessions asked
    min_rounds: int = 3  # customer messages so far, this one included

    def sampled(self, session_id: str) -> bool:
        digest = hashlib.sha256(session_id.encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self.sample_rate

    def asks(self, session_id: str, rounds: int, behavior: Behavior) -> bool:
        """Whether this reply asks for a rating; the caller asks once per session."""
        return rounds >= self.min_rounds and behavior in _RATED and self.sampled(session_id)

"""The turn graph's state as checkpoints store it (rebuild phase 3)."""

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from customer_agent.application.turn_graph import CHECKPOINT_TYPES, turn_input
from customer_agent.domain.conversation import AgentResponse, Behavior, Message, Role
from customer_agent.domain.guardrails import GuardrailAction, GuardrailDecision
from customer_agent.domain.intents import Intent
from customer_agent.domain.routing import RoutingResult, RoutingSource
from customer_agent.domain.session import Decision, NextStep, SessionState, WorkflowState


def test_every_type_of_a_full_turn_state_is_allowed_in_checkpoints() -> None:
    """A type missing from the allowlist is refused when a session is loaded: it comes back
    as a plain dict, and the turn fails."""
    session = SessionState(
        active_intent=Intent.ORDER_CANCEL,
        workflow=WorkflowState.AWAITING_TICKET_CONFIRMATION,
        slots={"order_id": "NE-100245"},
    )
    state = turn_input(
        [Message(role=Role.USER, content="Bitte stornieren")],
        session,
        screen=GuardrailDecision(stage="input", action=GuardrailAction.REDACT, reasons=["pii"]),
        routing=RoutingResult(
            intent=Intent.ORDER_CANCEL,
            confidence=0.9,
            standalone_query="NE-100245 stornieren",
            source=RoutingSource.LLM,
        ),
        verdict=GuardrailDecision(stage="input", action=GuardrailAction.ALLOW),
    )
    state["decision"] = Decision(
        step=NextStep.ORDER_GUIDANCE, intent=Intent.ORDER_CANCEL, state=session
    )
    state["response"] = AgentResponse(
        answer="Storniert.", behavior=Behavior.ANSWER, intent=Intent.ORDER_CANCEL
    )
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)

    restored = serde.loads_typed(serde.dumps_typed(state))

    assert restored == state

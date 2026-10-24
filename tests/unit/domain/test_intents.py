import pytest

from customer_agent.domain.intents import Intent, IntentGroup


@pytest.mark.parametrize("intent", list(Intent))
def test_every_intent_has_a_group(intent: Intent) -> None:
    assert isinstance(intent.group, IntentGroup)


def test_the_transactional_intents_are_those_about_one_order() -> None:
    """``decide`` asks for a login exactly for these."""
    transactional = {intent for intent in Intent if intent.group is IntentGroup.TRANSACTIONAL}

    assert transactional == {
        Intent.ORDER_STATUS,
        Intent.ORDER_CANCEL,
        Intent.ORDER_CHANGE,
        Intent.RETURN_REQUEST,
        Intent.REFUND_STATUS,
    }

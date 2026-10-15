import pytest

from customer_agent.domain.intents import Intent, IntentGroup


@pytest.mark.parametrize("intent", list(Intent))
def test_every_intent_has_a_group(intent: Intent) -> None:
    assert isinstance(intent.group, IntentGroup)


def test_only_transactional_intents_require_authentication() -> None:
    requiring = {intent for intent in Intent if intent.requires_authentication}

    assert requiring == {
        Intent.ORDER_STATUS,
        Intent.ORDER_CANCEL,
        Intent.ORDER_CHANGE,
        Intent.RETURN_REQUEST,
        Intent.REFUND_STATUS,
    }

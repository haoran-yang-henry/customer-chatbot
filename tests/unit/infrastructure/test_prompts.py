"""The LangChain renderer must keep every shipped prompt and cassette key unchanged."""

from pathlib import Path
from string import Template

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from customer_agent.application.prompts import PromptTemplate
from customer_agent.infrastructure.prompts import PROMPTS_DIR, load_prompt


@pytest.mark.parametrize("path", sorted(PROMPTS_DIR.glob("*/v*.yaml")), ids=lambda p: str(p)[-20:])
def test_langchain_renders_every_prompt_byte_for_byte(path: Path) -> None:
    prompt = load_prompt(path.parent.name, int(path.stem[1:]))
    values = {
        name: f'{name}: Rückgabe\n{{"preis": "€30"}} $message ${{history}} </document>'
        for name in Template(prompt.user).get_identifiers()
    }

    expected = Template(prompt.user).substitute(values)
    messages = prompt.chat_prompt.format_messages(**values)

    assert messages == [SystemMessage(prompt.system), HumanMessage(expected)]
    assert prompt.render_messages(**values)[1].text == expected
    assert prompt.chat_prompt.metadata == {"prompt": prompt.ref}


@pytest.mark.parametrize(
    "user",
    [
        '$value ${value} {"literal": "braces"} $$value $${value}',
        "adjacent: $left$right; escaped: $$$left; multiline:\n${right}",
        'no variables: {"amount": 12} and $$',
    ],
)
def test_dollars_and_json_remain_literal(user: str) -> None:
    prompt = PromptTemplate(name="test", version=1, description="", system="$literal {}", user=user)
    values = {"value": "{user} $not_a_variable", "left": "ä", "right": "€"}

    assert prompt.render_messages(**values)[1].text == Template(user).substitute(values)
    assert prompt.render_messages(**values)[0].content == "$literal {}"


def test_missing_values_fail_instead_of_sending_an_unrendered_prompt() -> None:
    with pytest.raises(KeyError):
        load_prompt("router", 5).chat_prompt.format_messages(message="Hallo")


def test_invalid_dollar_placeholder_is_rejected() -> None:
    prompt = PromptTemplate(name="test", version=1, description="", system="", user="$ invalid")
    with pytest.raises(ValueError, match="placeholder"):
        prompt.render_messages()

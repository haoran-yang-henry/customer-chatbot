"""Prompt template model. Loading from ``prompts/`` lives in ``infrastructure/prompts.py``."""

from functools import cached_property
from string import Template

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict

from customer_agent.application.ports import ChatMessage


def _chat_template(text: str) -> str:
    """Translate dollar placeholders without interpreting JSON braces or escaped dollars."""
    if not Template(text).is_valid():
        raise ValueError("invalid dollar placeholder in prompt")
    parts: list[str] = []
    end = 0
    for match in Template.pattern.finditer(text):
        parts.append(text[end : match.start()].replace("{", "{{").replace("}", "}}"))
        name = match.group("named") or match.group("braced")
        parts.append("{" + name + "}" if name else "$")
        end = match.end()
    parts.append(text[end:].replace("{", "{{").replace("}", "}}"))
    return "".join(parts)


class PromptTemplate(BaseModel):
    """Versioned YAML prompt, compiled to LangChain without changing its rendered text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: int
    description: str
    system: str
    user: str

    @property
    def ref(self) -> str:
        return f"{self.name}/v{self.version}"

    @cached_property
    def chat_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate(
            [SystemMessage(content=self.system), ("human", _chat_template(self.user))],
            metadata={"prompt": self.ref},
        )

    def render_messages(self, **values: str) -> list[ChatMessage]:
        """Bridge to the application port until the agents consume ChatPromptValue directly."""
        system, user = self.chat_prompt.format_messages(**values)
        return [
            ChatMessage(role="system", content=system.text),
            ChatMessage(role="user", content=user.text),
        ]

    def render_user(self, **values: str) -> str:
        return self.render_messages(**values)[1].content

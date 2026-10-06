"""Prompt template model. Loading from ``prompts/`` lives in ``infrastructure/prompts.py``."""

from string import Template

from pydantic import BaseModel, ConfigDict


class PromptTemplate(BaseModel):
    """``$placeholders`` (``string.Template``) so JSON or braces in prompts need no escaping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: int
    description: str
    system: str
    user: str

    @property
    def ref(self) -> str:
        return f"{self.name}/v{self.version}"

    def render_user(self, **values: str) -> str:
        return Template(self.user).substitute(values)

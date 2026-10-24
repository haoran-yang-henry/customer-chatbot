"""``customer-agent-prompts-sync``: publish every prompt version from ``prompts/`` to Langfuse.

Git stays the source (rebuild phase 4). Langfuse gets each version as a chat prompt with the
label ``v<version>``; the versions this configuration uses also get ``production``. A version
Langfuse already has with the same text is left alone, so the command can run after every
change. Needs ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``.
"""

from langfuse import Langfuse

from customer_agent.bootstrap import prompts_in_use
from customer_agent.config import get_settings
from customer_agent.infrastructure.prompts import PROMPTS_DIR, load_prompt
from customer_agent.observability.langfuse_prompts import sync_prompt


def main() -> int:
    settings = get_settings()
    key, secret = settings.langfuse_public_key, settings.langfuse_secret_key
    if key is None or secret is None:
        print("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are needed")
        return 2
    client = Langfuse(
        public_key=key,
        secret_key=secret.get_secret_value(),
        base_url=settings.langfuse_base_url,
        tracing_enabled=False,  # API calls only
    )
    in_use = set(prompts_in_use(settings))
    versions = sorted(
        (path.parent.name, int(path.stem.removeprefix("v")))
        for path in PROMPTS_DIR.glob("*/v*.yaml")
    )
    for name, version in versions:
        prompt = load_prompt(name, version)
        outcome = sync_prompt(client, prompt, production=(name, version) in in_use)
        print(f"{prompt.ref:<12} {outcome}")
    client.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

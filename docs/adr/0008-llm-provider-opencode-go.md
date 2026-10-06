# ADR-0008: Chat model via an OpenAI-compatible endpoint, OpenCode Go for development

- **Status:** Accepted
- **Date:** 2026-10-06
- **Deciders:** Project owner
- **Related:** requirements Q1, [ADR-0007](0007-openai-embeddings.md), NFR-02, NFR-03, NFR-14

## Context

Day 3 introduces answer generation: German answers grounded in retrieved evidence, with
citations. Development needs many LLM calls — every evaluation run is ~70 answer calls plus judge
calls — so per-token cost directly limits how often quality can be measured. The project owner
has an OpenCode Go subscription (flat monthly fee, 40+ open models such as DeepSeek, GLM, Kimi,
Qwen) and an OpenAI API key.

## Options considered

1. **OpenAI chat models** — stable quality, per-token billing.
2. **OpenCode Go** — flat fee, OpenAI-compatible endpoint (`https://opencode.ai/zen/go/v1`),
   zero-retention policy; designed primarily for coding agents, usage is monitored for abuse,
   no embedding models.
3. **Self-hosted open model** — no external dependency, but no suitable hardware (4 GB GPU).

## Decision

- The application talks to **any OpenAI-compatible chat completions endpoint** through one
  adapter behind the `LLMProvider` port. Provider, model and key are configuration
  (`CA_LLM_BASE_URL`, `CA_LLM_MODEL`, `CA_LLM_API_KEY`), separate from the embedding settings.
- **Development and evaluation use OpenCode Go**, default model `deepseek-v4-flash`
  (fast, inexpensive, good German). Other models are compared on the evaluation set by changing
  one setting.
- **Honest client identification.** OpenCode Go requires a real `User-Agent` and a stable
  per-conversation `x-opencode-session` header. The client identifies itself as
  `nordelektro-customer-agent/<version>` and sends a session id per conversation
  (`CA_LLM_USER_AGENT`, `CA_LLM_SESSION_HEADER`). It never impersonates a coding agent.
- **Accepted risk.** OpenCode Go's usage policy describes it as designed for coding agents and
  monitors traffic. Customer-service traffic does not match that description; the project owner
  accepts the risk of throttling or suspension of the subscription for this learning project.
  Switching to OpenAI (verified on Day 3: `gpt-5.2` with `CA_LLM_REASONING_EFFORT=none`,
  ~1.8 s, correct German with citations) is a configuration change.
- Embeddings stay on OpenAI (ADR-0007); OpenCode Go offers no embedding models.
- Evaluation runs may enable an on-disk response cache (`CA_LLM_CACHE_DIR`) so that unchanged
  prompts are not sent again.

## Consequences

**Positive**

- Evaluation can be run as often as needed without per-token cost.
- No vendor lock-in: OpenAI, Azure OpenAI or another compatible gateway is a configuration change.

**Negative / trade-offs**

- Provider terms are an engineering constraint, not an afterthought: technical compatibility
  ("the API call works") does not imply permitted use.
- A personal subscription is not a production contract. A real deployment needs a provider with a
  data processing agreement and an SLA; this ADR would then be superseded.
- Open models differ in instruction following; prompt behaviour must be re-validated with the
  evaluation set whenever the model changes.
- Two providers (embeddings, chat) means two keys and two failure points (fallbacks: Day 7).

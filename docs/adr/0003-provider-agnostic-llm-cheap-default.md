# ADR 0003: Provider-agnostic LLM layer with a cheap default

## Context
The project should not be locked to one vendor, and LLM spend (agent, judge, evals) is the only
variable cost. Providers differ in structured-output support.

## Decision
One factory (`llm/factory.py`) returns a LangChain chat model for DeepSeek (default), OpenAI or
Anthropic, chosen by `LLM_PROVIDER` / `LLM_MODEL`. The judge model is configured separately
(`JUDGE_PROVIDER` / `JUDGE_MODEL`) so the model under test is not its own judge. Every LLM call
returns a Pydantic object through structured output, using function calling for DeepSeek (which has
no strict JSON-schema mode). Optional persistent response cache for dev and eval reruns.

## Consequences
- A full question costs about $0.001 on DeepSeek (agent) plus a similar judge cost, versus a few cents on larger models; a full 38-question
  eval run is about $0.04 for the agent and $0.07 for the judge.
- DeepSeek occasionally answers in plain text instead of calling the schema function; structured
  calls retry with a changed prompt (so a cached bad answer is not replayed) and degrade gracefully.
- Model names live in config only; provider prices live in one table used for cost reports.
- Only DeepSeek has been run end to end so far; the provider-comparison table has one row until
  OpenAI/Anthropic keys are added (`eval.run_eval --provider ...`).

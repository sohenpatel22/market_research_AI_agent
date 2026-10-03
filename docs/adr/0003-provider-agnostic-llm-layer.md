# ADR 0003: Provider-agnostic LLM layer with a cost-efficient default

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
- A full question costs about $0.001 on DeepSeek, including routing, generation and the grading
  call. A complete 38-question eval run costs about $0.04 for the agent plus about $0.07 for the
  RAGAS judge.
- DeepSeek occasionally answers in plain text instead of calling the schema function; structured
  calls retry with a changed prompt (so a cached bad answer is not replayed) and degrade gracefully.
- Model names live in config only; provider prices live in one table used for cost reports.
- Measured, not assumed: DeepSeek, GPT-4o mini, Claude Haiku 4.5 and Claude Sonnet 5.5 were run on the
  same 69 golden questions (see the README's provider comparison). DeepSeek passed every deterministic
  check at $0.0008 per question; GPT-4o mini had a lower unit cost but over-refused (4 legitimate questions declined);
  Sonnet 5.5 cost about 14x more for the same faithfulness. Claude 5.x models also reject an explicit
  temperature, which the factory now leaves unset for them.

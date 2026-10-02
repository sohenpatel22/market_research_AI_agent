# ADR 0001: A bounded grade-and-retry loop instead of a single pass or a free-running agent

## Context
A single retrieve-then-answer pass sometimes answers from weak evidence, and the model cannot tell
when that happened. A fully autonomous tool-calling agent can loop, burn tokens and is hard to test.

## Decision
A LangGraph state machine with a fixed shape: route, gather, generate, **grade**, and, only on
failure, **rewrite** the search query using the grader's feedback and try again. The loop is capped
(`AGENT_MAX_RETRIES`, default 2) and LangGraph's recursion limit is a second guard. Tool results that do
not depend on the query (forecasts, SQL lookups) are computed once, not per retry.

## Consequences
- Cost and latency are bounded and predictable: 3 retrieval passes at worst; a typical question is about 4.5k tokens.
- If the grader itself fails, retrying cannot help, so the loop stops and the answer is marked unverified rather than crashing.
- The final `AgentAnswer` carries `quality_passed`, so a weak answer is shown as weak instead of confidently wrong.
- A fixed graph is easy to unit-test with fake LLMs (the retry bound is asserted in tests).
- Cost of the design: every question pays for at least one extra LLM call (the grader), and a retry
  repeats generation and grading; the per-question cost below includes both.

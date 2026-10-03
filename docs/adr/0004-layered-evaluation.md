# ADR 0004: Layered evaluation: deterministic checks, RAGAS in dev, DeepEval in CI

## Context
LLM-judged metrics are useful but noisy and cost money; some behaviours (refusals, tool use, numeric
facts) can be checked exactly. A CI gate must be cheap, fast and not flaky.

## Decision
1. **Deterministic checks (free):** refusal exactly when expected, forecast/SQL tools actually called,
   numeric facts from the database present in the answer (tolerant of "$90.0 billion" vs
   "90,007,000,000"), citations present.
2. **RAGAS (dev loop, run deliberately):** faithfulness, answer relevancy, context precision and recall on the 50 filing questions, with cost and latency per run.
3. **DeepEval (CI gate):** faithfulness and relevancy on 6 golden items, against a small committed corpus so it runs on an empty database, and skipped when no API key is present.
4. One thresholds file (`eval/thresholds.yaml`) for all of it.

## Consequences
- Baseline (DeepSeek, n = 99): faithfulness 0.95, answer relevancy 0.86, context precision 0.83,
  context recall 0.92, refusal accuracy, tool-use and numeric-fact rates 100%, unanswerable questions
  handled honestly 7/7, cross-company comparisons citing every company 4/6, about $0.0013 per question.
- DeepEval's relevancy is a fraction of statements judged on-topic, so it is coarse: the same
  correct answer scored between 0.45 and 1.0 across runs. The CI threshold was calibrated to 0.55
  and the reason is written next to it; RAGAS answer relevancy is the finer-grained measure.
- The golden questions were drafted by an LLM from sampled filing chunks (so the ground-truth source is
  known by construction) and then curated by hand; 99 items is enough to catch regressions, not to
  claim fine-grained accuracy differences.
- Provider comparison and retrieval ablations reuse the same harness.

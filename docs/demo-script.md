# 90-second demo script

Run `make docker-up`, open <http://localhost:7860>. Record at 1280x800 with the browser zoomed to 110%.
Have the Langfuse project open in a second tab for the last beat.

| Time | Do | Say (about) |
|---|---|---|
| 0:00 | Open the **Ask the agent** tab | "This is a research assistant over SEC filings, prices and trained forecasting models. Everything it says is cited and quality-checked." |
| 0:08 | Click the first example: *What supply chain risks does Apple report, and what is the model's AAPL volatility outlook?* | "One question, two kinds of evidence: filing text and a model forecast." |
| 0:15 | Point at the progress checklist as it fills in | "It routes the question, retrieves excerpts with hybrid search and a reranker, runs the forecaster, drafts an answer, and then a judge checks it against the sources." |
| 0:35 | Scroll to the answer, sources table and forecast block | "Citations point to the exact filing, section and date. The forecast shows the LSTM next to the HAR baseline and says it is an uncertain estimate." |
| 0:45 | Ask: *Should I put my savings into NVDA?* | "Out-of-scope requests are refused at the first step, before any tool runs." |
| 0:55 | Ask: *What does NVIDIA say about export controls on its data center products?* | "If the grader isn't satisfied, it rewrites the search and retries, capped at two retries. Here it passed (or: it retried once; the status line shows how many)." |
| 1:10 | **Forecast** tab: AAPL, volatility | "The model forecast against realized volatility and the HAR baseline. On held-out data the LSTM beat HAR with a Diebold-Mariano p of 0.001." |
| 1:20 | Click the `trace` link under an answer (Langfuse) | "Every question is one trace: each step, token counts, cost and latency, plus the grader's verdict as scores." |
| 1:30 | End | "Evaluated with a golden set, RAGAS and a DeepEval CI gate; about a tenth of a cent per question." |

Notes for a clean take:
- The retry beat is non-deterministic; if it passes first time, say so and move on, don't re-record it.
- Keep the forecast tab on a ticker whose chart has a clear forecast marker (AAPL, MSFT).
- Don't claim the direction model is predictive; the UI says it is weak, and so should you.

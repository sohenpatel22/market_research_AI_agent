# ADR 0002: Hybrid retrieval, section-aware chunks, and a cross-encoder reranker

## Context
Filings mix prose (risk factors, MD&A) with tables and boilerplate. Pure vector search misses exact
terms; pure keyword search misses paraphrases. Fixed-size chunks cut across the structure of a 10-K.

## Decision
- Chunk by SEC `Item` heading (table-of-contents lines are merged away), keeping the section as metadata.
- Retrieve with pgvector (HNSW, cosine) and Postgres full-text (GIN), fuse the rankings with
  reciprocal rank fusion, then rerank the pool with a cross-encoder.
- Measure it: a retrieval-only ablation on the 23 filing questions of the golden set (no LLM calls).

## Evidence (k = 6)

| Retriever | NDCG@6 | Recall@6 | MRR |
|---|---|---|---|
| dense | 0.483 | 0.826 | 0.605 |
| sparse (keywords) | 0.250 | 0.522 | 0.333 |
| hybrid (RRF) | 0.447 | 0.783 | 0.570 |
| **hybrid + rerank** | **0.591** | **0.913** | **0.772** |

## Consequences
- The honest finding is that plain hybrid slightly *trailed* dense-only here: the keyword leg
  alone is weak. The reranker is what pays off (+0.14 NDCG@6), so it is on by default.
- Down-weighting the keyword leg helped un-reranked hybrid but not the reranked pipeline, so the
  standard equal-weight RRF stays.
- With 23 questions these differences are indicative, not statistically established.
- Cost: the reranker adds a 90 MB model and some CPU latency per question; it is baked into the Docker image.

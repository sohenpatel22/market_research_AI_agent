# ADR 0005: One PostgreSQL (pgvector) for vectors, text search, tables and BI

## Context
The system needs a vector index, keyword search, structured price/fundamentals tables, published
model output for dashboards, and something simple to deploy.

## Decision
Use one PostgreSQL 16+ with the pgvector extension: HNSW index for embeddings, a generated
`tsvector` column with a GIN index for keywords, ordinary tables for prices and fundamentals, and
views plus published tables for Power BI.

## Consequences
- Hybrid retrieval, the SQL tool and BI all read the same database; one service to run and secure.
- The SQL tool never accepts SQL: the model picks one of three whitelisted, parameterized lookups
  and runs it in a read-only transaction.
- Managed Postgres works the same locally and in production (Neon): the connection layer
  normalizes `postgres://` URLs, uses timeouts and recycling suited to serverless databases, and
  disables prepared statements behind a pooler.
- Ingestion keeps transactions short (embedding thousands of chunks takes minutes, and a serverless
  database drops idle connections); this was found when the first production seed failed.
- Limits: ~5,400 chunks is tiny. A much larger corpus would call for a dedicated vector store or
  partitioning, which the retriever interface would allow without touching the agent.

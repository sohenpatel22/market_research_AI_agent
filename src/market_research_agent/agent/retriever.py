"""Hybrid retrieval over the pgvector documents table"""

import re
from collections.abc import Callable
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.orm import Session

from market_research_agent.agent.schemas import RetrievedChunk
from market_research_agent.config import settings

RRF_K = 60
_FILTERS = """
    (CAST(:tickers AS text[]) IS NULL OR ticker = ANY(CAST(:tickers AS text[])))
    AND (CAST(:forms AS text[]) IS NULL OR filing_type = ANY(CAST(:forms AS text[])))
"""
_COLS = "id, ticker, filing_type, filed_date, section, accession_number, chunk_index, chunk_text"

DENSE_SQL = text(
    f"SELECT {_COLS} FROM documents WHERE {_FILTERS} "
    "ORDER BY embedding <=> CAST(:emb AS vector) LIMIT :n"
)
SPARSE_SQL = text(
    f"SELECT {_COLS} FROM documents, to_tsquery('english', :q) AS query "
    f"WHERE fts @@ query AND {_FILTERS} ORDER BY ts_rank_cd(fts, query) DESC LIMIT :n"
)


def to_or_tsquery(query: str, max_terms: int = 12) -> str:
    """Sanitize free text into an OR-joined tsquery (safe: alphanumerics only)"""
    seen: list[str] = []
    for tok in re.findall(r"[A-Za-z0-9]+", query.lower()):
        if len(tok) >= 3 and tok not in seen:
            seen.append(tok)
    return " | ".join(seen[:max_terms])


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank)"""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(
    session: Session,
    query: str,
    query_embedding: list[float],
    k: int = 6,
    tickers: list[str] | None = None,
    filing_types: list[str] | None = None,
    candidates: int = 30,
    mode: str = "hybrid",
) -> list[RetrievedChunk]:
    params = {
        "tickers": [t.upper() for t in tickers] if tickers else None,
        "forms": filing_types or None,
        "n": candidates,
    }
    # `mode` exists for ablations: "dense" (vectors only), "sparse" (keywords only), "hybrid".
    dense = []
    if mode in ("dense", "hybrid"):
        emb = {**params, "emb": _vector_literal(query_embedding)}
        dense = session.execute(DENSE_SQL, emb).all()
    sparse = []
    sparse_q = to_or_tsquery(query)
    if mode in ("sparse", "hybrid") and sparse_q:
        sparse = session.execute(SPARSE_SQL, {**params, "q": sparse_q}).all()

    rows = {r.id: r for r in [*dense, *sparse]}
    scores = rrf_fuse([[r.id for r in dense], [r.id for r in sparse]])
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [
        RetrievedChunk(
            id=i,
            ticker=rows[i].ticker,
            filing_type=rows[i].filing_type,
            filed_date=rows[i].filed_date,
            section=rows[i].section,
            accession_number=rows[i].accession_number,
            chunk_index=rows[i].chunk_index,
            text=rows[i].chunk_text,
            score=scores[i],
        )
        for i in top
    ]


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model_name)


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Rescore chunks with a cross-encoder (query and passage read together)"""
    if not chunks:
        return chunks
    scores = _cross_encoder().predict([(query, c.text) for c in chunks])
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda p: p[1], reverse=True)
    return [c.model_copy(update={"score": float(s)}) for c, s in ranked[:top_k]]


def retrieve(
    query: str,
    tickers: list[str] | None = None,
    filing_types: list[str] | None = None,
    k: int = 6,
    use_rerank: bool | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    session: Session | None = None,
    mode: str = "hybrid",
) -> list[RetrievedChunk]:
    """Embed the query, run hybrid search and (optionally) rerank"""
    from market_research_agent.data.db import session_scope
    from market_research_agent.data.embeddings import embed_texts

    embed = embed_fn or embed_texts
    use_rerank = settings.use_reranker if use_rerank is None else use_rerank
    pool = max(k * 3, 15) if use_rerank else k
    emb = embed([query])[0]

    def run(s: Session) -> list[RetrievedChunk]:
        return hybrid_search(s, query, emb, pool, tickers, filing_types, mode=mode)

    if session is not None:
        found = run(session)
    else:
        with session_scope() as s:
            found = run(s)
    return rerank(query, found, k) if use_rerank else found[:k]

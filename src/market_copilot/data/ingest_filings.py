"""Ingest recent 10-K/10-Q filings for a ticker: download -> clean -> chunk -> embed -> insert."""

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from market_copilot.data.chunking import chunk_text, clean_filing_text
from market_copilot.data.edgar import FilingRef, fetch_filing_text, get_recent_filings
from market_copilot.data.embeddings import embed_texts
from market_copilot.data.models import Document


def _chunk_filing(filing: FilingRef) -> list[dict]:
    cleaned = clean_filing_text(fetch_filing_text(filing))
    chunks = chunk_text(cleaned)
    if not chunks:
        return []

    embeddings = embed_texts(chunks)
    return [
        {
            "ticker": filing.ticker,
            "filing_type": filing.filing_type,
            "filed_date": filing.filed_date,
            "accession_number": filing.accession_number,
            "chunk_index": i,
            "chunk_text": chunk,
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
    ]


def ingest_filings(session: Session, ticker: str, limit_per_form: int = 1) -> int:
    """Fetch, chunk, embed, and upsert filings for a ticker. Returns rows inserted."""
    rows = []
    for filing in get_recent_filings(ticker, limit_per_form=limit_per_form):
        rows.extend(_chunk_filing(filing))

    if not rows:
        return 0

    stmt = insert(Document).values(rows)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=["accession_number", "chunk_index"]
    ).returning(Document.id)
    # psycopg3 rowcount is -1 for multi-row INSERT ON CONFLICT; count via RETURNING.
    return len(session.execute(stmt).fetchall())

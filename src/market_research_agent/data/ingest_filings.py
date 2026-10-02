"""Ingest recent 10-K/10-Q filings for a ticker: download -> clean -> chunk -> embed -> insert.

The slow part (downloading and embedding thousands of chunks) is separated from the database
work, so callers can avoid holding a connection open while embedding: serverless Postgres such
as Neon closes idle connections after a few minutes. Filings already in the database are skipped
before any embedding happens, which also makes re-runs cheap.
"""

from collections.abc import Collection

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from market_research_agent.data.chunking import chunk_by_section, clean_filing_text
from market_research_agent.data.edgar import FilingRef, fetch_filing_text, get_recent_filings
from market_research_agent.data.embeddings import embed_texts
from market_research_agent.data.models import Document

INSERT_BATCH_SIZE = 500


def _chunk_filing(filing: FilingRef) -> list[dict]:
    cleaned = clean_filing_text(fetch_filing_text(filing))
    sectioned = chunk_by_section(cleaned)
    if not sectioned:
        return []

    chunks = [chunk for _, chunk in sectioned]
    embeddings = embed_texts(chunks)
    return [
        {
            "ticker": filing.ticker,
            "filing_type": filing.filing_type,
            "filed_date": filing.filed_date,
            "accession_number": filing.accession_number,
            "chunk_index": i,
            "section": section,
            "chunk_text": chunk,
            "embedding": embedding,
        }
        for i, ((section, chunk), embedding) in enumerate(zip(sectioned, embeddings, strict=True))
    ]


def existing_accessions(session: Session) -> set[str]:
    """Accession numbers of filings that already have chunks in the database."""
    return set(session.execute(select(Document.accession_number).distinct()).scalars())


def prepare_filing_rows(
    ticker: str, limit_per_form: int = 1, skip_accessions: Collection[str] = ()
) -> list[dict]:
    """Download, chunk and embed recent filings. Touches the network and CPU, not the database."""
    rows: list[dict] = []
    for filing in get_recent_filings(ticker, limit_per_form=limit_per_form):
        if filing.accession_number in skip_accessions:
            continue
        rows.extend(_chunk_filing(filing))
    return rows


def insert_filing_rows(session: Session, rows: list[dict]) -> int:
    """Insert prepared rows in batches (idempotent). Returns rows inserted."""
    inserted = 0
    for start in range(0, len(rows), INSERT_BATCH_SIZE):
        stmt = insert(Document).values(rows[start : start + INSERT_BATCH_SIZE])
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["accession_number", "chunk_index"]
        ).returning(Document.id)
        # psycopg3 rowcount is -1 for multi-row INSERT ON CONFLICT; count via RETURNING.
        inserted += len(session.execute(stmt).fetchall())
    return inserted


def ingest_filings(session: Session, ticker: str, limit_per_form: int = 1) -> int:
    """Fetch, chunk, embed, and upsert filings for a ticker. Returns rows inserted."""
    rows = prepare_filing_rows(ticker, limit_per_form, skip_accessions=existing_accessions(session))
    return insert_filing_rows(session, rows)

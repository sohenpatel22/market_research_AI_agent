from sqlalchemy import select
from sqlalchemy.orm import Session

from market_research_agent.data import ingest_filings
from market_research_agent.data.edgar import FilingRef
from market_research_agent.data.models import EMBEDDING_DIM, Document

# Not a real ticker/accession number, so this can never collide with live data
# sitting in the same dev database the tests run against.
TEST_TICKER = "ZZZTEST"


def _fake_filing() -> FilingRef:
    return FilingRef(
        ticker=TEST_TICKER,
        cik="0000000000",
        accession_number="0000000000-99-999999",
        filing_type="10-K",
        filed_date="2024-11-01",
        primary_document="fake-20240928.htm",
    )


def test_ingest_filings_inserts_chunks(db_session: Session, monkeypatch):
    filing = _fake_filing()
    monkeypatch.setattr(
        ingest_filings, "get_recent_filings", lambda ticker, limit_per_form=1: [filing]
    )
    monkeypatch.setattr(ingest_filings, "fetch_filing_text", lambda f: "Item 1. Business. " * 500)
    monkeypatch.setattr(
        ingest_filings,
        "embed_texts",
        lambda texts: [[0.0] * EMBEDDING_DIM for _ in texts],
    )

    n_inserted = ingest_filings.ingest_filings(db_session, TEST_TICKER)

    assert n_inserted > 0
    rows = (
        db_session.execute(select(Document).where(Document.ticker == TEST_TICKER)).scalars().all()
    )
    assert len(rows) == n_inserted
    assert len(rows[0].embedding) == EMBEDDING_DIM


def test_ingest_filings_is_idempotent(db_session: Session, monkeypatch):
    filing = _fake_filing()
    monkeypatch.setattr(
        ingest_filings, "get_recent_filings", lambda ticker, limit_per_form=1: [filing]
    )
    monkeypatch.setattr(ingest_filings, "fetch_filing_text", lambda f: "Item 1. Business. " * 500)
    monkeypatch.setattr(
        ingest_filings,
        "embed_texts",
        lambda texts: [[0.0] * EMBEDDING_DIM for _ in texts],
    )

    ingest_filings.ingest_filings(db_session, TEST_TICKER)
    n_second_run = ingest_filings.ingest_filings(db_session, TEST_TICKER)

    assert n_second_run == 0


def test_existing_filings_are_skipped_before_embedding(db_session: Session, monkeypatch):
    filing = _fake_filing()
    embedded: list[int] = []

    def fake_embed(texts):
        embedded.append(len(texts))
        return [[0.0] * EMBEDDING_DIM for _ in texts]

    monkeypatch.setattr(
        ingest_filings, "get_recent_filings", lambda ticker, limit_per_form=1: [filing]
    )
    monkeypatch.setattr(ingest_filings, "fetch_filing_text", lambda f: "Item 1. Business. " * 500)
    monkeypatch.setattr(ingest_filings, "embed_texts", fake_embed)

    assert ingest_filings.ingest_filings(db_session, TEST_TICKER) > 0
    assert len(embedded) == 1
    assert ingest_filings.ingest_filings(db_session, TEST_TICKER) == 0
    assert len(embedded) == 1  # the second run never re-embedded the stored filing


def test_rows_are_inserted_in_batches(db_session: Session, monkeypatch):
    monkeypatch.setattr(ingest_filings, "INSERT_BATCH_SIZE", 7)
    rows = [
        {
            "ticker": TEST_TICKER,
            "filing_type": "10-K",
            "filed_date": "2024-11-01",
            "accession_number": "0000000000-99-999998",
            "chunk_index": i,
            "section": "Item 1",
            "chunk_text": f"chunk {i}",
            "embedding": [0.0] * EMBEDDING_DIM,
        }
        for i in range(20)
    ]
    assert ingest_filings.insert_filing_rows(db_session, rows) == 20
    assert ingest_filings.insert_filing_rows(db_session, rows) == 0  # idempotent

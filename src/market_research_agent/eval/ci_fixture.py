"""A tiny committed corpus so the CI eval gate can run against an empty database"""

import argparse
import json
import random
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from market_research_agent.data.db import init_db, session_scope
from market_research_agent.data.embeddings import embed_texts
from market_research_agent.data.models import Document, Price
from market_research_agent.eval.checks import load_thresholds
from market_research_agent.eval.golden import load_golden

FIXTURE_PATH = Path("tests/eval/fixtures/ci_corpus.json")
DISTRACTORS_PER_TICKER = 12
PRICE_ROWS = 60


def export(path: Path = FIXTURE_PATH) -> None:
    ci_ids = set(load_thresholds()["deepeval"]["ci_items"])
    items = [i for i in load_golden() if i.id in ci_ids]
    tickers = sorted({t for i in items for t in i.expected_tickers})
    rng = random.Random(0)
    chunks: dict[tuple[str, int], dict] = {}
    with session_scope() as s:
        for item in items:
            if not item.source:
                continue
            src = item.source
            rows = s.execute(
                text(
                    "SELECT ticker, filing_type, filed_date, accession_number, chunk_index, "
                    "section, chunk_text FROM documents WHERE accession_number = :a "
                    "AND chunk_index BETWEEN :lo AND :hi"
                ),
                {"a": src.accession_number, "lo": src.chunk_index - 2, "hi": src.chunk_index + 2},
            ).all()
            for r in rows:
                chunks[(r.accession_number, r.chunk_index)] = _row(r)
        for ticker in tickers:
            rows = s.execute(
                text(
                    "SELECT ticker, filing_type, filed_date, accession_number, chunk_index, "
                    "section, chunk_text FROM documents WHERE ticker = :t "
                    "AND length(chunk_text) > 500 ORDER BY id"
                ),
                {"t": ticker},
            ).all()
            for r in rng.sample(rows, min(DISTRACTORS_PER_TICKER, len(rows))):
                chunks.setdefault((r.accession_number, r.chunk_index), _row(r))
        prices = []
        for ticker in tickers:
            rows = s.execute(
                text(
                    "SELECT ticker, date, open, high, low, close, adj_close, volume FROM prices "
                    "WHERE ticker = :t ORDER BY date DESC LIMIT :n"
                ),
                {"t": ticker, "n": PRICE_ROWS},
            ).all()
            prices += [{**r._asdict(), "date": str(r.date)} for r in reversed(rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"chunks": list(chunks.values()), "prices": prices}, indent=1), encoding="utf-8"
    )
    print(f"Wrote {len(chunks)} chunks and {len(prices)} price rows to {path}")


def _row(r) -> dict:
    return {
        "ticker": r.ticker,
        "filing_type": r.filing_type,
        "filed_date": str(r.filed_date),
        "accession_number": r.accession_number,
        "chunk_index": r.chunk_index,
        "section": r.section,
        "chunk_text": r.chunk_text,
    }


def load(session: Session | None = None, path: Path = FIXTURE_PATH) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))

    def _load(s: Session) -> None:
        chunks = data["chunks"]
        embeddings = embed_texts([c["chunk_text"] for c in chunks])
        rows = [{**c, "embedding": e} for c, e in zip(chunks, embeddings, strict=True)]
        s.execute(
            insert(Document)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["accession_number", "chunk_index"])
        )
        s.execute(
            insert(Price)
            .values(data["prices"])
            .on_conflict_do_nothing(index_elements=["ticker", "date"])
        )

    if session is not None:
        _load(session)
    else:
        init_db()
        with session_scope() as s:
            _load(s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "load"])
    args = parser.parse_args()
    export() if args.action == "export" else load()


if __name__ == "__main__":
    main()

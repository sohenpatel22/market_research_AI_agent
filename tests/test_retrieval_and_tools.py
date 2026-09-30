import contextlib
import datetime as dt

import pytest

from market_research_agent.agent import tools
from market_research_agent.agent.retriever import hybrid_search, rrf_fuse, to_or_tsquery
from market_research_agent.agent.schemas import DataRequest
from market_research_agent.agent.tools import ToolInputError, sql_tool
from market_research_agent.data.chunking import chunk_by_section, split_sections
from market_research_agent.data.models import EMBEDDING_DIM, Document, Fundamental, Price

TICKER = "ZZZR"  # never collides with live data in the shared dev database


def unit(i: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    return v


def test_rrf_rewards_agreement_between_rankings():
    scores = rrf_fuse([[1, 2, 3], [3, 1, 4]])
    assert max(scores, key=scores.get) == 1
    assert scores[1] > scores[2] and scores[3] > scores[4]


def test_tsquery_is_sanitized():
    q = to_or_tsquery("What's Apple's supply-chain risk?; DROP TABLE documents; --")
    assert "drop" in q and ";" not in q and "'" not in q and "-" not in q
    assert to_or_tsquery("a an of") == ""


@pytest.fixture
def docs(db_session):
    rows = [
        ("supply chain disruption risk in Asia manufacturing", unit(0), "Item 1A"),
        ("cloud revenue growth and operating margin expansion", unit(1), "Item 7"),
        ("pending litigation and legal proceedings", unit(2), "Item 3"),
    ]
    for i, (text, emb, section) in enumerate(rows):
        db_session.add(
            Document(
                ticker=TICKER,
                filing_type="10-K",
                filed_date=dt.date(2025, 1, 1),
                accession_number="0000000000-25-000001",
                chunk_index=i,
                section=section,
                chunk_text=text,
                embedding=emb,
            )
        )
    db_session.flush()
    return db_session


def test_hybrid_search_fuses_dense_and_keyword_hits(docs):
    # Dense likes doc 1 (revenue); keywords like doc 0 (supply chain). Both must surface.
    got = hybrid_search(docs, "supply chain risk", unit(1), k=3, tickers=[TICKER])
    texts = [c.text for c in got]
    assert any("supply chain" in t for t in texts) and any("cloud revenue" in t for t in texts)
    assert got[0].section in {"Item 1A", "Item 7"} and got[0].ticker == TICKER


def test_hybrid_search_respects_filters(docs):
    assert hybrid_search(docs, "supply chain", unit(0), tickers=["NOPE"]) == []
    assert (
        hybrid_search(docs, "supply chain", unit(0), tickers=[TICKER], filing_types=["10-Q"]) == []
    )
    assert hybrid_search(docs, "supply chain", unit(0), tickers=[TICKER], filing_types=["10-K"])


def test_sql_tool_rejects_bad_input():
    with pytest.raises(ToolInputError):
        sql_tool(DataRequest(kind="latest_price", ticker="AAPL; DROP TABLE prices"))
    with pytest.raises(ToolInputError):
        sql_tool(DataRequest(kind="fundamental", ticker="AAPL"))  # metric required
    with pytest.raises(ToolInputError):
        sql_tool(DataRequest(kind="fundamental", ticker="AAPL", metric="x'; DELETE FROM prices"))


@pytest.fixture
def sql_db(db_session, monkeypatch):
    for d, close in [(dt.date(2026, 1, 1), 100.0), (dt.date(2026, 1, 31), 110.0)]:
        db_session.add(
            Price(
                ticker=TICKER, date=d, open=1, high=1, low=1, close=close, adj_close=close, volume=5
            )
        )
    db_session.add(
        Fundamental(ticker=TICKER, period=dt.date(2025, 12, 31), metric="Total Revenue", value=5e9)
    )
    db_session.flush()
    monkeypatch.setattr(tools, "session_scope", lambda: contextlib.nullcontext(db_session))
    return db_session


def test_sql_tool_lookups(sql_db):
    latest = sql_tool(DataRequest(kind="latest_price", ticker=TICKER.lower()))
    assert "110.00" in latest.summary
    change = sql_tool(DataRequest(kind="price_change", ticker=TICKER, days=60))
    assert change.rows[0]["pct_change"] == 10.0
    fund = sql_tool(DataRequest(kind="fundamental", ticker=TICKER, metric="revenue"))
    assert fund.rows[0]["value"] == 5e9
    missing = sql_tool(DataRequest(kind="fundamental", ticker=TICKER, metric="Nonexistent"))
    assert "No fundamentals" in missing.summary
    with pytest.raises(ToolInputError):  # LIKE wildcards are not valid metric names
        sql_tool(DataRequest(kind="fundamental", ticker=TICKER, metric="100%"))


def test_sql_tool_is_read_only(sql_db):
    sql_tool(DataRequest(kind="latest_price", ticker=TICKER))
    with pytest.raises(Exception, match="read-only"):
        sql_db.execute(tools.text("DELETE FROM prices"))


def test_section_aware_chunking():
    body_a, body_b = "Risk sentence about supply chains. " * 30, "Revenue rose this quarter. " * 30
    text = f"Item 1.\nBusiness\n\nItem 1A. Risk Factors\n{body_a}\nItem 7. MD&A\n{body_b}"
    sections = split_sections(text)
    assert [s for s, _ in sections] == ["Item 1A", "Item 7"]  # short TOC-like Item 1 is merged
    pairs = chunk_by_section(text, chunk_size=400, chunk_overlap=0)
    assert {s for s, _ in pairs} == {"Item 1A", "Item 7"}
    assert all("Revenue" not in c for s, c in pairs if s == "Item 1A")

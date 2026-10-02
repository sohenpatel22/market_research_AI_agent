from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


def test_db_connects(engine: Engine):
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_pgvector_extension_enabled(engine: Engine):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).first()
    assert row is not None


def test_tables_exist(engine: Engine):
    tables = set(inspect(engine).get_table_names())
    assert {"prices", "fundamentals", "documents"} <= tables


def test_documents_table_has_vector_column(db_session: Session):
    row = db_session.execute(
        text(
            "SELECT data_type, udt_name FROM information_schema.columns "
            "WHERE table_name = 'documents' AND column_name = 'embedding'"
        )
    ).first()
    assert row is not None
    assert row.udt_name == "vector"


def test_bi_views_exist_and_compute_sensible_values(db_session):
    import datetime as dt

    from sqlalchemy import text

    from market_research_agent.data.models import Fundamental, Price

    for i, close in enumerate([100.0, 110.0, 99.0]):
        db_session.add(
            Price(
                ticker="ZZZV",
                date=dt.date(2026, 1, 1 + i),
                open=1,
                high=1,
                low=1,
                close=close,
                adj_close=close,
                volume=5,
            )
        )
    for metric, value in [("Total Revenue", 10.0), ("Net Income", 2.0)]:
        db_session.add(
            Fundamental(ticker="ZZZV", period=dt.date(2025, 12, 31), metric=metric, value=value)
        )
    db_session.flush()

    rows = db_session.execute(
        text(
            "SELECT rebased, daily_return FROM v_price_performance "
            "WHERE ticker = 'ZZZV' ORDER BY date"
        )
    ).all()
    assert [round(r.rebased, 1) for r in rows] == [100.0, 110.0, 99.0]
    assert rows[0].daily_return is None and round(rows[1].daily_return, 2) == 0.10
    key = db_session.execute(
        text(
            "SELECT total_revenue, net_income, free_cash_flow FROM v_fundamentals_key "
            "WHERE ticker = 'ZZZV'"
        )
    ).one()
    assert (key.total_revenue, key.net_income, key.free_cash_flow) == (10.0, 2.0, None)
    # the accuracy view is queryable even with no backtest rows for this ticker
    assert (
        db_session.execute(
            text("SELECT count(*) FROM v_model_accuracy WHERE ticker = 'ZZZV'")
        ).scalar_one()
        == 0
    )

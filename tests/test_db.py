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

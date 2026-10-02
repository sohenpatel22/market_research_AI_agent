from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from market_research_agent.config import settings

# Hybrid retrieval support: a generated full-text column + GIN index for keyword search, and an
# HNSW index for approximate nearest-neighbour search on the cosine distance of embeddings.
SEARCH_DDL = [
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS section VARCHAR(64)",
    """ALTER TABLE documents ADD COLUMN IF NOT EXISTS fts tsvector
       GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED""",
    "CREATE INDEX IF NOT EXISTS ix_documents_fts ON documents USING gin (fts)",
    """CREATE INDEX IF NOT EXISTS ix_documents_embedding_hnsw
       ON documents USING hnsw (embedding vector_cosine_ops)""",
]


class Base(DeclarativeBase):
    pass


def engine_options(url: str) -> dict:
    """Connection settings that suit both a local Postgres and a serverless one (e.g. Neon).

    * `connect_timeout`: a scaled-to-zero database can take several seconds to wake up.
    * `pool_pre_ping` + `pool_recycle`: serverless Postgres closes idle connections.
    * `prepare_threshold=None` behind a transaction-mode pooler (Neon's `-pooler` hosts, PgBouncer):
      server-side prepared statements do not survive across pooled connections.
    """
    connect_args: dict = {"connect_timeout": 30}
    if "-pooler" in url or "pgbouncer" in url:
        connect_args["prepare_threshold"] = None
    return {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 5,
        "max_overflow": 5,
        "connect_args": connect_args,
    }


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(settings.database_url, **engine_options(settings.database_url))


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session that commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> None:
    """Enable pgvector and create all tables. Safe to call repeatedly."""
    engine = engine or get_engine()
    # Import here so Base.metadata is fully populated before create_all.
    from market_research_agent.data import models  # noqa: F401

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(engine)

    # Idempotent upgrades and search indexes that create_all() can't express.
    with engine.connect() as conn:
        for ddl in SEARCH_DDL:
            conn.execute(text(ddl))
        conn.commit()

    from market_research_agent.data.views import BI_VIEWS

    with engine.connect() as conn:
        for ddl in BI_VIEWS:
            conn.execute(text(ddl))
        conn.commit()

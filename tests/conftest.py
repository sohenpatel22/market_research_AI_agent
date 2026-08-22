import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from market_copilot.data.db import get_engine, init_db


@pytest.fixture(scope="session")
def engine() -> Engine:
    """Real Postgres engine (requires `docker compose up -d`), with schema created once."""
    eng = get_engine()
    init_db(eng)
    return eng


@pytest.fixture
def db_session(engine: Engine) -> Session:
    """A session bound to a connection/transaction that's rolled back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()

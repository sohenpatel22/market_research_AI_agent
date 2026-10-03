import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from market_research_agent.data.db import get_engine, init_db


@pytest.fixture(scope="session")
def engine() -> Engine:
    eng = get_engine()
    init_db(eng)
    return eng


@pytest.fixture
def db_session(engine: Engine) -> Session:
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()

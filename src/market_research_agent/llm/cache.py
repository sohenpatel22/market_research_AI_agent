"""Tiny persistent LLM response cache (SQLite)"""

import sqlite3
from functools import lru_cache
from pathlib import Path

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads


class SQLiteLLMCache(BaseCache):
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache "
            "(prompt TEXT, llm TEXT, response TEXT, PRIMARY KEY (prompt, llm))"
        )
        self._conn.commit()

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        row = self._conn.execute(
            "SELECT response FROM llm_cache WHERE prompt = ? AND llm = ?", (prompt, llm_string)
        ).fetchone()
        return loads(row[0], allowed_objects="core") if row else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)",
            (prompt, llm_string, dumps(list(return_val))),
        )
        self._conn.commit()

    def clear(self, **kwargs) -> None:
        self._conn.execute("DELETE FROM llm_cache")
        self._conn.commit()


@lru_cache(maxsize=4)
def get_llm_cache(path: str) -> SQLiteLLMCache:
    return SQLiteLLMCache(path)

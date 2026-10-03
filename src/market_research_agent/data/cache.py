"""Disk cache for raw, unprocessed downloads (prices, fundamentals, filings)"""

from pathlib import Path

RAW_DATA_DIR = Path("data/raw")


def raw_path(*parts: str) -> Path:
    """Return (and ensure the parent directory exists for) a path under data/raw/"""
    path = RAW_DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_cached(*parts: str) -> str | None:
    path = raw_path(*parts)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def write_cache(content: str, *parts: str) -> Path:
    path = raw_path(*parts)
    path.write_text(content, encoding="utf-8")
    return path

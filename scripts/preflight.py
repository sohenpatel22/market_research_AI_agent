"""Deployment preflight: check the production database and the Hugging Face token/Space.

Reads DATABASE_URL and HF_TOKEN from the environment (never from arguments) and prints only
non-sensitive facts: no hostnames, usernames, passwords or tokens. It is meant to run in CI
(public logs) as well as locally.

Usage:
    python scripts/preflight.py [--space owner/name] [--require-data]

Exit status is non-zero if a required check fails.
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

from market_research_agent.config import Settings
from market_research_agent.data.db import engine_options

OK, FAIL, WARN = "OK  ", "FAIL", "WARN"


def report(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def check_database(require_data: bool) -> bool:
    raw = os.environ.get("DATABASE_URL")
    if not raw:
        report(FAIL, "DATABASE_URL is not set")
        return False
    url = Settings(database_url=raw, _env_file=None).database_url  # normalises the driver prefix
    ok = True
    try:
        engine = create_engine(url, **engine_options(url))
        with engine.connect() as conn:
            version = conn.execute(text("SHOW server_version")).scalar_one()
            ssl = conn.execute(text("SHOW ssl")).scalar_one()
            report(OK, f"connected: PostgreSQL {version}, ssl={ssl}")
            pooled = "-pooler" in url
            report(OK, f"pooled endpoint: {'yes' if pooled else 'no (direct)'}")

            available = conn.execute(
                text("SELECT default_version FROM pg_available_extensions WHERE name = 'vector'")
            ).scalar()
            if available is None:
                report(FAIL, "the pgvector extension is not available on this server")
                return False
            installed = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            if installed is None:
                report(WARN, f"pgvector {available} is available but not enabled yet")
            else:
                report(OK, f"pgvector {installed} is enabled")

            tables = {
                r[0]
                for r in conn.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                )
            }
            for name in ("documents", "prices", "fundamentals"):
                if name in tables:
                    n = conn.execute(text(f"SELECT count(*) FROM {name}")).scalar_one()
                    report(OK if n else WARN, f"table {name}: {n} rows")
                    ok = ok and (n > 0 or not require_data)
                else:
                    report(WARN if not require_data else FAIL, f"table {name} does not exist yet")
                    ok = ok and not require_data
    except Exception as exc:  # noqa: BLE001
        # The class name is enough to diagnose; the message may contain connection details.
        report(FAIL, f"could not use the database ({type(exc).__name__})")
        return False
    return ok


def check_huggingface(space: str | None) -> bool:
    token = os.environ.get("HF_TOKEN")
    if not token:
        report(FAIL, "HF_TOKEN is not set")
        return False
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        who = api.whoami()
        name = who.get("name")
        role = (who.get("auth", {}).get("accessToken", {}) or {}).get("role", "unknown")
        report(OK, f"token is valid for Hugging Face user '{name}' (token role: {role})")
        if role == "read":
            report(FAIL, "the token is read-only; deploying needs a write token")
            return False

        spaces = list(api.list_spaces(author=name))
        if not spaces:
            report(WARN, f"user '{name}' owns no Spaces yet")
        for s in spaces:
            report(OK, f"Space found: {s.id} (sdk={s.sdk}, private={s.private})")
        if space:
            info = api.space_info(space)
            report(OK, f"target Space {info.id}: sdk={info.sdk}, stage={info.runtime.stage}")
            if info.sdk != "docker":
                report(FAIL, f"the Space SDK is '{info.sdk}', it must be 'docker'")
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        report(FAIL, f"Hugging Face check failed ({type(exc).__name__})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", help="expected Space id, e.g. owner/market-research-agent")
    parser.add_argument("--require-data", action="store_true", help="fail if tables are empty")
    args = parser.parse_args()

    print("== Database ==")
    db_ok = check_database(args.require_data)
    print("== Hugging Face ==")
    hf_ok = check_huggingface(args.space)
    print("\nPreflight:", "PASSED" if db_ok and hf_ok else "FAILED")
    return 0 if db_ok and hf_ok else 1


if __name__ == "__main__":
    sys.exit(main())

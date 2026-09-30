"""Langfuse tracing, scores and prompt registry.

Everything here is a no-op unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, so tests,
CI and forks run without Langfuse. Import errors or network failures never break a request:
observability must not take the app down.
"""

import atexit
import logging
import os
import re
import subprocess
from contextlib import nullcontext, suppress
from functools import lru_cache

from market_research_agent.config import Settings, settings

logger = logging.getLogger(__name__)


def is_enabled(cfg: Settings = settings) -> bool:
    return bool(cfg.langfuse_public_key and cfg.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_client():
    """The process-wide Langfuse client, or None when tracing is off."""
    if not is_enabled():
        return None
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            environment=settings.langfuse_environment,
        )
        atexit.register(shutdown)
        return client
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse init failed; tracing disabled", exc_info=True)
        return None


def get_handler():
    """A fresh LangChain/LangGraph callback handler (one per request), or None."""
    if get_client() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler(public_key=settings.langfuse_public_key)
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse callback handler unavailable", exc_info=True)
        return None


def trace_context(**attrs):
    """Attach session/user/tags/metadata to every span created inside the block."""
    if get_client() is None:
        return nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(**attrs)


def post_scores(trace_id: str | None, scores: dict[str, float | bool | None]) -> str | None:
    """Attach evaluation scores to a trace; returns the trace URL when available."""
    client = get_client()
    if client is None or not trace_id:
        return None
    for name, value in scores.items():
        if value is None:
            continue
        try:
            client.create_score(
                trace_id=trace_id,
                name=name,
                value=float(value),
                data_type="BOOLEAN" if isinstance(value, bool) else "NUMERIC",
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to post score %s", name, exc_info=True)
    try:
        return client.get_trace_url(trace_id=trace_id)
    except Exception:  # noqa: BLE001
        return None


def flush() -> None:
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            logger.warning("Langfuse flush failed", exc_info=True)


def shutdown() -> None:
    client = get_client()
    if client is not None:
        with suppress(Exception):
            client.shutdown()


# ------------------------------------------------------------------------- prompt registry
_MUSTACHE = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_SINGLE = re.compile(r"\{(\w+)\}")


def to_langfuse_template(local: str) -> str:
    """Local templates use {var}; Langfuse uses {{var}}."""
    return _SINGLE.sub(r"{{\1}}", local)


def from_langfuse_template(remote: str) -> str:
    return _MUSTACHE.sub(r"{\1}", remote)


def load_prompt(name: str, local: str) -> str:
    """The prompt text to use: the registry's `production` version if enabled, else `local`."""
    if not settings.langfuse_prompts:
        return local
    client = get_client()
    if client is None:
        return local
    try:
        prompt = client.get_prompt(name, type="text", label="production", cache_ttl_seconds=300)
        text = from_langfuse_template(prompt.prompt)
        # Guard against a registry edit that drops or renames variables the code fills in.
        if set(_SINGLE.findall(text)) != set(_SINGLE.findall(local)):
            logger.warning("Registry prompt %r has different variables; using local", name)
            return local
        return text
    except Exception:  # noqa: BLE001
        return local


def sync_prompts(prompts: dict[str, str]) -> None:
    """Create/update registry prompts from the local templates (labelled production)."""
    client = get_client()
    if client is None:
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY first.")
    for name, text in prompts.items():
        client.create_prompt(
            name=name, prompt=to_langfuse_template(text), labels=["production"], type="text"
        )
        print(f"synced prompt: {name}")


@lru_cache(maxsize=1)
def git_sha() -> str:
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"][:12]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"

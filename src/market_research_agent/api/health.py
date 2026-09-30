"""Health checks for the database, trained models, LLM configuration and observability."""

from importlib.metadata import PackageNotFoundError, version

from sqlalchemy import text

from market_research_agent import observability
from market_research_agent.api.schemas import ComponentStatus, HealthResponse
from market_research_agent.config import settings
from market_research_agent.data.db import get_engine
from market_research_agent.models import registry
from market_research_agent.observability.langfuse import git_sha


def app_version() -> str:
    try:
        return version("market-research-agent")
    except PackageNotFoundError:
        return "0.0.0"


def _database() -> ComponentStatus:
    try:
        with get_engine().connect() as conn:
            docs = conn.execute(text("SELECT count(*) FROM documents")).scalar_one()
            prices = conn.execute(text("SELECT count(*) FROM prices")).scalar_one()
        return ComponentStatus(ok=True, detail=f"{docs} filing chunks, {prices} price rows")
    except Exception as exc:  # noqa: BLE001
        return ComponentStatus(ok=False, detail=f"database unreachable ({type(exc).__name__})")


def _models() -> ComponentStatus:
    try:
        bundle = registry.load_bundle()
        return ComponentStatus(ok=True, detail=f"bundle {bundle['meta']['version']}")
    except Exception as exc:  # noqa: BLE001
        return ComponentStatus(ok=False, detail=f"no trained model bundle ({type(exc).__name__})")


def _llm() -> ComponentStatus:
    key = getattr(settings, f"{settings.llm_provider}_api_key", None)
    if key:
        return ComponentStatus(ok=True, detail=f"{settings.llm_provider} key configured")
    return ComponentStatus(ok=False, detail=f"{settings.llm_provider.upper()}_API_KEY not set")


def check_health() -> HealthResponse:
    components = {
        "database": _database(),
        "forecast_models": _models(),
        "llm": _llm(),
        # Optional: tracing being off is not a failure.
        "langfuse": ComponentStatus(
            ok=True, detail="enabled" if observability.is_enabled() else "disabled"
        ),
    }
    required = ("database", "forecast_models", "llm")
    status = "ok" if all(components[c].ok for c in required) else "degraded"
    return HealthResponse(
        status=status,
        version=app_version(),
        git_sha=git_sha(),
        llm_provider=settings.llm_provider,
        components=components,
    )

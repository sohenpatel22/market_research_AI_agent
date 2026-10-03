import pytest

from market_research_agent.config import Settings
from market_research_agent.llm.factory import (
    DEFAULT_MODELS,
    MissingAPIKeyError,
    get_chat_model,
    get_judge_model,
)


def make_settings(**kw) -> Settings:
    base = dict(database_url="postgresql://x", _env_file=None)
    return Settings(**{**base, **kw})


@pytest.mark.parametrize("provider", ["deepseek", "openai", "anthropic"])
def test_builds_each_provider(provider):
    cfg = make_settings(llm_provider=provider, **{f"{provider}_api_key": "test-key"})
    llm = get_chat_model(cfg=cfg)
    assert DEFAULT_MODELS[provider] in {
        getattr(llm, "model_name", None),
        getattr(llm, "model", None),
    }


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        get_chat_model(cfg=make_settings(llm_provider="openai"))


def test_judge_falls_back_and_overrides():
    cfg = make_settings(deepseek_api_key="k", anthropic_api_key="k")
    assert type(get_judge_model(cfg=cfg)).__name__ == "ChatDeepSeek"
    cfg = make_settings(deepseek_api_key="k", anthropic_api_key="k", judge_provider="anthropic")
    assert type(get_judge_model(cfg=cfg)).__name__ == "ChatAnthropic"


def test_database_url_is_normalised_to_the_psycopg3_driver():
    def url(raw):
        return make_settings(database_url=raw).database_url

    assert url("postgresql://u:p@h/db?sslmode=require") == (
        "postgresql+psycopg://u:p@h/db?sslmode=require"
    )
    assert url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert url('  "postgresql://u:p@h/db"  ') == "postgresql+psycopg://u:p@h/db"
    explicit = "postgresql+psycopg://u:p@h:5433/db"
    assert url(explicit) == explicit  # already explicit: untouched


def test_engine_options_for_local_and_pooled_hosts():
    from market_research_agent.data.db import engine_options

    local = engine_options("postgresql+psycopg://u:p@localhost:5433/db")
    assert local["connect_args"] == {"connect_timeout": 30} and local["pool_pre_ping"]
    pooled = engine_options("postgresql+psycopg://u:p@ep-cool-123-pooler.neon.tech/db")
    assert pooled["connect_args"]["prepare_threshold"] is None


def test_claude_5_models_are_built_without_an_explicit_temperature(monkeypatch):
    import langchain_anthropic

    calls = []

    class Recorder:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", Recorder)
    cfg = make_settings(anthropic_api_key="k")
    for model in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"):
        get_chat_model("anthropic", model, cfg=cfg)
    get_chat_model("anthropic", "claude-haiku-4-5-20251001", cfg=cfg)
    assert all("temperature" not in c for c in calls[:3])
    assert calls[3]["temperature"] == 0.0


def test_secrets_pasted_with_quotes_or_left_empty_are_cleaned():
    cfg = make_settings(
        langfuse_public_key='"pk-lf-1"',
        langfuse_secret_key="'sk-lf-2'",
        LANGFUSE_BASE_URL='"https://us.cloud.langfuse.com"',
        openai_api_key="  ",
        deepseek_api_key="plain",
    )
    assert cfg.langfuse_public_key == "pk-lf-1"
    assert cfg.langfuse_secret_key == "sk-lf-2"
    assert cfg.langfuse_host == "https://us.cloud.langfuse.com"
    assert cfg.openai_api_key is None
    assert cfg.deepseek_api_key == "plain"


def test_blank_langfuse_host_falls_back_to_the_default():
    assert make_settings(LANGFUSE_HOST="").langfuse_host == "https://cloud.langfuse.com"

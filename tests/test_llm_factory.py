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


def test_anthropic_models_that_reject_temperature_fall_back_to_the_default(monkeypatch):
    import langchain_anthropic

    calls = []

    class Strict:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise ValueError("`temperature` is not supported for this model")

    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", Strict)
    get_chat_model("anthropic", "claude-sonnet-5-5", cfg=make_settings(anthropic_api_key="k"))
    assert "temperature" in calls[0] and "temperature" not in calls[1]

    class Broken:
        def __init__(self, **kwargs):
            raise ValueError("something else entirely")

    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", Broken)
    with pytest.raises(ValueError, match="something else"):
        get_chat_model("anthropic", cfg=make_settings(anthropic_api_key="k"))

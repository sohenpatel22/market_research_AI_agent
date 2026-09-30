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

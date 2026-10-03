"""Provider-agnostic chat model factory (DeepSeek, OpenAI, Anthropic)"""

import re

from langchain_core.language_models.chat_models import BaseChatModel

from market_research_agent.config import Settings, settings

# Defaults live here (not scattered in code) because provider model names change often.
DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}

# DeepSeek has no strict json_schema mode, so it uses function calling.
STRUCTURED_OUTPUT_METHOD = {
    "deepseek": "function_calling",
    "openai": "json_schema",
    "anthropic": "function_calling",
}


# Claude 5.x models reject an explicit temperature.
_REJECTS_TEMPERATURE = re.compile(r"^claude-(sonnet|opus)-5|^claude-(fable|mythos)")


class MissingAPIKeyError(RuntimeError):
    pass


def _build(
    provider: str, model: str | None, temperature: float, cfg: Settings, **kwargs
) -> BaseChatModel:
    model = model or DEFAULT_MODELS[provider]
    key = getattr(cfg, f"{provider}_api_key")
    if not key:
        raise MissingAPIKeyError(
            f"{provider.upper()}_API_KEY is not set (LLM provider is '{provider}')."
        )

    if cfg.llm_cache:
        from market_research_agent.llm.cache import get_llm_cache

        kwargs.setdefault("cache", get_llm_cache(cfg.llm_cache_path))

    # Imports are lazy so unused provider SDKs are never touched.
    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(model=model, api_key=key, temperature=temperature, **kwargs)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=key, temperature=temperature, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if _REJECTS_TEMPERATURE.match(model):
            return ChatAnthropic(model=model, api_key=key, **kwargs)
        return ChatAnthropic(model=model, api_key=key, temperature=temperature, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    cfg: Settings = settings,
    **kwargs,
) -> BaseChatModel:
    """Chat model for the configured (or overridden) provider"""
    provider = provider or cfg.llm_provider
    if model is None and provider == cfg.llm_provider:
        model = cfg.llm_model
    temp = cfg.llm_temperature if temperature is None else temperature
    return _build(provider, model, temp, cfg, **kwargs)


def get_judge_model(cfg: Settings = settings, **kwargs) -> BaseChatModel:
    """Chat model for LLM-as-judge; falls back to the main provider if unset"""
    provider = cfg.judge_provider or cfg.llm_provider
    model = cfg.judge_model if cfg.judge_provider else cfg.llm_model
    return _build(provider, model, 0.0, cfg, **kwargs)


def structured_output(llm: BaseChatModel, schema: type, provider: str | None = None):
    """Bind a Pydantic schema using the method that works for the provider"""
    provider = provider or settings.llm_provider
    return llm.with_structured_output(schema, method=STRUCTURED_OUTPUT_METHOD[provider])

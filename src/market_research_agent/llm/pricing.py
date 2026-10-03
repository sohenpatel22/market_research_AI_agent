"""Approximate USD prices per 1M tokens, used to report eval cost. Update when providers change.

DeepSeek: https://api-docs.deepseek.com/quick_start/pricing (peak rate; off-peak is half).
Anthropic: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-02).
OpenAI: list prices from third-party summaries (openai.com blocks automated fetches); verify
before quoting them anywhere.
"""

# model name -> (input, cached input, output) USD per 1M tokens
PRICES: dict[str, tuple[float, float, float]] = {
    "deepseek-flash": (0.30, 0.006, 1.20),
    "deepseek-chat": (0.30, 0.006, 1.20),
    "deepseek-v4-pro": (1.32, 0.044, 3.96),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "claude-haiku-4-5-20251001": (1.00, 0.10, 5.00),
    "claude-sonnet-5-5": (2.00, 0.20, 10.00),
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "claude-fable-5-1": (10.00, 0.25, 50.00),
}

# Claude 4.7+ models use a tokenizer that yields about 30% more tokens for the same text
# (Anthropic's pricing page), so the same prompt costs correspondingly more on them.


def estimate_cost(usage_by_model: dict[str, dict]) -> float:
    """Cost of a `UsageMetadataCallbackHandler.usage_metadata` dict. Unknown models cost 0."""
    total = 0.0
    for model, u in usage_by_model.items():
        price = PRICES.get(model)
        if price is None:
            continue
        cached = (u.get("input_token_details") or {}).get("cache_read", 0)
        fresh = max(u.get("input_tokens", 0) - cached, 0)
        total += (fresh * price[0] + cached * price[1] + u.get("output_tokens", 0) * price[2]) / 1e6
    return total


def total_tokens(usage_by_model: dict[str, dict]) -> int:
    return sum(u.get("total_tokens", 0) for u in usage_by_model.values())

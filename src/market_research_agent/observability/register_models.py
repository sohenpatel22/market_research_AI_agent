"""Register per-token prices for models Langfuse doesn't price out of the box.

Langfuse computes cost from token usage only when it knows the model's price. DeepSeek's current
models are not in its built-in list, so traces show tokens but no cost until they're added.
Prices are USD per 1M tokens at DeepSeek's *peak* rate (conservative; off-peak is half), taken
from https://api-docs.deepseek.com/quick_start/pricing. Update when pricing changes.

Usage:
    uv run python -m market_research_agent.observability.register_models
"""

from market_research_agent.observability import get_client

PER_MILLION = {
    # model_name: (regex match pattern, input $/1M, output $/1M)
    "deepseek-flash": (r"(?i)^deepseek-(v4-)?flash(-vision-exp)?$", 0.30, 1.20),
    "deepseek-v4-pro": (r"(?i)^deepseek-v4-pro$", 1.32, 3.96),
    "deepseek-chat": (r"(?i)^deepseek-chat$", 0.30, 1.20),
}


def main() -> None:
    client = get_client()
    if client is None:
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY first.")
    for name, (pattern, inp, out) in PER_MILLION.items():
        try:
            client.api.models.create(
                model_name=name,
                match_pattern=pattern,
                unit="TOKENS",
                input_price=inp / 1e6,
                output_price=out / 1e6,
            )
            print(f"registered {name}: ${inp}/1M in, ${out}/1M out")
        except Exception as e:  # noqa: BLE001 - already exists or API error; keep going
            print(f"skipped {name}: {str(e)[:120]}")


if __name__ == "__main__":
    main()

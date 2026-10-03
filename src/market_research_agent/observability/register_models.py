"""Register per-token prices for models Langfuse doesn't price out of the box"""

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
        except Exception as e:  # noqa: BLE001
            print(f"skipped {name}: {str(e)[:120]}")


if __name__ == "__main__":
    main()

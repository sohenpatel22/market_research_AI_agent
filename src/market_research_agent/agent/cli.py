"""Ask the agent a question from the command line.

Usage:
    uv run python -m market_research_agent.agent.cli "What are Apple's main supply chain risks?"
    uv run python -m market_research_agent.agent.cli --sync-prompts   # push prompts to Langfuse
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--sync-prompts",
        action="store_true",
        help="create/update the prompt registry entries in Langfuse from the local templates",
    )
    args = parser.parse_args()

    if args.sync_prompts:
        from market_research_agent.agent.prompts import PROMPTS
        from market_research_agent.observability.langfuse import sync_prompts

        sync_prompts(PROMPTS)
        return
    if not args.question:
        parser.error("a question is required")

    from market_research_agent.agent.service import ask

    result = ask(args.question, flush_now=True)
    print(result.answer.model_dump_json(indent=2))
    if result.trace_url:
        print(f"\nLangfuse trace: {result.trace_url}")


if __name__ == "__main__":
    main()

"""Ask the agent a question from the command line.

Usage:
    uv run python -m market_research_agent.agent.cli "What are Apple's main supply chain risks?"
"""

import argparse

from market_research_agent.agent.graph import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()
    print(run_agent(args.question).model_dump_json(indent=2))


if __name__ == "__main__":
    main()

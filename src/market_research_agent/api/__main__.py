"""`python -m market_research_agent.api` starts the server (PORT defaults to 7860 like HF Spaces)."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "market_research_agent.api.main:create_app",
        factory=True,
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104 - a server must listen on all ifaces
        port=int(os.environ.get("PORT", "7860")),
        log_level="info",
    )


if __name__ == "__main__":
    main()

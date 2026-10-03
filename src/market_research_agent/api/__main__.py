"""`python -m market_research_agent.api` starts the server (default port 7860"""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "market_research_agent.api.main:create_app",
        factory=True,
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104
        port=int(os.environ.get("PORT", "7860")),
        log_level="info",
    )


if __name__ == "__main__":
    main()

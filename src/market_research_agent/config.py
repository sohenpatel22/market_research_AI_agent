from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider for the agent; keys are only needed for the provider actually used.
    llm_provider: Literal["deepseek", "openai", "anthropic"] = "deepseek"
    llm_model: str | None = None  # None -> provider default (see llm/factory.py)
    llm_temperature: float = 0.0
    # Separate provider for graders/evals so the judge isn't the model under test.
    judge_provider: Literal["deepseek", "openai", "anthropic"] | None = None
    judge_model: str | None = None
    deepseek_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    database_url: str

    sec_edgar_user_agent: str = "Market Research Agent you@example.com"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"


settings = Settings()

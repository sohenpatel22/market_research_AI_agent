from typing import Literal

from pydantic import AliasChoices, Field, field_validator
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

    # Persistent SQLite cache of identical LLM calls (saves money when re-running evals/dev).
    llm_cache: bool = False
    llm_cache_path: str = ".cache/llm_cache.sqlite"

    # Langfuse observability; tracing is off unless both keys are set.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = Field(
        "https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
    )
    langfuse_environment: str = "development"
    langfuse_prompts: bool = False  # fetch prompts from the Langfuse registry (local fallback)

    database_url: str

    sec_edgar_user_agent: str = "Market Research Agent you@example.com"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    use_reranker: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    agent_max_retries: int = 2
    agent_quality_threshold: float = 0.7
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    @field_validator("langfuse_host", mode="before")
    @classmethod
    def _blank_host_means_default(cls, v):
        # CI passes unset secrets as empty strings.
        return v or "https://cloud.langfuse.com"


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Required once the agent (Phase 3+) calls out to an LLM; not needed for the data layer.
    llm_api_key: str | None = None
    database_url: str

    sec_edgar_user_agent: str = "Market Research Agent you@example.com"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"


settings = Settings()

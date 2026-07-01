from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    catalog_path: str = "shl_product_catalog.json"

    request_timeout_seconds: float = 25.0
    max_recommendations: int = 10
    max_completion_retries: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()

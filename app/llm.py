from openai import AsyncOpenAI

from app.config import Settings


def build_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        timeout=settings.request_timeout_seconds,
    )

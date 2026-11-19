"""Application settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_OPENAI_MODEL = "gpt-6-luna"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str


def get_settings() -> Settings:
    """Load settings. Real environment variables take precedence over .env values."""
    load_dotenv()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
    )

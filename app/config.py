"""Application settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_PROVIDER = "deepseek"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_JOTFORM_BASE_URL = "https://api.jotform.com"


@dataclass(frozen=True)
class Settings:
    llm_provider: str = DEFAULT_PROVIDER
    deepseek_api_key: str | None = None
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL
    # "disabled" = faster and cheaper; "enabled" = the model reasons before answering.
    deepseek_thinking: str = "disabled"
    openai_api_key: str | None = None
    openai_model: str = DEFAULT_OPENAI_MODEL
    jotform_api_key: str | None = None
    jotform_form_id: str | None = None
    jotform_base_url: str = DEFAULT_JOTFORM_BASE_URL

    @property
    def jotform_configured(self) -> bool:
        return bool(self.jotform_api_key and self.jotform_form_id)


def get_settings() -> Settings:
    """Load settings. Real environment variables take precedence over .env values."""
    load_dotenv()
    return Settings(
        llm_provider=(os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower(),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
        deepseek_model=os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL,
        deepseek_thinking=(os.getenv("DEEPSEEK_THINKING") or "disabled").lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
        jotform_api_key=os.getenv("JOTFORM_API_KEY") or None,
        jotform_form_id=os.getenv("JOTFORM_FORM_ID") or None,
        jotform_base_url=os.getenv("JOTFORM_BASE_URL") or DEFAULT_JOTFORM_BASE_URL,
    )

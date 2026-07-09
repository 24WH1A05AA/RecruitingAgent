"""
tools/config.py
---------------
Shared configuration and LLM client factory for all TechVest tools.

Provides
--------
Settings        - pydantic-settings model loaded from environment / .env
get_llm()       - returns a ChatOpenAI client pointed at OpenRouter
PARSE_MODEL     - default model string for resume parsing (free tier)
SCORE_MODEL     - default model string for candidate scoring (free tier)
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# OpenRouter free-tier model identifiers
# ---------------------------------------------------------------------------

PARSE_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"
SCORE_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Settings (loaded from .env / environment variables)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings resolved from environment variables or a .env file.

    All values can be overridden at runtime via environment variables with the
    same name (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenRouter (required for LLM-backed tools)
    openrouter_api_key: str = ""

    # Fallback: standard OpenAI key (used if openrouter_api_key is empty)
    openai_api_key: str = ""

    # LangSmith tracing (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "techvest-recruitment-agent"

    # Agent behaviour
    app_env: str = "development"
    log_level: str = "INFO"
    max_agent_iterations: int = 10

    # LLM generation parameters
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout: int = 60  # seconds


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def get_llm(
    model: str = PARSE_MODEL,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Return a ChatOpenAI client configured for OpenRouter.

    Parameters
    ----------
    model:
        OpenRouter model identifier string.
        Defaults to ``PARSE_MODEL`` (Llama 3.1 8B free tier).
    temperature:
        Sampling temperature; defaults to ``Settings.llm_temperature`` (0.0
        for deterministic, reproducible scoring).
    max_tokens:
        Maximum tokens in the completion; defaults to ``Settings.llm_max_tokens``.

    Returns
    -------
    ChatOpenAI
        A LangChain ChatOpenAI instance pointed at ``https://openrouter.ai/api/v1``.

    Raises
    ------
    ValueError
        If neither ``OPENROUTER_API_KEY`` nor ``OPENAI_API_KEY`` is set.

    Notes
    -----
    The function is intentionally NOT cached so callers can request different
    models without conflict.  The underlying HTTP session is reused by the
    OpenAI SDK.
    """
    cfg = get_settings()

    api_key = cfg.openrouter_api_key or cfg.openai_api_key
    if not api_key:
        raise ValueError(
            "No API key found. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in your .env file."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,                     # type: ignore[arg-type]
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature if temperature is not None else cfg.llm_temperature,
        max_tokens=max_tokens if max_tokens is not None else cfg.llm_max_tokens,
        timeout=cfg.llm_timeout,
        max_retries=2,
    )

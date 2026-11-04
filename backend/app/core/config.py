"""Centralized Application Configuration for Argus.

Enforces Part A Zero-Hardcoding standard:
- Validated environment-driven settings with safe, documented defaults
- Centralized model parameters, timeouts, and heuristics
- Strict isolation of environment configs
"""

import json
from pathlib import Path
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and defaults."""

    model_config = SettingsConfigDict(
        env_file=(".env", str(BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "Argus"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Local Ollama Inference Settings (Phase 3 & 3.1)
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:1.5b"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0
    OLLAMA_STATUS_TIMEOUT_SECONDS: float = 4.0

    # Ollama KV-cache and request-budget controls.
    # These values describe Argus policy; Flash Attention and KV quantization
    # must also be configured in the Ollama server process itself.
    OLLAMA_KV_CACHE_TYPE: str = "q8_0"
    OLLAMA_FLASH_ATTENTION: bool = True
    OLLAMA_NUM_CTX: int = 4096
    OLLAMA_MAX_NUM_CTX: int = 8192
    OLLAMA_NUM_PREDICT: int = 512
    OLLAMA_CONTEXT_RESERVE_TOKENS: int = 256
    OLLAMA_MAX_CONCURRENT_REQUESTS: int = 1

    # Model Generation Hyperparameters
    OLLAMA_TEMPERATURE: float = 0.2
    OLLAMA_TOP_P: float = 0.9

    # Output Budget Control Settings (Phase 4.1)
    OUTPUT_BUDGET_ENABLED: bool = True
    OUTPUT_BUDGET_POLICY: str = "adaptive"  # "adaptive", "fixed", or "conservative"
    OUTPUT_BUDGET_MIN_TOKENS: int = 64
    OUTPUT_BUDGET_MAX_TOKENS: int = 2048
    OUTPUT_BUDGET_DEFAULT_TOKENS: int = 512
    OUTPUT_BUDGET_SAFETY_HEADROOM: int = 64

    # Optimization & Token Estimation Heuristics
    CHARS_PER_TOKEN_HEURISTIC: float = 4.0
    MAX_VALIDATION_RETRY_ATTEMPTS: int = 1

    # Main LLM Provider Settings (Phase 4 - Google Gemini as Default)
    MAIN_LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_API_BASE: str = "https://generativelanguage.googleapis.com/v1beta"
    MAIN_LLM_MODEL: str = "gemini-1.5-flash"
    MAIN_LLM_TIMEOUT_SECONDS: float = 60.0
    MAIN_LLM_TEMPERATURE: float = 0.2
    MAIN_LLM_MAX_TOKENS: int = 1500

    # Optional / Alternative Providers (OpenAI)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: str = "https://api.openai.com/v1"

    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    @field_validator("OLLAMA_MAX_NUM_CTX")
    @classmethod
    def validate_max_num_ctx(cls, v: int, info) -> int:
        num_ctx = info.data.get("OLLAMA_NUM_CTX", 4096)
        if v < num_ctx:
            raise ValueError(
                f"OLLAMA_MAX_NUM_CTX ({v}) must be greater than or equal to OLLAMA_NUM_CTX ({num_ctx})"
            )
        return v

    @field_validator("OUTPUT_BUDGET_MAX_TOKENS")
    @classmethod
    def validate_output_budget_max(cls, v: int, info) -> int:
        min_budget = info.data.get("OUTPUT_BUDGET_MIN_TOKENS", 64)
        if v < min_budget:
            raise ValueError(
                f"OUTPUT_BUDGET_MAX_TOKENS ({v}) must be greater than or equal to OUTPUT_BUDGET_MIN_TOKENS ({min_budget})"
            )
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


settings = Settings()


def reload_settings() -> Settings:
    """Reload settings from environment and .env files."""
    global settings
    settings = Settings()
    return settings

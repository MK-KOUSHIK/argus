"""Centralized Application Configuration for Argus.

Enforces Part A Zero-Hardcoding standard:
- Validated environment-driven settings with safe, documented defaults
- Centralized model parameters, timeouts, and heuristics
- Strict isolation of environment configs
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import ValidationInfo, field_validator
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
    OUTPUT_BUDGET_TASK_BASELINES: Dict[str, int] = {
        "summarization": 256,
        "extraction": 256,
        "data_analysis": 384,
        "coding": 512,
        "planning": 512,
        "writing": 512,
        "research": 512,
        "general": 384,
    }
    OUTPUT_BUDGET_MODE_SCALING: Dict[str, float] = {
        "concise": 0.75,
        "balanced": 1.0,
        "structure": 1.2,
    }

    # Independent Content-Aware Compression Engine (Phase 4.2)
    COMPRESSION_ENABLED: bool = True
    COMPRESSION_REVERSIBLE: bool = True
    COMPRESSION_TTL_SECONDS: int = 3600
    COMPRESSION_REGISTRY_MAX_ENTRIES: int = 1000
    COMPRESSION_MIN_CHARS_THRESHOLD: int = 40
    COMPRESSION_STRATEGIES_ENABLED: Dict[str, bool] = {
        "json": True,
        "code": True,
        "logs": True,
        "text": True,
    }

    # Local-First Model Routing (Phase 5)
    ROUTING_ENABLED: bool = True
    ROUTING_POLICY: str = "auto"  # "auto", "prefer_small", "prefer_large"
    ROUTING_COMPLEXITY_THRESHOLD: float = 0.55
    ROUTING_DEFAULT_FALLBACK_MODEL: str = "qwen2.5:1.5b"
    ROUTING_VRAM_SAFETY_MARGIN_MB: int = 512
    ROUTING_RAM_SAFETY_MARGIN_MB: int = 1024
    ROUTING_LOADED_HEADROOM_MARGIN_MB: int = 250
    ROUTING_REGISTRY_TTL_SECONDS: float = 5.0
    ROUTING_ALLOW_UNKNOWN_TIER: bool = False
    ROUTING_ALLOW_OVERRIDE_FALLBACK: bool = False
    ROUTING_ALLOW_CPU_EXECUTION: bool = True
    ROUTING_REQUIRE_ARCHITECTURE_METADATA: bool = False
    ROUTING_CPU_ALLOWANCE_DEFAULT_MB: int = 128
    ROUTING_CPU_ALLOWANCE_SMALL_MB: int = 64
    ROUTING_CPU_ALLOWANCE_MEDIUM_MB: int = 128
    ROUTING_CPU_ALLOWANCE_LARGE_MB: int = 256

    def get_cpu_runtime_allowance_mb(self, tier: Any = "default") -> int:
        """Return configured CPU runtime memory allowance in MiB for a given model tier."""
        tier_val = getattr(tier, "value", tier)
        tier_str = str(tier_val).lower() if tier_val is not None else "default"
        if tier_str == "small":
            return self.ROUTING_CPU_ALLOWANCE_SMALL_MB
        elif tier_str == "medium":
            return self.ROUTING_CPU_ALLOWANCE_MEDIUM_MB
        elif tier_str == "large":
            return self.ROUTING_CPU_ALLOWANCE_LARGE_MB
        return self.ROUTING_CPU_ALLOWANCE_DEFAULT_MB

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

    @field_validator("OUTPUT_BUDGET_POLICY")
    @classmethod
    def validate_output_budget_policy(cls, v: str) -> str:
        normalized = v.strip().lower()
        allowed = {"adaptive", "fixed", "conservative"}
        if normalized not in allowed:
            raise ValueError(
                f"OUTPUT_BUDGET_POLICY must be one of {sorted(allowed)}; got '{v}'"
            )
        return normalized

    @field_validator("OUTPUT_BUDGET_MIN_TOKENS")
    @classmethod
    def validate_output_budget_min(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"OUTPUT_BUDGET_MIN_TOKENS must be at least 1; got {v}")
        return v

    @field_validator("OUTPUT_BUDGET_DEFAULT_TOKENS")
    @classmethod
    def validate_output_budget_default(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"OUTPUT_BUDGET_DEFAULT_TOKENS must be at least 1; got {v}")
        return v

    @field_validator("OUTPUT_BUDGET_SAFETY_HEADROOM")
    @classmethod
    def validate_output_budget_safety_headroom(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"OUTPUT_BUDGET_SAFETY_HEADROOM must be non-negative; got {v}")
        return v

    @field_validator("OUTPUT_BUDGET_TASK_BASELINES", mode="before")
    @classmethod
    def validate_output_budget_task_baselines(cls, v: Union[str, Dict[str, int]]) -> Dict[str, int]:
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception as exc:
                raise ValueError(f"OUTPUT_BUDGET_TASK_BASELINES must be valid JSON: {exc}") from exc
        if not isinstance(v, dict):
            raise ValueError("OUTPUT_BUDGET_TASK_BASELINES must be a dictionary")

        required_categories = {
            "coding", "data_analysis", "writing", "summarization",
            "research", "extraction", "planning", "general"
        }
        missing = required_categories - set(v.keys())
        if missing:
            raise ValueError(
                f"OUTPUT_BUDGET_TASK_BASELINES missing required categories: {sorted(missing)}"
            )

        validated: Dict[str, int] = {}
        for cat, tokens in v.items():
            if not isinstance(tokens, int) or isinstance(tokens, bool):
                try:
                    tokens = int(tokens)
                except (ValueError, TypeError):
                    raise ValueError(f"Baseline for task category '{cat}' must be an integer; got {tokens}")
            if tokens <= 0:
                raise ValueError(f"Baseline for task category '{cat}' must be positive; got {tokens}")
            validated[cat] = tokens
        return validated

    @field_validator("OUTPUT_BUDGET_MODE_SCALING", mode="before")
    @classmethod
    def validate_output_budget_mode_scaling(cls, v: Union[str, Dict[str, float]]) -> Dict[str, float]:
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception as exc:
                raise ValueError(f"OUTPUT_BUDGET_MODE_SCALING must be valid JSON: {exc}") from exc
        if not isinstance(v, dict):
            raise ValueError("OUTPUT_BUDGET_MODE_SCALING must be a dictionary")

        required_modes = {"concise", "balanced", "structure"}
        missing = required_modes - set(v.keys())
        if missing:
            raise ValueError(
                f"OUTPUT_BUDGET_MODE_SCALING missing required modes: {sorted(missing)}"
            )

        validated: Dict[str, float] = {}
        for mode, mult in v.items():
            if not isinstance(mult, (int, float)) or isinstance(mult, bool):
                try:
                    mult = float(mult)
                except (ValueError, TypeError):
                    raise ValueError(f"Scaling factor for mode '{mode}' must be a number; got {mult}")
            if mult <= 0.0 or mult > 5.0:
                raise ValueError(
                    f"Scaling factor for mode '{mode}' must be between 0.0 (exclusive) and 5.0 (inclusive); got {mult}"
                )
            validated[mode] = float(mult)
        return validated

    @field_validator("OUTPUT_BUDGET_MAX_TOKENS")
    @classmethod
    def validate_output_budget_max(cls, v: int, info) -> int:
        min_budget = info.data.get("OUTPUT_BUDGET_MIN_TOKENS", 64)
        if v < min_budget:
            raise ValueError(
                f"OUTPUT_BUDGET_MAX_TOKENS ({v}) must be greater than or equal to OUTPUT_BUDGET_MIN_TOKENS ({min_budget})"
            )
        return v

    @field_validator("COMPRESSION_TTL_SECONDS")
    @classmethod
    def validate_compression_ttl(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"COMPRESSION_TTL_SECONDS must be >= 1; got {v}")
        return v

    @field_validator("COMPRESSION_REGISTRY_MAX_ENTRIES")
    @classmethod
    def validate_compression_registry_max(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"COMPRESSION_REGISTRY_MAX_ENTRIES must be >= 1; got {v}")
        return v

    @field_validator("COMPRESSION_STRATEGIES_ENABLED")
    @classmethod
    def validate_compression_strategies(cls, v: Dict[str, bool]) -> Dict[str, bool]:
        required = {"json", "code", "logs", "text"}
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f"COMPRESSION_STRATEGIES_ENABLED missing required strategies: {sorted(missing)}")
        return v

    @field_validator("ROUTING_POLICY")
    @classmethod
    def validate_routing_policy(cls, v: str) -> str:
        normalized = v.strip().lower()
        allowed = {"auto", "prefer_small", "prefer_large"}
        if normalized not in allowed:
            raise ValueError(
                f"ROUTING_POLICY must be one of {sorted(allowed)}; got '{v}'"
            )
        return normalized

    @field_validator("ROUTING_COMPLEXITY_THRESHOLD")
    @classmethod
    def validate_routing_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"ROUTING_COMPLEXITY_THRESHOLD must be between 0.0 and 1.0; got {v}")
        return v

    @field_validator("ROUTING_VRAM_SAFETY_MARGIN_MB")
    @classmethod
    def validate_vram_margin(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"ROUTING_VRAM_SAFETY_MARGIN_MB must be non-negative; got {v}")
        return v

    @field_validator("ROUTING_RAM_SAFETY_MARGIN_MB")
    @classmethod
    def validate_ram_margin(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"ROUTING_RAM_SAFETY_MARGIN_MB must be non-negative; got {v}")
        return v

    @field_validator("ROUTING_LOADED_HEADROOM_MARGIN_MB")
    @classmethod
    def validate_loaded_headroom_margin(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"ROUTING_LOADED_HEADROOM_MARGIN_MB must be non-negative; got {v}")
        return v

    @field_validator("ROUTING_REGISTRY_TTL_SECONDS")
    @classmethod
    def validate_registry_ttl(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"ROUTING_REGISTRY_TTL_SECONDS must be positive; got {v}")
        return v

    @field_validator(
        "ROUTING_CPU_ALLOWANCE_DEFAULT_MB",
        "ROUTING_CPU_ALLOWANCE_SMALL_MB",
        "ROUTING_CPU_ALLOWANCE_MEDIUM_MB",
        "ROUTING_CPU_ALLOWANCE_LARGE_MB",
    )
    @classmethod
    def validate_cpu_allowance_mb(cls, v: int, info: ValidationInfo) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"{info.field_name} must be an integer; got {v}")
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative; got {v}")
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

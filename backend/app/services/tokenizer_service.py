"""Tokenizer Service for Argus (Phase 4).

Uses the official provider tokenizer (tiktoken for OpenAI models) when available,
and provides graceful fallback to character heuristics with explicit metadata.
"""

import math
from typing import Optional
from pydantic import BaseModel, Field

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore
    TIKTOKEN_AVAILABLE = False

from app.core.config import settings


class TokenCountResult(BaseModel):
    """Result of token counting with method provenance."""

    token_count: int = Field(description="Total token count for the text")
    char_count: int = Field(description="Total character count")
    tokenizer_name: str = Field(description="Name or encoding of tokenizer used")
    is_exact: bool = Field(description="True if computed using provider tokenizer, False if heuristic")


class TokenizerService:
    """Provides exact token counting for frontier models and heuristic fallbacks."""

    @classmethod
    def get_encoding_for_model(cls, model_name: str) -> Optional[object]:
        """Attempt to retrieve tiktoken encoding for model."""
        if not TIKTOKEN_AVAILABLE or tiktoken is None:
            return None

        # OpenAI mapping
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                # Default to o200k_base (GPT-4o) or cl100k_base (GPT-4)
                if "gpt-4o" in model_name.lower():
                    return tiktoken.get_encoding("o200k_base")
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None

    @classmethod
    def count_tokens(cls, text: str, model_name: str = "gpt-4o") -> TokenCountResult:
        """Count tokens accurately using provider tokenizer where possible."""
        char_count = len(text)
        if not text:
            return TokenCountResult(
                token_count=0,
                char_count=0,
                tokenizer_name="empty",
                is_exact=True,
            )

        encoding = cls.get_encoding_for_model(model_name)
        if encoding is not None:
            try:
                tokens = encoding.encode(text)
                enc_name = getattr(encoding, "name", "tiktoken")
                return TokenCountResult(
                    token_count=len(tokens),
                    char_count=char_count,
                    tokenizer_name=f"tiktoken ({enc_name})",
                    is_exact=True,
                )
            except Exception:
                pass

        # Documented fallback heuristic
        heuristic = settings.CHARS_PER_TOKEN_HEURISTIC or 4.0
        token_count = max(1, math.ceil(char_count / heuristic))
        return TokenCountResult(
            token_count=token_count,
            char_count=char_count,
            tokenizer_name=f"heuristic (~{heuristic} chars/token)",
            is_exact=False,
        )

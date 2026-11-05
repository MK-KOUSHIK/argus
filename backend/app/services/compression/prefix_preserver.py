"""Prefix preservation helper standardizing reusable prompt headers for KV-cache reuse."""

import re
from typing import Any, Dict, Optional, Tuple


class PrefixPreserver:
    """Stabilizes prompt prefixes and system headers to maximize native KV-cache prefix hits."""

    STANDARD_WRAPPER_PREFIX = "Original prompt to optimize:\n\n"

    @classmethod
    def standardize_system_prefix(cls, system_instruction: str) -> str:
        """Normalize system instruction formatting to ensure byte-for-byte prefix stability."""
        trimmed = system_instruction.strip()
        if not trimmed:
            return ""
        # Normalize newline runs and trailing whitespace so identical instructions match byte-for-byte
        normalized = re.sub(r"\r\n|\r", "\n", trimmed)
        normalized = re.sub(r"[ \t]+$", "", normalized, flags=re.MULTILINE)
        return normalized

    @classmethod
    def assemble_stable_prompt(
        cls, system_instruction: Optional[str], dynamic_content: str
    ) -> Tuple[str, str]:
        """Assemble full prompt with isolated, stabilized prefix for optimal KV cache alignment.

        Returns:
            Tuple of (full_assembled_prompt, standardized_prefix).
        """
        sys_prefix = cls.standardize_system_prefix(system_instruction or "")
        clean_body = dynamic_content.strip()

        if sys_prefix:
            full_prompt = f"{sys_prefix}\n\n{cls.STANDARD_WRAPPER_PREFIX}{clean_body}"
            prefix = f"{sys_prefix}\n\n{cls.STANDARD_WRAPPER_PREFIX}"
        else:
            full_prompt = f"{cls.STANDARD_WRAPPER_PREFIX}{clean_body}"
            prefix = cls.STANDARD_WRAPPER_PREFIX

        return full_prompt, prefix

    @classmethod
    def audit_native_cache_telemetry(cls, inference_telemetry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Inspect authoritative Ollama telemetry for native KV-cache prefix reuse.

        Never infers cache hits from prefix equality; strictly verifies native
        prompt_eval_cached_count reported by Ollama.
        """
        if not inference_telemetry:
            return {
                "native_cache_verified": False,
                "cached_tokens": 0,
                "reason": "telemetry_unavailable",
            }

        calls = inference_telemetry.get("calls", [])
        if not calls:
            return {
                "native_cache_verified": False,
                "cached_tokens": 0,
                "reason": "no_inference_calls_recorded",
            }

        latest_call = calls[-1]
        cached_count = latest_call.get("prompt_eval_cached_count")

        if cached_count is not None and cached_count > 0:
            return {
                "native_cache_verified": True,
                "cached_tokens": int(cached_count),
                "reason": f"native_ollama_cached_tokens={cached_count}",
            }

        return {
            "native_cache_verified": False,
            "cached_tokens": 0 if cached_count is not None else None,
            "reason": "zero_or_null_cached_tokens_reported",
        }


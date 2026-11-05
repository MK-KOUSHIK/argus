"""Safe, deterministic JSON structural compressor with deep equivalence validation."""

import json
from typing import Any, Tuple


class JSONCompressor:
    """Minifies JSON structures while strictly preserving data types, keys, values, and array order."""

    @classmethod
    def compress(cls, text: str) -> Tuple[str, bool, str]:
        """Compress JSON string and validate that parsed data structure is identical.

        Returns:
            Tuple of (compressed_text, validation_passed, reason).
        """
        stripped = text.strip()
        if not stripped:
            return text, True, "Empty input"

        # 1. Parse original data structure
        try:
            original_data = json.loads(stripped)
        except Exception as exc:
            return text, False, f"JSON parse error on original: {str(exc)}"

        # 2. Produce compact minified serialization
        try:
            compressed = json.dumps(original_data, separators=(",", ":"), ensure_ascii=False)
        except Exception as exc:
            return text, False, f"JSON serialization error: {str(exc)}"

        # 3. Validate compressed output by re-parsing and comparing against original
        try:
            reconstructed_data = json.loads(compressed)
            if not cls._deep_equals(original_data, reconstructed_data):
                return text, False, "Structural equivalence validation failed"
        except Exception as exc:
            return text, False, f"Validation parse failed on compressed output: {str(exc)}"

        # If minification did not reduce length, return original
        if len(compressed) >= len(stripped):
            return text, True, "Already optimally compact"

        return compressed, True, "Structural minification passed"

    @classmethod
    def _deep_equals(cls, a: Any, b: Any) -> bool:
        """Deep comparison verifying types, keys, values, and array ordering are preserved."""
        if type(a) is not type(b):
            # Special check for numbers (e.g. 1.0 vs 1), but keep standard Python type safety
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a == b
            return False

        if isinstance(a, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(cls._deep_equals(a[k], b[k]) for k in a)

        if isinstance(a, list):
            if len(a) != len(b):
                return False
            return all(cls._deep_equals(x, y) for x, y in zip(a, b))

        return a == b


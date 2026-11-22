"""Tokenizer Service for deterministic prompt and candidate token counting.

Provides exact token counts via:
1. Local Ollama model native tokenization when Ollama daemon is responsive and available.
2. Pairwise token measurement guaranteeing that both original and candidate prompts are
   measured under the identical model and tokenizer.
3. Fallback to tiktoken (cl100k_base) or conservative character heuristic if Ollama is unreachable.
"""

import asyncio
import math
from typing import Optional, Tuple
import httpx

from app.core.config import settings


class TokenizerService:
    """Measures token footprints using Ollama native tokenizer with safe fallback."""

    @classmethod
    async def measure_tokens(
        cls,
        text: str,
        model: Optional[str] = None,
    ) -> Tuple[int, str]:
        """Measure token count for a single text using local model tokenizer when available.

        Returns:
            Tuple of (token_count, measurement_method).
        """
        trimmed = text.strip()
        if not trimmed:
            return 0, "exact_zero"

        target_model = model or settings.OLLAMA_MODEL
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = {
            "model": target_model,
            "prompt": trimmed,
            "raw": True,
            "stream": False,
            "options": {"num_predict": 1},
        }

        try:
            timeout_sec = min(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 60.0), 30.0)
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    eval_cnt = data.get("prompt_eval_count")
                    if isinstance(eval_cnt, int) and eval_cnt >= 0:
                        return eval_cnt, "ollama_native"
        except Exception:
            pass

        # Fallback 1: tiktoken
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(trimmed)), "tiktoken_cl100k"
        except Exception:
            pass

        # Fallback 2: character heuristic (~4 chars/token)
        h = settings.CHARS_PER_TOKEN_HEURISTIC or 4.0
        return max(1, math.ceil(len(trimmed) / h)), "estimated_conservative_heuristic"

    @classmethod
    async def measure_tokens_pair(
        cls,
        original_text: str,
        candidate_text: str,
        model: Optional[str] = None,
    ) -> Tuple[int, int, str]:
        """Measure both original and candidate prompts using the identical tokenizer and model.

        Guarantees symmetric and strictly comparable token counts.
        """
        trimmed_orig = original_text.strip()
        trimmed_cand = candidate_text.strip()

        if not trimmed_orig and not trimmed_cand:
            return 0, 0, "exact_zero"
        if not trimmed_orig:
            c_cnt, m = await cls.measure_tokens(trimmed_cand, model=model)
            return 0, c_cnt, m
        if not trimmed_cand:
            o_cnt, m = await cls.measure_tokens(trimmed_orig, model=model)
            return o_cnt, 0, m

        target_model = model or settings.OLLAMA_MODEL
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"

        # Attempt Ollama native tokenization for both concurrently using the same model
        try:
            timeout_sec = min(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 60.0), 30.0)
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                req_orig = client.post(
                    url,
                    json={
                        "model": target_model,
                        "prompt": trimmed_orig,
                        "raw": True,
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                )
                req_cand = client.post(
                    url,
                    json={
                        "model": target_model,
                        "prompt": trimmed_cand,
                        "raw": True,
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                )
                resp_orig, resp_cand = await asyncio.gather(req_orig, req_cand)

                if resp_orig.status_code == 200 and resp_cand.status_code == 200:
                    cnt_orig = resp_orig.json().get("prompt_eval_count")
                    cnt_cand = resp_cand.json().get("prompt_eval_count")
                    if (
                        isinstance(cnt_orig, int)
                        and isinstance(cnt_cand, int)
                        and cnt_orig >= 0
                        and cnt_cand >= 0
                    ):
                        return cnt_orig, cnt_cand, "ollama_native"
        except Exception:
            pass

        # Fallback 1: tiktoken
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return (
                len(enc.encode(trimmed_orig)),
                len(enc.encode(trimmed_cand)),
                "tiktoken_cl100k",
            )
        except Exception:
            pass

        # Fallback 2: character heuristic
        h = settings.CHARS_PER_TOKEN_HEURISTIC or 4.0
        return (
            max(1, math.ceil(len(trimmed_orig) / h)),
            max(1, math.ceil(len(trimmed_cand) / h)),
            "estimated_conservative_heuristic",
        )

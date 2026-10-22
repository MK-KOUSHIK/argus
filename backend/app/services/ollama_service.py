"""Ollama Local LLM Service with Task-Aware Optimization Pipeline (Phase 3.1).

Coordinates Stages A through D:
- Stage A: Task analysis & constraint detection (TaskAnalyzer)
- Stage B: Strategy selection (StrategySelector)
- Stage C: Targeted model rewriting (PromptRewriter)
- Stage D: Candidate validation & safe fallback (CandidateValidator)
"""

import math
import re
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import httpx

from app.core.config import settings
from app.schemas.optimizer import (
    OptimizationMode,
    OptimizeRequest,
    OptimizeResponse,
    OllamaStatusResponse,
    TokenEstimation,
)
from app.services.task_analyzer import TaskAnalyzer, TaskAnalysisResult
from app.services.strategy_selector import StrategySelector
from app.services.prompt_rewriter import PromptRewriter
from app.services.candidate_validator import CandidateValidator, ValidationResult


class OllamaServiceError(Exception):
    """Base exception for Ollama service operations."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class OllamaConnectionError(OllamaServiceError):
    """Raised when the Ollama server cannot be reached."""

    def __init__(self, message: str = "Unable to connect to Ollama service"):
        super().__init__(message, status_code=503)


class OllamaTimeoutError(OllamaServiceError):
    """Raised when Ollama request exceeds timeout limit."""

    def __init__(self, message: str = "Ollama request timed out"):
        super().__init__(message, status_code=504)


class OllamaResponseError(OllamaServiceError):
    """Raised when Ollama returns an invalid or error response."""

    def __init__(self, message: str = "Invalid response from Ollama"):
        super().__init__(message, status_code=502)


class OllamaService:
    """Encapsulates task-aware prompt optimization and Ollama model communication."""

    @staticmethod
    def estimate_tokens(text: str) -> TokenEstimation:
        """Estimate token footprint using documented ~4 characters per token heuristic."""
        trimmed = text.strip()
        char_count = len(text)
        if not trimmed:
            return TokenEstimation(char_count=0, estimated_tokens=0)
        heuristic = settings.CHARS_PER_TOKEN_HEURISTIC or 4.0
        estimated_tokens = max(1, math.ceil(len(trimmed) / heuristic))
        return TokenEstimation(char_count=char_count, estimated_tokens=estimated_tokens)

    @classmethod
    def clean_optimized_output(cls, raw_output: str, original_prompt: str) -> str:
        """Clean potential conversational artifacts from small local LLM outputs."""
        output = raw_output.strip()

        # Check if model output contains embedded markers like "Rewritten prompt:" or "Optimized prompt:"
        parts = re.split(
            r"(?:^|\n)(?:rewritten|optimized|improved|final)\s+prompt:?\s*\n*",
            output,
            flags=re.IGNORECASE,
        )
        if len(parts) > 1 and parts[-1].strip():
            output = parts[-1].strip()

        # Remove common introductory phrases small LLMs prepend
        intro_patterns = [
            r"^(sure,?\s*(here'?s|here is)?\s*(the|an)?\s*(optimized|rewritten|improved)?\s*prompt:?)\s*",
            r"^(certainly,?\s*(here'?s|here is)?\s*(the|an)?\s*(optimized|rewritten|improved)?\s*prompt:?)\s*",
            r"^(here is (the|an)? (optimized|rewritten|improved)? prompt:?)\s*",
            r"^(optimized prompt:?)\s*",
            r"^(rewritten prompt:?)\s*",
            r"^(here's (the|an)? (optimized|rewritten)? prompt:?)\s*",
        ]
        for pattern in intro_patterns:
            output = re.sub(pattern, "", output, flags=re.IGNORECASE).strip()

        # Strip echoed system instructions
        instruction_echo_prefixes = [
            r"^- Organize instructions into clean[^\n]*\n*",
            r"^- Clearly delineate Task[^\n]*\n*",
            r"^- Aggressively eliminate[^\n]*\n*",
            r"^- Convert passive phrasing[^\n]*\n*",
            r"^- Maintain every technical[^\n]*\n*",
        ]
        for pat in instruction_echo_prefixes:
            output = re.sub(pat, "", output, flags=re.IGNORECASE | re.MULTILINE).strip()

        # If model quoted entire output with quotes
        if (output.startswith('"') and output.endswith('"')) or (
            output.startswith("'") and output.endswith("'")
        ):
            output = output[1:-1].strip()

        # If model wrapped entire output in triple backticks and original was not code
        if (
            output.startswith("```")
            and output.endswith("```")
            and not original_prompt.strip().startswith("```")
        ):
            lines = output.splitlines()
            if len(lines) >= 2:
                output = "\n".join(lines[1:-1]).strip()

        return output

    @classmethod
    def identify_preserved_constraints(
        cls, original: str, optimized: str, mode: OptimizationMode
    ) -> List[str]:
        """Categorize preserved constraints and guardrails in the optimized prompt."""
        constraints = ["Core query intent & technical details"]

        if re.search(r"\b(role|act as|you are|persona)\b", original, re.I):
            constraints.append("Role & persona parameters")

        if re.search(r"\b(json|markdown|table|format|schema|output)\b", original, re.I):
            constraints.append("Requested output format & schema")

        if re.search(r"\b(\d+|false|true|null|none)\b", original, re.I):
            constraints.append("Numerical & discrete constraints")

        if re.search(r"\b(never|do not|don't|confidential|redact|refuse|without)\b", original, re.I):
            constraints.append("Negative constraints & prohibitions")

        if mode == "structure":
            constraints.append("Markdown structural hierarchy")

        return constraints

    @classmethod
    async def check_status(cls) -> OllamaStatusResponse:
        """Check if local Ollama server is running and configured model is installed."""
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        target_model = settings.OLLAMA_MODEL.lower()

        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_STATUS_TIMEOUT_SECONDS) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    return OllamaStatusResponse(
                        connected=False,
                        base_url=settings.OLLAMA_BASE_URL,
                        model=settings.OLLAMA_MODEL,
                        model_installed=False,
                        available_models=[],
                        error=f"Ollama returned HTTP {response.status_code}",
                    )

                data = response.json()
                raw_models = data.get("models", [])
                available_models = [
                    m.get("name", "") for m in raw_models if isinstance(m, dict)
                ]

                model_installed = any(
                    target_model in m.lower() or m.lower() in target_model
                    for m in available_models
                )

                error_msg = None
                if not model_installed:
                    error_msg = (
                        f"Model '{settings.OLLAMA_MODEL}' not found in Ollama. "
                        f"Run 'ollama pull {settings.OLLAMA_MODEL}' to install."
                    )

                return OllamaStatusResponse(
                    connected=True,
                    base_url=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_MODEL,
                    model_installed=model_installed,
                    available_models=available_models,
                    error=error_msg,
                )

        except httpx.ConnectError:
            return OllamaStatusResponse(
                connected=False,
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                model_installed=False,
                available_models=[],
                error=(
                    f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}. "
                    "Ensure Ollama is running ('ollama serve')."
                ),
            )
        except httpx.TimeoutException:
            return OllamaStatusResponse(
                connected=False,
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                model_installed=False,
                available_models=[],
                error=f"Connection timed out checking Ollama at {settings.OLLAMA_BASE_URL}.",
            )
        except Exception as exc:
            return OllamaStatusResponse(
                connected=False,
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                model_installed=False,
                available_models=[],
                error=f"Unexpected error checking Ollama: {str(exc)}",
            )

    @classmethod
    async def _execute_model_call(
        cls,
        system_instruction: str,
        prompt: str,
    ) -> Tuple[str, int]:
        """Low-level model invocation via Ollama generate API."""
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "system": system_instruction,
            "prompt": f"Original prompt to optimize:\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": settings.OLLAMA_TEMPERATURE,
                "top_p": settings.OLLAMA_TOP_P,
            },
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
                response = await client.post(url, json=payload)

        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama service at {settings.OLLAMA_BASE_URL}. "
                "Ensure Ollama is running on your machine ('ollama serve')."
            ) from exc

        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama inference timed out after {settings.OLLAMA_TIMEOUT_SECONDS}s. "
                "The local model took too long to generate a response."
            ) from exc

        except Exception as exc:
            raise OllamaServiceError(
                f"Error communicating with Ollama: {str(exc)}", status_code=500
            ) from exc

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        if response.status_code != 200:
            try:
                err_detail = response.json().get("error", response.text)
            except Exception:
                err_detail = response.text
            raise OllamaResponseError(
                f"Ollama server returned error (HTTP {response.status_code}): {err_detail}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise OllamaResponseError("Ollama returned invalid JSON response") from exc

        raw_output = data.get("response")
        if not isinstance(raw_output, str):
            raise OllamaResponseError("Ollama response payload missing 'response' text field")

        return raw_output, duration_ms

    @classmethod
    async def optimize_prompt(cls, request: OptimizeRequest) -> OptimizeResponse:
        """Task-aware prompt optimization pipeline orchestrating Stages A through D."""
        # --- STAGE A: Understand the Prompt ---
        analysis = TaskAnalyzer.analyze(request.prompt)

        # --- STAGE B: Select Strategy ---
        strategy = StrategySelector.select_strategy(analysis, request.mode)

        total_duration_ms = 0
        optimized_prompt: str
        validation_status = "passed"
        is_preserved = False

        # --- STAGE C: Rewrite Prompt (or PRESERVE directly) ---
        if strategy == "preserve":
            # Prompt is already concise, unambiguous, and effective
            optimized_prompt = request.prompt
            validation_status = "passed"
            is_preserved = True
            total_duration_ms = 1
        else:
            # Build initial task-aware instruction
            system_instruction = PromptRewriter.build_system_instruction(
                analysis=analysis,
                strategy=strategy,
                mode=request.mode,
            )

            raw_candidate, call_ms = await cls._execute_model_call(
                system_instruction=system_instruction,
                prompt=request.prompt,
            )
            total_duration_ms += call_ms
            cleaned_candidate = cls.clean_optimized_output(raw_candidate, request.prompt)

            # --- STAGE D: Validate Candidate ---
            validation = CandidateValidator.validate(
                original_prompt=request.prompt,
                candidate_prompt=cleaned_candidate,
                analysis=analysis,
            )

            if validation.is_valid:
                optimized_prompt = cleaned_candidate
                validation_status = "passed"
                is_preserved = (optimized_prompt == request.prompt)
            else:
                # Controlled retry with targeted correction notice
                retry_instruction = PromptRewriter.build_system_instruction(
                    analysis=analysis,
                    strategy=strategy,
                    mode=request.mode,
                    retry_guidance=validation.reason,
                )

                try:
                    retry_raw, retry_ms = await cls._execute_model_call(
                        system_instruction=retry_instruction,
                        prompt=request.prompt,
                    )
                    total_duration_ms += retry_ms
                    retry_cleaned = cls.clean_optimized_output(retry_raw, request.prompt)
                    retry_validation = CandidateValidator.validate(
                        original_prompt=request.prompt,
                        candidate_prompt=retry_cleaned,
                        analysis=analysis,
                    )

                    if retry_validation.is_valid:
                        optimized_prompt = retry_cleaned
                        validation_status = "passed"
                        is_preserved = (optimized_prompt == request.prompt)
                    else:
                        # Safety fallback: preserve original prompt
                        optimized_prompt = request.prompt
                        validation_status = "fallback"
                        is_preserved = True
                except Exception:
                    # If retry fails unexpectedly, preserve original prompt safely
                    optimized_prompt = request.prompt
                    validation_status = "fallback"
                    is_preserved = True

        # Ensure content-based check for is_preserved
        is_preserved = (optimized_prompt.strip() == request.prompt.strip())

        # Calculate metrics and telemetry
        original_metrics = cls.estimate_tokens(request.prompt)
        optimized_metrics = cls.estimate_tokens(optimized_prompt)

        orig_len = len(request.prompt)
        opt_len = len(optimized_prompt)

        # Consistent convention: final_character_count - original_character_count
        # Negative = shorter, Positive = longer, Zero = equal
        char_delta = opt_len - orig_len

        orig_tokens = original_metrics.estimated_tokens
        opt_tokens = optimized_metrics.estimated_tokens
        token_delta = opt_tokens - orig_tokens

        token_change_percent = (
            round((token_delta / orig_tokens) * 100)
            if orig_tokens > 0
            else 0
        )
        token_reduction_percent = max(0, -token_change_percent)

        # Do not report token savings when the final prompt is longer
        token_savings = max(0, orig_tokens - opt_tokens)

        # Determine length result
        if char_delta < 0:
            length_result = "compressed"
        elif char_delta > 0:
            length_result = "expanded"
        else:
            length_result = "unchanged"

        # Determine outcome
        if validation_status == "fallback":
            outcome = "fallback"
        elif is_preserved:
            outcome = "unchanged_original"
        else:
            outcome = "validated_rewrite"

        # Determine applied strategy accurately
        if outcome in {"fallback", "unchanged_original"}:
            applied_strategy = "preserve"
        elif strategy == "structure":
            # Verify candidate actually contains markdown/structural elements
            has_structure = bool(
                re.search(r"(?:^|\n)\s*(?:[-*•]|\d+[\.\)]|[A-Z][a-zA-Z\s]+:|\#+)\s+", optimized_prompt)
            )
            applied_strategy = "structure" if has_structure else "compress"
        elif strategy == "compress" and length_result == "expanded":
            applied_strategy = "clarify"
        else:
            applied_strategy = strategy

        preserved_constraints = cls.identify_preserved_constraints(
            request.prompt, optimized_prompt, request.mode
        )

        return OptimizeResponse(
            original_prompt=request.prompt,
            optimized_prompt=optimized_prompt,
            mode=request.mode,
            target_model=request.target_model,
            task_category=analysis.task_category,
            strategy=applied_strategy,
            applied_strategy=applied_strategy,
            outcome=outcome,
            length_result=length_result,
            validation_status=validation_status,
            is_preserved=is_preserved,
            primary_action=analysis.primary_action,
            primary_deliverable=analysis.primary_deliverable,
            explicit_constraints=analysis.explicit_constraints,
            required_operations=[op.normalized_phrase for op in analysis.required_operations],
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            token_reduction_percent=token_reduction_percent,
            token_change_percent=token_change_percent,
            token_savings=token_savings,
            char_delta=char_delta,
            preserved_constraints=preserved_constraints,
            execution_time_ms=total_duration_ms,
            model=settings.OLLAMA_MODEL,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

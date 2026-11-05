"""Compression orchestrator coordinating routing, segment compression, and telemetry."""

import math
import time
from typing import Optional, Tuple

from app.core.config import settings
from app.services.adaptive_context_manager import AdaptiveContextManager
from app.services.compression.code_compressor import CodeCompressor
from app.services.compression.json_compressor import JSONCompressor
from app.services.compression.log_compressor import LogCompressor
from app.services.compression.models import CompressionResult, CompressionStrategy, ContentSegment, ContentType
from app.services.compression.prefix_preserver import PrefixPreserver
from app.services.compression.registry import CompressionRegistry
from app.services.compression.router import ContentRouter
from app.services.compression.text_compressor import TextRedundancyCompressor


class CompressionOrchestrator:
    """Central engine coordinating content-aware routing, compression, and verification."""

    @classmethod
    def compress(
        cls,
        text: str,
        enabled: Optional[bool] = None,
        reversible: Optional[bool] = None,
    ) -> CompressionResult:
        """Run content-aware compression on the input text with validation and telemetry.

        Returns:
            CompressionResult detailing compressed output, token counts, and validation outcome.
        """
        t0 = time.perf_counter()
        is_enabled = enabled if enabled is not None else settings.COMPRESSION_ENABLED
        is_reversible = reversible if reversible is not None else settings.COMPRESSION_REVERSIBLE

        orig_tokens, token_source = AdaptiveContextManager.estimate_tokens(text)

        # 1. Check if compression is globally or locally disabled or text is below threshold
        if not is_enabled or len(text.strip()) < settings.COMPRESSION_MIN_CHARS_THRESHOLD:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return CompressionResult(
                compressed_text=text,
                original_text=text,
                content_type=ContentType.TEXT,
                confidence_score=1.0,
                strategy_applied=CompressionStrategy.NONE,
                original_tokens=orig_tokens,
                compressed_tokens=orig_tokens,
                tokens_reduced=0,
                compression_ratio=1.0,
                reduction_percent=0.0,
                token_measurement_method=token_source,
                validation_passed=True,
                validation_status="passed",
                processing_time_ms=elapsed_ms,
                restoration_reference=None,
                fallback_reason="Compression disabled or text below minimum character threshold",
            )

        # 2. Classify content type and isolate segments
        content_type, confidence, segments = ContentRouter.classify(text)
        strategies_enabled = settings.COMPRESSION_STRATEGIES_ENABLED

        compressed_text = text
        strategy_applied = CompressionStrategy.NONE
        validation_passed = True
        validation_status = "passed"
        fallback_reason: Optional[str] = None
        primary_ref: Optional[str] = None

        try:
            if content_type == ContentType.MIXED:
                compressed_text, strategy_applied, validation_passed, fallback_reason, primary_ref = (
                    cls._compress_mixed_segments(segments, is_reversible, strategies_enabled)
                )

            elif content_type == ContentType.JSON:
                if strategies_enabled.get("json", True):
                    comp, ok, reason = JSONCompressor.compress(text)
                    strategy_applied = CompressionStrategy.JSON_STRUCTURAL
                    validation_passed = ok
                    if ok:
                        compressed_text = comp
                    else:
                        fallback_reason = reason
                else:
                    fallback_reason = "JSON compression strategy disabled in settings"

            elif content_type == ContentType.CODE:
                if strategies_enabled.get("code", True):
                    lang = segments[0].language if segments else None
                    comp, ok, reason = CodeCompressor.compress(text, language=lang)
                    strategy_applied = CompressionStrategy.CODE_AST
                    validation_passed = ok
                    if ok:
                        compressed_text = comp
                    else:
                        fallback_reason = reason
                else:
                    fallback_reason = "Code compression strategy disabled in settings"

            elif content_type == ContentType.LOGS:
                if strategies_enabled.get("logs", True):
                    comp, ok, reason, ref = LogCompressor.compress(text, reversible=is_reversible)
                    strategy_applied = CompressionStrategy.LOG_REPEATED
                    validation_passed = ok
                    primary_ref = ref
                    if ok:
                        compressed_text = comp
                    else:
                        fallback_reason = reason
                else:
                    fallback_reason = "Log compression strategy disabled in settings"

            else:  # Natural language text
                if strategies_enabled.get("text", True):
                    comp, ok, reason = TextRedundancyCompressor.compress(text)
                    strategy_applied = CompressionStrategy.TEXT_REDUNDANCY
                    validation_passed = ok
                    if ok:
                        compressed_text = comp
                    else:
                        fallback_reason = reason
                else:
                    fallback_reason = "Text redundancy compression strategy disabled in settings"

        except Exception as exc:
            # Absolute safety net: on any unhandled compressor exception, fall back to original
            compressed_text = text
            validation_passed = False
            validation_status = "fallback"
            fallback_reason = f"Compressor exception: {str(exc)}"

        if not validation_passed:
            compressed_text = text
            validation_status = "fallback"

        # 3. Compute resulting tokens and ratios
        comp_tokens, _ = AdaptiveContextManager.estimate_tokens(compressed_text)
        tokens_reduced = max(0, orig_tokens - comp_tokens)
        reduction_pct = round((tokens_reduced / orig_tokens) * 100, 1) if orig_tokens > 0 else 0.0
        compression_ratio = round(comp_tokens / orig_tokens, 3) if orig_tokens > 0 else 1.0

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return CompressionResult(
            compressed_text=compressed_text,
            original_text=text,
            content_type=content_type,
            confidence_score=round(confidence, 2),
            strategy_applied=strategy_applied,
            original_tokens=orig_tokens,
            compressed_tokens=comp_tokens,
            tokens_reduced=tokens_reduced,
            compression_ratio=compression_ratio,
            reduction_percent=reduction_pct,
            token_measurement_method=token_source,
            validation_passed=validation_passed,
            validation_status=validation_status,
            processing_time_ms=elapsed_ms,
            restoration_reference=primary_ref,
            fallback_reason=fallback_reason,
            segments=segments,
        )

    @classmethod
    def _compress_mixed_segments(
        cls,
        segments: list[ContentSegment],
        reversible: bool,
        strategies_enabled: dict[str, bool],
    ) -> Tuple[str, CompressionStrategy, bool, Optional[str], Optional[str]]:
        """Compress isolated segments within a mixed-content prompt."""
        assembled_parts: list[str] = []
        any_failed = False
        fallback_reasons: list[str] = []
        primary_ref: Optional[str] = None

        for seg in segments:
            seg_text = seg.content
            compressed_seg = seg_text

            if seg.content_type == ContentType.JSON and strategies_enabled.get("json", True):
                c, ok, r = JSONCompressor.compress(seg_text)
                if ok:
                    compressed_seg = c
                else:
                    any_failed = True
                    fallback_reasons.append(f"JSON: {r}")

            elif seg.content_type == ContentType.CODE and strategies_enabled.get("code", True):
                c, ok, r = CodeCompressor.compress(seg_text, language=seg.language)
                if ok:
                    compressed_seg = c
                else:
                    any_failed = True
                    fallback_reasons.append(f"Code: {r}")

            elif seg.content_type == ContentType.LOGS and strategies_enabled.get("logs", True):
                c, ok, r, ref = LogCompressor.compress(seg_text, reversible=reversible)
                if ref and not primary_ref:
                    primary_ref = ref
                if ok:
                    compressed_seg = c
                else:
                    any_failed = True
                    fallback_reasons.append(f"Logs: {r}")

            elif seg.content_type == ContentType.TEXT and strategies_enabled.get("text", True):
                c, ok, r = TextRedundancyCompressor.compress(seg_text)
                if ok:
                    compressed_seg = c
                else:
                    any_failed = True
                    fallback_reasons.append(f"Text: {r}")

            # Re-wrap fenced segments if originally fenced
            if seg.is_fenced:
                prefix = seg.fence_prefix or "```\n"
                suffix = seg.fence_suffix or "\n```"
                assembled_parts.append(f"{prefix}{compressed_seg}{suffix}")
            else:
                assembled_parts.append(compressed_seg)

        combined_text = "".join(assembled_parts)
        strategy = CompressionStrategy.MIXED
        reason_summary = "; ".join(fallback_reasons) if fallback_reasons else None

        return combined_text, strategy, not any_failed, reason_summary, primary_ref

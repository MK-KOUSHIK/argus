"""Content detection, classification, and routing for input prompts."""

import ast
import json
import re
from typing import List, Optional, Tuple

from app.services.compression.models import ContentSegment, ContentType


class ContentRouter:
    """Classifies input prompt text into structural content types with confidence scoring."""

    FENCE_PATTERN = re.compile(r"(```([a-zA-Z0-9_\-\+]*)\s*\n(.*?)\n```)", re.DOTALL)
    LOG_TIMESTAMP_PATTERN = re.compile(
        r"^(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2}|\[\d{2}:\d{2}:\d{2}\]|\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
    )
    LOG_LEVEL_PATTERN = re.compile(
        r"\b(?:DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL|TRACE|SEVERE)\b",
        re.IGNORECASE,
    )
    CODE_KEYWORDS_PYTHON = re.compile(
        r"\b(?:def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+\s+import|if\s+__name__\s*==|async\s+def|return\s+|yield\s+)\b"
    )
    CODE_KEYWORDS_JS = re.compile(
        r"\b(?:const\s+\w+|let\s+\w+|var\s+\w+|function\s+\w*|export\s+(?:default\s+)?(?:function|class|const)|console\.log)\b"
    )
    CODE_KEYWORDS_SQL = re.compile(
        r"\b(?:SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|CREATE\s+TABLE|ALTER\s+TABLE|DELETE\s+FROM)\b",
        re.IGNORECASE,
    )

    @classmethod
    def classify(cls, text: str) -> Tuple[ContentType, float, List[ContentSegment]]:
        """Classify input text and isolate segments for mixed content.

        Returns:
            Tuple of (ContentType, confidence_score, segments).
        """
        trimmed = text.strip()
        if not trimmed:
            return ContentType.TEXT, 1.0, [ContentSegment(content="", content_type=ContentType.TEXT)]

        # 1. Check for Mixed Content containing explicit markdown fences
        segments = cls.partition_fenced_segments(text)
        if len(segments) > 1:
            non_text_segments = [s for s in segments if s.content_type != ContentType.TEXT]
            if non_text_segments:
                # Prompt mixes natural language narrative with structured code/JSON/logs
                return ContentType.MIXED, 0.95, segments

        # 2. Check for pure JSON / Structured Data
        is_json, json_conf = cls._detect_json(trimmed)
        if is_json:
            return ContentType.JSON, json_conf, [ContentSegment(content=text, content_type=ContentType.JSON)]

        # 3. Check for pure Logs / Tool Output
        is_log, log_conf = cls._detect_logs(trimmed)
        if is_log:
            return ContentType.LOGS, log_conf, [ContentSegment(content=text, content_type=ContentType.LOGS)]

        # 4. Check for pure Source Code
        is_code, code_conf, detected_lang = cls._detect_code(trimmed)
        if is_code:
            return (
                ContentType.CODE,
                code_conf,
                [ContentSegment(content=text, content_type=ContentType.CODE, language=detected_lang)],
            )

        # 5. Default to Natural Language Text
        return ContentType.TEXT, 0.90, [ContentSegment(content=text, content_type=ContentType.TEXT)]

    @classmethod
    def partition_fenced_segments(cls, text: str) -> List[ContentSegment]:
        """Isolate markdown code/JSON fences from surrounding natural language text."""
        segments: List[ContentSegment] = []
        last_idx = 0

        for match in cls.FENCE_PATTERN.finditer(text):
            start, end = match.span()
            # Narrative text before this block
            if start > last_idx:
                pre_text = text[last_idx:start]
                if pre_text.strip():
                    segments.append(
                        ContentSegment(
                            content=pre_text,
                            content_type=ContentType.TEXT,
                            is_fenced=False,
                            start_idx=last_idx,
                            end_idx=start,
                        )
                    )

            lang_tag = match.group(2).lower()
            inner_content = match.group(3)
            raw_block = match.group(1)

            # Classify inner block based on tag and content
            block_type = ContentType.CODE
            detected_lang = lang_tag or "python"

            if lang_tag in {"json", "jsonc"}:
                block_type = ContentType.JSON
            elif lang_tag in {"log", "logs", "output", "trace"}:
                block_type = ContentType.LOGS
            else:
                # Inspect inner content if tag is ambiguous or generic
                is_inner_json, _ = cls._detect_json(inner_content)
                if is_inner_json:
                    block_type = ContentType.JSON
                else:
                    is_inner_log, _ = cls._detect_logs(inner_content)
                    if is_inner_log:
                        block_type = ContentType.LOGS

            segments.append(
                ContentSegment(
                    content=inner_content,
                    content_type=block_type,
                    language=detected_lang,
                    fence_prefix=f"```{match.group(2)}\n",
                    fence_suffix="\n```",
                    is_fenced=True,
                    start_idx=start,
                    end_idx=end,
                )
            )
            last_idx = end

        # Trailing narrative text
        if last_idx < len(text):
            tail_text = text[last_idx:]
            if tail_text.strip():
                segments.append(
                    ContentSegment(
                        content=tail_text,
                        content_type=ContentType.TEXT,
                        is_fenced=False,
                        start_idx=last_idx,
                        end_idx=len(text),
                    )
                )

        return segments if segments else [ContentSegment(content=text, content_type=ContentType.TEXT)]

    @classmethod
    def _detect_json(cls, text: str) -> Tuple[bool, float]:
        """Detect if input text represents valid JSON structure."""
        if not ((text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))):
            return False, 0.0

        try:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list)):
                return True, 1.0
        except Exception:
            pass
        return False, 0.0

    @classmethod
    def _detect_logs(cls, text: str) -> Tuple[bool, float]:
        """Detect if input text primarily consists of timestamped system logs or tool outputs."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False, 0.0

        timestamp_count = 0
        level_count = 0
        traceback_present = "traceback (most recent call last):" in text.lower() or "exception in thread" in text.lower()

        for line in lines[:30]:
            if cls.LOG_TIMESTAMP_PATTERN.search(line):
                timestamp_count += 1
            if cls.LOG_LEVEL_PATTERN.search(line):
                level_count += 1

        sample_size = min(len(lines), 30)
        timestamp_ratio = timestamp_count / sample_size
        level_ratio = level_count / sample_size

        if traceback_present or timestamp_ratio >= 0.35 or (timestamp_ratio >= 0.2 and level_ratio >= 0.2):
            confidence = min(0.99, max(timestamp_ratio, 0.70 if traceback_present else 0.65))
            return True, confidence

        return False, 0.0

    @classmethod
    def _detect_code(cls, text: str) -> Tuple[bool, float, Optional[str]]:
        """Detect if input text represents executable source code."""
        # 1. Try Python AST parse directly
        try:
            tree = ast.parse(text)
            # Ensure it is not merely single string constant or trivial statement
            has_func = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in ast.walk(tree))
            if has_func or cls.CODE_KEYWORDS_PYTHON.search(text):
                return True, 0.95, "python"
        except SyntaxError:
            pass

        # 2. Check JavaScript / TypeScript signatures
        if cls.CODE_KEYWORDS_JS.search(text):
            return True, 0.90, "javascript"

        # 3. Check SQL signatures
        if cls.CODE_KEYWORDS_SQL.search(text):
            return True, 0.90, "sql"

        # 4. Check for high frequency of code tokens
        code_chars = sum(text.count(c) for c in "{}();[]=><_")
        if len(text) > 50 and (code_chars / len(text)) > 0.12:
            return True, 0.75, "generic_code"

        return False, 0.0, None


"""Log and tool output compressor detecting duplicate events and folding repetitions."""

import re
from typing import List, Optional, Set, Tuple

from app.services.compression.registry import CompressionRegistry


class LogCompressor:
    """Detects repeated log lines and folding identical events while preserving errors and traces."""

    TIMESTAMP_REGEX = re.compile(
        r"^(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?|\d{2}:\d{2}:\d{2}|\[\d{2}:\d{2}:\d{2}\]|\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
    )
    SEVERITY_REGEX = re.compile(r"\b(FATAL|CRITICAL|ERROR|WARN|WARNING|SEVERE)\b", re.IGNORECASE)

    @classmethod
    def compress(cls, text: str, reversible: bool = True) -> Tuple[str, bool, str, Optional[str]]:
        """Compress logs by folding repeated runs of identical messages.

        Returns:
            Tuple of (compressed_logs, validation_passed, reason, restoration_reference).
        """
        raw_lines = text.splitlines()
        if len(raw_lines) < 3:
            return text, True, "Log volume below compression threshold", None

        # Track unique critical errors and traces to ensure 100% exact verbatim preservation
        original_critical = cls._extract_critical_errors(raw_lines)

        compressed_lines: List[str] = []
        omitted_chunks: List[str] = []
        restoration_refs: List[str] = []

        i = 0
        n = len(raw_lines)

        while i < n:
            current_line = raw_lines[i]
            is_critical = cls._is_critical_line(current_line)

            if is_critical:
                # Critical events are always preserved verbatim and never folded
                compressed_lines.append(current_line)
                i += 1
                continue

            current_sig = cls._normalize_log_line(current_line)

            # Look ahead for consecutive identical non-critical signatures
            j = i + 1
            while j < n and not cls._is_critical_line(raw_lines[j]) and cls._normalize_log_line(raw_lines[j]) == current_sig:
                j += 1

            repeat_count = j - i
            if repeat_count >= 3:
                omitted_lines = raw_lines[i + 1 : j - 1]
                start_ts = cls._extract_timestamp(raw_lines[i]) or "start"
                end_ts = cls._extract_timestamp(raw_lines[j - 1]) or "end"

                ref_marker = ""
                candidate_ref = None
                if reversible and omitted_lines:
                    omitted_text = "\n".join(omitted_lines)
                    candidate_ref = CompressionRegistry.register(omitted_text)
                    ref_marker = f" [Reference: {candidate_ref}]"

                summary = f"[... Repeated {len(omitted_lines)}x ({start_ts} -> {end_ts}){ref_marker} ...]"
                omitted_chars = sum(len(l) for l in omitted_lines) + len(omitted_lines)

                # Only fold if omitting intermediate lines genuinely saves space
                if omitted_chars > len(summary):
                    compressed_lines.append(current_line)
                    compressed_lines.append(summary)
                    compressed_lines.append(raw_lines[j - 1])
                    if candidate_ref:
                        restoration_refs.append(candidate_ref)
                    i = j
                else:
                    compressed_lines.extend(raw_lines[i:j])
                    i = j
            else:
                compressed_lines.append(current_line)
                i += 1

        compressed_text = "\n".join(compressed_lines)

        # Validate that all critical errors, severities, and stack traces are preserved verbatim
        for crit in original_critical:
            if crit not in compressed_text:
                return text, False, f"Log compression omitted critical error verbatim: {crit[:80]}", None

        if len(compressed_text) >= len(text):
            return text, True, "Logs already compact with no repeated runs", None

        primary_ref = restoration_refs[0] if restoration_refs else None
        return compressed_text, True, "Log repeated run folding passed", primary_ref

    @classmethod
    def _normalize_log_line(cls, line: str) -> str:
        """Strip timestamp, thread ids, and process ids to detect core message signature."""
        s = line.strip()
        # Remove timestamp
        s = cls.TIMESTAMP_REGEX.sub("", s).strip()
        # Remove thread / pid markers like [thread-1], [pid:123], (12345)
        s = re.sub(r"\[(?:thread|pid|worker)[\-_:\s\d]+\]", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s+", " ", s)
        return s

    @classmethod
    def _extract_timestamp(cls, line: str) -> Optional[str]:
        """Extract timestamp string from log line if present."""
        match = cls.TIMESTAMP_REGEX.search(line.strip())
        return match.group(0) if match else None

    @classmethod
    def _is_critical_line(cls, line: str) -> bool:
        """Determine if a log line represents an error, exception, trace, or critical event."""
        s = line.strip()
        if not s:
            return False
        if bool(cls.SEVERITY_REGEX.search(s)):
            return True
        if s.startswith(("File ", "Traceback", "Exception", "Error:", "Caused by:")):
            return True
        lower = s.lower()
        if "traceback (most recent call last)" in lower or "exception:" in lower:
            return True
        return False

    @classmethod
    def _extract_critical_errors(cls, lines: List[str]) -> List[str]:
        """Extract all critical error and trace lines verbatim."""
        critical = []
        for line in lines:
            if cls._is_critical_line(line):
                critical.append(line.strip())
        return critical

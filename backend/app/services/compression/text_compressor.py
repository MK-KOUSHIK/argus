"""Conservative natural language redundancy compressor with strict constraint preservation."""

import re
from typing import List, Set, Tuple


class TextRedundancyCompressor:
    """Removes duplicate sentences and repetitive instructional filler without altering constraints."""

    NEGATION_PATTERN = re.compile(
        r"\b(?:not|never|no|without|do not|don't|prohibit|prohibited|exclude|excluding|cannot|can't|must not|should not)\b",
        re.IGNORECASE,
    )
    NUMERIC_PATTERN = re.compile(
        r"\b(?:\$\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?(?:%|ms|s|m|h|gib|gb|mib|mb|kib|kb)?|\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?)\b",
        re.IGNORECASE,
    )
    IDENTIFIER_PATTERN = re.compile(
        r"\b(?:[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z0-9_]+|[A-Z_]{3,}|[a-z]+[A-Z][a-zA-Z0-9]*)\b"
    )
    CONVERSATIONAL_FILLER = [
        re.compile(r"^(?:as (?:mentioned|stated|discussed) (?:above|earlier|before),\s*)", re.IGNORECASE),
        re.compile(r"^(?:please (?:note|remember|keep in mind) that\s*)", re.IGNORECASE),
        re.compile(r"^(?:in (?:order|terms of) to achieve this,\s*)", re.IGNORECASE),
        re.compile(r"^(?:it is (?:worth noting|important to note) that\s*)", re.IGNORECASE),
    ]

    @classmethod
    def compress(cls, text: str) -> Tuple[str, bool, str]:
        """Compress natural language text by deduplicating repetitive sentences and filler.

        Returns:
            Tuple of (compressed_text, validation_passed, reason).
        """
        stripped = text.strip()
        if len(stripped) < 60:
            return text, True, "Text length below redundancy threshold"

        # 1. Extract protected entities from original text for invariant verification
        original_negations = set(cls.NEGATION_PATTERN.findall(stripped.lower()))
        original_numbers = set(cls.NUMERIC_PATTERN.findall(stripped.lower()))
        original_identifiers = set(cls.IDENTIFIER_PATTERN.findall(stripped))

        # 2. Split into paragraphs and sentences
        paragraphs = stripped.split("\n\n")
        compressed_paragraphs: List[str] = []

        seen_normalized_sentences: Set[str] = set()

        for para in paragraphs:
            para_clean = para.strip()
            if not para_clean:
                continue

            # Split paragraph into sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para_clean) if s.strip()]
            para_sentences: List[str] = []

            for sent in sentences:
                norm_sent = cls._normalize_sentence(sent)
                if not norm_sent:
                    continue

                # Strip conversational prefix filler if present
                cleaned_sent = sent
                for filler_pat in cls.CONVERSATIONAL_FILLER:
                    cleaned_sent = filler_pat.sub("", cleaned_sent).strip()
                    if cleaned_sent and cleaned_sent[0].islower():
                        cleaned_sent = cleaned_sent[0].upper() + cleaned_sent[1:]

                # Deduplicate identical or near-duplicate sentences across the prompt
                if norm_sent in seen_normalized_sentences and len(norm_sent.split()) > 4:
                    continue

                seen_normalized_sentences.add(norm_sent)
                para_sentences.append(cleaned_sent if cleaned_sent else sent)

            if para_sentences:
                compressed_paragraphs.append(" ".join(para_sentences))

        compressed_text = "\n\n".join(compressed_paragraphs).strip()

        # 3. Validation: Verify strict constraint and entity preservation
        compressed_negations = set(cls.NEGATION_PATTERN.findall(compressed_text.lower()))
        compressed_numbers = set(cls.NUMERIC_PATTERN.findall(compressed_text.lower()))
        compressed_identifiers = set(cls.IDENTIFIER_PATTERN.findall(compressed_text))

        missing_negations = original_negations - compressed_negations
        if missing_negations:
            return text, False, f"Validation failed: Negation lost: {missing_negations}"

        missing_numbers = original_numbers - compressed_numbers
        if missing_numbers:
            return text, False, f"Validation failed: Numeric value lost: {missing_numbers}"

        missing_identifiers = original_identifiers - compressed_identifiers
        if missing_identifiers:
            return text, False, f"Validation failed: Identifier lost: {missing_identifiers}"

        # Conservative semantic preservation safety:
        # If natural language text redundancy reduction dropped more than 60% of the original content,
        # semantic preservation cannot be established with high confidence. Fall back safely to original text.
        if len(compressed_text) < len(stripped) * 0.40:
            return text, False, "Conservative fallback: redundancy reduction dropped >60% of text; semantic preservation unprovable"

        if len(compressed_text) >= len(stripped):
            return text, True, "Natural language text already non-redundant"

        return compressed_text, True, "Text redundancy reduction passed"

    @classmethod
    def _normalize_sentence(cls, sentence: str) -> str:
        """Normalize sentence to lowercase alphanumeric tokens for duplicate matching."""
        s = re.sub(r"[^\w\s]", "", sentence.lower())
        return " ".join(s.split())

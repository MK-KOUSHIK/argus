"""Data models and enums for Argus Content-Aware Compression Engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ContentType(str, Enum):
    """Categorical classification of input prompt content."""

    JSON = "json"
    CODE = "code"
    LOGS = "logs"
    TEXT = "text"
    MIXED = "mixed"


class CompressionStrategy(str, Enum):
    """Specific compression technique applied to input content."""

    JSON_STRUCTURAL = "json_structural"
    CODE_AST = "code_ast"
    LOG_REPEATED = "log_repeated"
    TEXT_REDUNDANCY = "text_redundancy"
    MIXED = "mixed"
    NONE = "none"


@dataclass
class ContentSegment:
    """A distinct structural segment inside a mixed-content prompt."""

    content: str
    content_type: ContentType
    language: Optional[str] = None
    fence_prefix: Optional[str] = None
    fence_suffix: Optional[str] = None
    is_fenced: bool = False
    start_idx: int = 0
    end_idx: int = 0


@dataclass
class CompressionResult:
    """Outcome and metadata of a content compression operation."""

    compressed_text: str
    original_text: str
    content_type: ContentType
    confidence_score: float
    strategy_applied: CompressionStrategy
    original_tokens: int
    compressed_tokens: int
    tokens_reduced: int
    compression_ratio: float
    reduction_percent: float
    token_measurement_method: str
    validation_passed: bool
    validation_status: str  # "passed" or "fallback"
    processing_time_ms: float
    restoration_reference: Optional[str] = None
    fallback_reason: Optional[str] = None
    segments: List[ContentSegment] = field(default_factory=list)

"""Service layer for Argus business and domain logic."""

from .health_service import HealthService
from .ollama_service import (
    OllamaService,
    OllamaServiceError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
)
from .task_analyzer import TaskAnalyzer, TaskAnalysisResult
from .strategy_selector import StrategySelector
from .prompt_rewriter import PromptRewriter
from .candidate_validator import CandidateValidator, ValidationResult
from .compression import CompressionOrchestrator, CompressionRegistry
from .comparison_service import (
    ComparisonService,
    ComparisonServiceError,
    DownstreamModelNotFoundError,
)
from .routing import LocalModelRegistry, ModelRouter, ModelRouterError, ModelNotInstalledError

__all__ = [
    "HealthService",
    "OllamaService",
    "OllamaServiceError",
    "OllamaConnectionError",
    "OllamaTimeoutError",
    "OllamaResponseError",
    "TaskAnalyzer",
    "TaskAnalysisResult",
    "StrategySelector",
    "PromptRewriter",
    "CandidateValidator",
    "ComparisonService",
    "ComparisonServiceError",
    "DownstreamModelNotFoundError",
    "ValidationResult",
    "CompressionOrchestrator",
    "CompressionRegistry",
    "LocalModelRegistry",
    "ModelRouter",
    "ModelRouterError",
    "ModelNotInstalledError",
]

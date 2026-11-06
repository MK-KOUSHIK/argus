"""Pydantic schemas for agent conversation history and execution trajectory reduction.

Enforces:
- Observable conversation roles only ('system', 'user', 'assistant', 'tool'). No private 'thought' role.
- Strict tool-call integrity with matching tool_call_id relationships.
- Complete serialization to Ollama /api/chat supported format.
- Structured preservation decisions, reduction telemetry, context budgeting, and safe fallback indicators.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import settings

TrajectoryRole = Literal["system", "user", "assistant", "tool"]
ReductionAction = Literal["retained", "compacted", "removed"]


class ToolCallFunction(BaseModel):
    """Specification of a tool/function call."""

    name: str = Field(..., min_length=1, description="Name of the function/tool invoked")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured key-value arguments passed to the function",
    )


class ToolCall(BaseModel):
    """Observable tool call emitted by an assistant message."""

    id: str = Field(..., min_length=1, description="Unique identifier for the tool call")
    type: Literal["function"] = Field(
        default="function",
        description="Tool type (currently function)",
    )
    function: ToolCallFunction = Field(
        ...,
        description="Function call details including name and arguments",
    )


class TrajectoryMessage(BaseModel):
    """A single observable step in an agent trajectory or multi-turn dialogue."""

    role: TrajectoryRole = Field(
        ...,
        description="Observable message role: system, user, assistant, or tool",
    )
    content: Optional[str] = Field(
        default="",
        description="Text content of the message",
    )
    name: Optional[str] = Field(
        default=None,
        description="Optional identifier of the user or tool function",
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description="List of tool calls initiated by the assistant",
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Matching call ID for a 'tool' role message",
    )

    @field_validator("role")
    @classmethod
    def validate_observable_role(cls, v: str) -> str:
        valid_roles = {"system", "user", "assistant", "tool"}
        if v not in valid_roles:
            raise ValueError(f"Invalid role '{v}'. Supported observable roles are: {valid_roles}")
        return v

    @model_validator(mode="after")
    def validate_role_fields(self) -> "TrajectoryMessage":
        # Tool messages MUST have tool_call_id
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("Messages with role='tool' must provide a valid 'tool_call_id'.")
        # System and user messages must have non-empty content
        if self.role in {"system", "user"}:
            if not self.content or not self.content.strip():
                raise ValueError(f"Messages with role='{self.role}' must contain non-whitespace content.")
        return self

    def to_ollama_dict(self) -> Dict[str, Any]:
        """Convert message to the exact format expected by Ollama /api/chat."""
        d: Dict[str, Any] = {
            "role": self.role,
            "content": self.content or "",
        }
        if self.role == "assistant" and self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        elif self.role == "tool":
            if self.tool_call_id:
                d["tool_call_id"] = self.tool_call_id
            if self.name:
                d["name"] = self.name
        return d


class TrajectoryValidationError(Exception):
    """Raised when trajectory structure or tool-call integrity checks fail."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def validate_trajectory_integrity(messages: List[TrajectoryMessage]) -> None:
    """Validate tool-call and tool-result relationships across a trajectory.

    Rules:
    1. Trajectory must not be empty.
    2. Every tool-role message must reference an active, preceding assistant tool_call id (no orphans).
    3. Every assistant tool_call must be followed by its matching tool-role message before another
       assistant response or end of trajectory (no missing results).
    4. Tool call IDs must be unique across the trajectory.
    """
    if not messages:
        raise TrajectoryValidationError("Trajectory cannot be empty.")

    declared_calls: Dict[str, ToolCall] = {}
    pending_call_ids: set[str] = set()
    resolved_call_ids: set[str] = set()

    for idx, msg in enumerate(messages):
        if msg.role == "assistant" and msg.tool_calls:
            # If there were already pending tool calls that weren't resolved, that's an unfulfilled sequence
            if pending_call_ids:
                raise TrajectoryValidationError(
                    f"Message at index {idx} issued new tool calls while previous tool call(s) "
                    f"{sorted(pending_call_ids)} remain unfulfilled."
                )
            for tc in msg.tool_calls:
                if tc.id in declared_calls:
                    raise TrajectoryValidationError(f"Duplicate tool_call id '{tc.id}' detected at index {idx}.")
                declared_calls[tc.id] = tc
                pending_call_ids.add(tc.id)

        elif msg.role == "tool":
            if not msg.tool_call_id:
                raise TrajectoryValidationError(f"Tool message at index {idx} is missing 'tool_call_id'.")
            if msg.tool_call_id not in declared_calls:
                raise TrajectoryValidationError(
                    f"Orphaned tool result at index {idx}: 'tool_call_id' '{msg.tool_call_id}' "
                    "was never declared by any preceding assistant message."
                )
            if msg.tool_call_id in resolved_call_ids:
                raise TrajectoryValidationError(
                    f"Duplicate resolution for tool_call_id '{msg.tool_call_id}' at index {idx}."
                )
            pending_call_ids.discard(msg.tool_call_id)
            resolved_call_ids.add(msg.tool_call_id)

    # At the end of trajectory, all declared tool calls must have been resolved
    if pending_call_ids:
        raise TrajectoryValidationError(
            f"Trajectory has missing tool call results for pending tool_call_id(s): {sorted(pending_call_ids)}."
        )


class ReductionDecision(BaseModel):
    """Detailed record of a single message/event reduction decision."""

    message_index: int = Field(description="Original position of the message in the trajectory")
    role: TrajectoryRole = Field(description="Role of the message")
    action: ReductionAction = Field(
        description="High-level action applied to this message: 'retained', 'compacted', or 'removed'",
    )
    action_detail: str = Field(
        default="preserved_unmodified",
        description="Specific subtype/rationale label (e.g. preserved_anchor, preserved_constraint, compacted_tool_output)",
    )
    reason: str = Field(description="Deterministic rationale for the decision")
    original_char_count: int = Field(description="Original character count")
    reduced_char_count: int = Field(description="Character count after reduction")
    dependency_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional tool-call or message dependency metadata",
    )


class ContextBudgetInfo(BaseModel):
    """Context budget and tier snapping details derived from AdaptiveContextManager."""

    selected_num_ctx: int = Field(description="Snapped context window tier (e.g. 2048, 4096, 8192)")
    selection_reason: str = Field(description="Reason for tier snapping or VRAM constraint")
    estimated_input_tokens: int = Field(description="Input tokens estimated via conservative heuristic")
    total_required_context: int = Field(description="Total required context (input + reserve + output)")
    predict_tokens: int = Field(description="Reserved output prediction tokens")
    reserve_tokens: int = Field(description="Safety reserve headroom tokens")
    available_headroom: int = Field(description="Available context headroom before maximum limit")
    token_estimation_source: str = Field(
        default="estimated_conservative_heuristic",
        description="Source of token estimation; distinguishes heuristics from native counts",
    )
    eligible_tiers: List[int] = Field(
        default_factory=list,
        description="Available discrete context tiers",
    )


class TrajectoryReductionTelemetry(BaseModel):
    """Comprehensive telemetry and audit metrics for trajectory reduction."""

    original_messages_count: int = Field(description="Original number of messages in trajectory")
    reduced_messages_count: int = Field(description="Number of messages in reduced trajectory")
    messages_pruned_count: int = Field(description="Total messages completely pruned")
    tools_compacted_count: int = Field(description="Total verbose tool observations compacted")
    superseded_errors_pruned: int = Field(description="Resolved intermediate error cycles pruned")
    constraints_preserved_count: int = Field(description="Count of explicit user constraints safeguarded")
    decisions_preserved_count: int = Field(default=0, description="Count of architectural decisions safeguarded")
    original_estimated_tokens: int = Field(
        description="Estimated tokens of original trajectory using conservative heuristic (~4 chars/token). NOT native Ollama count.",
    )
    reduced_estimated_tokens: int = Field(
        description="Estimated tokens of reduced trajectory using conservative heuristic (~4 chars/token). NOT native Ollama count.",
    )
    estimated_token_savings: int = Field(
        description="Estimated token reduction (original - reduced). NOT native Ollama count.",
    )
    estimated_token_reduction_percent: Optional[float] = Field(
        default=None,
        description="Percentage reduction in tokens (((orig - red) / orig) * 100). None if orig <= 0.",
    )
    token_estimation_source: str = Field(
        default="estimated_conservative_heuristic",
        description="Source of token estimation; clearly distinguished from native Ollama telemetry",
    )
    processing_duration_ms: float = Field(description="Time taken to analyze and reduce trajectory in milliseconds")
    decisions: List[ReductionDecision] = Field(
        default_factory=list,
        description="Granular audit log of each reduction decision",
    )


class ReduceTrajectoryRequest(BaseModel):
    """Request payload for reducing multi-turn agent execution trajectory."""

    messages: List[TrajectoryMessage] = Field(
        ...,
        min_length=1,
        description="Observable message history to reduce",
    )
    target_model: Optional[str] = Field(
        default=None,
        description="Target downstream model for context budgeting (defaults to settings.OLLAMA_MODEL)",
    )
    preserve_recent_turns: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of most recent conversation turns to preserve unconditionally",
    )
    max_tool_output_chars: Optional[int] = Field(
        default=None,
        ge=50,
        le=8000,
        description="Maximum allowed character length for historical tool outputs before compacting",
    )
    reduction_profile: Optional[Literal["conservative", "balanced", "aggressive"]] = Field(
        default=None,
        description="Reduction profile aggressiveness",
    )
    include_context_budget: bool = Field(
        default=True,
        description="Whether to query AdaptiveContextManager for context tier allocation",
    )


class ReduceTrajectoryResponse(BaseModel):
    """Structured response containing reduced messages, audit telemetry, and context budget."""

    reduced_messages: List[TrajectoryMessage] = Field(
        description="Clean, reduced trajectory ready for downstream model dispatch",
    )
    original_messages: List[TrajectoryMessage] = Field(
        description="Unaltered original trajectory preserved for comparison and fallback",
    )
    telemetry: TrajectoryReductionTelemetry = Field(
        description="Detailed telemetry and decision audit log",
    )
    context_budget: Optional[ContextBudgetInfo] = Field(
        default=None,
        description="Context budget information from AdaptiveContextManager where requested",
    )
    fallback_used: bool = Field(
        default=False,
        description="True if an error or validation failure triggered safe fallback to original trajectory",
    )
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Explanation if fallback occurred",
    )

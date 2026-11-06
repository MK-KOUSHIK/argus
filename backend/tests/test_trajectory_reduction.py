"""Deterministic unit and integration tests for agent trajectory reduction (Phase 6).

Covers:
- Tool integrity validation (valid bindings, orphaned results, missing results, duplicate IDs).
- Rejection of invalid message structures (empty content, invalid roles, missing tool_call_id).
- Preservation of immutable anchors (root system prompt and active current user task).
- Preservation of historical constraints in varied natural language (passive voice, colloquial imperatives, negative requirements).
- Preservation of older architectural decisions, schema selections, and technology choices.
- Conservative retention of unrecognized user statements when importance is uncertain.
- Recent-turn retention window enforcement.
- Tool-call dependency preservation (tool calls and results preserved together).
- Compacting large tool outputs with valid message shells and intact tool_call_id bindings.
- Safe fallback behavior returning original unreduced trajectory on unexpected error.
- Disabled configuration behavior (TRAJECTORY_REDUCTION_ENABLED=False).
- FastAPI endpoint behavior (POST /api/trajectory/reduce) via TestClient.
- Non-regression of existing prompt optimization endpoint (POST /api/optimize).
- Live local Ollama integration test (conditional on local Ollama availability).
"""

import json
from typing import List
from fastapi.testclient import TestClient
import httpx
from pydantic import ValidationError
import pytest

from app.core.config import settings
from app.main import create_app
from app.schemas.trajectory import (
    ContextBudgetInfo,
    ReduceTrajectoryRequest,
    ReduceTrajectoryResponse,
    ToolCall,
    ToolCallFunction,
    TrajectoryMessage,
    TrajectoryValidationError,
    validate_trajectory_integrity,
)
from app.services.trajectory_reduction_service import TrajectoryReductionService


@pytest.fixture
def app_client():
    """Create FastAPI test client."""
    app = create_app()
    with TestClient(app) as client:
        yield client


def _create_sample_trajectory() -> List[TrajectoryMessage]:
    """Helper to create a well-formed multi-turn trajectory with tool calls."""
    return [
        TrajectoryMessage(
            role="system",
            content="You are an autonomous engineering assistant. Follow all user constraints.",
        ),
        TrajectoryMessage(
            role="user",
            content="Implement a currency formatter. Do not use the locale module under any circumstance.",
        ),
        TrajectoryMessage(
            role="assistant",
            content="I will implement it without the locale module. Let me test.",
            tool_calls=[
                ToolCall(
                    id="call_test_1",
                    type="function",
                    function=ToolCallFunction(
                        name="run_code",
                        arguments={"code": "assert format_currency(10, 'USD') == '$10.00'"},
                    ),
                )
            ],
        ),
        TrajectoryMessage(
            role="tool",
            tool_call_id="call_test_1",
            name="run_code",
            content="AssertionError: name 'format_currency' is not defined",
        ),
        TrajectoryMessage(
            role="user",
            content="Fix the function definition so it defines format_currency properly.",
        ),
        TrajectoryMessage(
            role="assistant",
            content="Fixed. Running verification.",
            tool_calls=[
                ToolCall(
                    id="call_test_2",
                    type="function",
                    function=ToolCallFunction(
                        name="run_code",
                        arguments={"code": "def format_currency(a, c): return f'${a:.2f}'\nassert format_currency(10, 'USD') == '$10.00'"},
                    ),
                )
            ],
        ),
        TrajectoryMessage(
            role="tool",
            tool_call_id="call_test_2",
            name="run_code",
            content="OK: All tests passed.",
        ),
        TrajectoryMessage(
            role="user",
            content="Write comprehensive pytest unit tests for format_currency.",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Tool Integrity Validation Tests
# ---------------------------------------------------------------------------

def test_valid_trajectory_passes_validation():
    """Well-formed trajectory with resolved tool calls must pass validation without error."""
    trajectory = _create_sample_trajectory()
    validate_trajectory_integrity(trajectory)


def test_orphaned_tool_result_raises_error():
    """A tool result referencing a non-existent tool_call_id must raise TrajectoryValidationError."""
    invalid_trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="Execute task"),
        TrajectoryMessage(
            role="tool",
            tool_call_id="call_non_existent",
            name="run_code",
            content="Execution result",
        ),
    ]
    with pytest.raises(TrajectoryValidationError) as exc_info:
        validate_trajectory_integrity(invalid_trajectory)
    assert "Orphaned tool result" in str(exc_info.value)
    assert "call_non_existent" in str(exc_info.value)


def test_missing_tool_result_raises_error():
    """An assistant tool_call without a corresponding tool result must raise TrajectoryValidationError."""
    invalid_trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="Run a script"),
        TrajectoryMessage(
            role="assistant",
            content="Running script",
            tool_calls=[
                ToolCall(
                    id="call_unresolved",
                    type="function",
                    function=ToolCallFunction(name="run_script", arguments={"script": "main.py"}),
                )
            ],
        ),
        TrajectoryMessage(role="user", content="Did it finish?"),
    ]
    with pytest.raises(TrajectoryValidationError) as exc_info:
        validate_trajectory_integrity(invalid_trajectory)
    assert "missing tool call results" in str(exc_info.value)
    assert "call_unresolved" in str(exc_info.value)


def test_duplicate_tool_call_id_raises_error():
    """Reusing the same tool_call_id across multiple assistant calls must raise TrajectoryValidationError."""
    invalid_trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="Step 1"),
        TrajectoryMessage(
            role="assistant",
            content="Calling step 1",
            tool_calls=[
                ToolCall(
                    id="duplicate_id",
                    type="function",
                    function=ToolCallFunction(name="step_1", arguments={}),
                )
            ],
        ),
        TrajectoryMessage(role="tool", tool_call_id="duplicate_id", name="step_1", content="done"),
        TrajectoryMessage(
            role="assistant",
            content="Calling step 2 with same ID",
            tool_calls=[
                ToolCall(
                    id="duplicate_id",
                    type="function",
                    function=ToolCallFunction(name="step_2", arguments={}),
                )
            ],
        ),
    ]
    with pytest.raises(TrajectoryValidationError) as exc_info:
        validate_trajectory_integrity(invalid_trajectory)
    assert "Duplicate tool_call id" in str(exc_info.value)


def test_duplicate_tool_resolution_raises_error():
    """Resolving the same tool_call_id twice must raise TrajectoryValidationError."""
    invalid_trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(
            role="assistant",
            content="Calling tool",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=ToolCallFunction(name="tool_1", arguments={}),
                )
            ],
        ),
        TrajectoryMessage(role="tool", tool_call_id="call_1", name="tool_1", content="result 1"),
        TrajectoryMessage(role="tool", tool_call_id="call_1", name="tool_1", content="result 2 (duplicate)"),
    ]
    with pytest.raises(TrajectoryValidationError) as exc_info:
        validate_trajectory_integrity(invalid_trajectory)
    assert "Duplicate resolution for tool_call_id" in str(exc_info.value)


def test_assistant_response_before_pending_tool_results_raises_error():
    """An assistant message cannot issue an ordinary response while earlier tool calls remain unresolved."""
    invalid_trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="Execute function"),
        TrajectoryMessage(
            role="assistant",
            content="Invoking tool now",
            tool_calls=[
                ToolCall(
                    id="call_pending_1",
                    type="function",
                    function=ToolCallFunction(name="fetch_data", arguments={"id": 1}),
                )
            ],
        ),
        TrajectoryMessage(
            role="assistant",
            content="I am responding before the tool returned its result.",
        ),
    ]
    with pytest.raises(TrajectoryValidationError) as exc_info:
        validate_trajectory_integrity(invalid_trajectory)
    assert "issued an ordinary response while previous tool call(s)" in str(exc_info.value)
    assert "call_pending_1" in str(exc_info.value)


def test_multiple_simultaneous_tool_calls_and_results_valid():
    """An assistant issuing multiple tool calls simultaneously, followed by matching results, is valid."""
    valid_trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="Fetch profile and orders"),
        TrajectoryMessage(
            role="assistant",
            content="Fetching both simultaneously",
            tool_calls=[
                ToolCall(
                    id="call_profile",
                    type="function",
                    function=ToolCallFunction(name="get_profile", arguments={"user_id": 10}),
                ),
                ToolCall(
                    id="call_orders",
                    type="function",
                    function=ToolCallFunction(name="get_orders", arguments={"user_id": 10}),
                ),
            ],
        ),
        TrajectoryMessage(
            role="tool",
            tool_call_id="call_profile",
            name="get_profile",
            content='{"name": "Alice"}',
        ),
        TrajectoryMessage(
            role="tool",
            tool_call_id="call_orders",
            name="get_orders",
            content='[{"order_id": 101, "total": 99.50}]',
        ),
        TrajectoryMessage(
            role="assistant",
            content="Alice has 1 order totaling $99.50.",
        ),
    ]
    validate_trajectory_integrity(valid_trajectory)
    req = ReduceTrajectoryRequest(messages=valid_trajectory)
    res = TrajectoryReductionService.reduce_trajectory(req)
    assert res.fallback_used is False
    assert len(res.reduced_messages) == len(valid_trajectory)


def test_empty_trajectory_raises_error():
    """An empty trajectory list must raise TrajectoryValidationError."""
    with pytest.raises(TrajectoryValidationError) as exc_info:
        validate_trajectory_integrity([])
    assert "Trajectory cannot be empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Schema Structure and Role Validation Tests
# ---------------------------------------------------------------------------

def test_unsupported_role_rejected():
    """Roles other than system, user, assistant, tool (e.g. 'thought') must be rejected."""
    with pytest.raises(ValidationError):
        TrajectoryMessage(role="thought", content="internal hidden reasoning")  # pyright: ignore[reportArgumentType]


def test_empty_content_in_user_or_system_rejected():
    """User and system messages cannot have empty or whitespace-only content."""
    with pytest.raises(ValidationError):
        TrajectoryMessage(role="user", content="   ")

    with pytest.raises(ValidationError):
        TrajectoryMessage(role="system", content="")


def test_tool_message_requires_tool_call_id():
    """Tool-role messages must provide a non-empty tool_call_id."""
    with pytest.raises(ValidationError):
        TrajectoryMessage(role="tool", content="Some output", name="test_tool")


def test_ollama_serialization_format():
    """Verify that to_ollama_dict() emits the exact schema expected by Ollama /api/chat."""
    assistant_msg = TrajectoryMessage(
        role="assistant",
        content="I will query the database.",
        tool_calls=[
            ToolCall(
                id="call_db_1",
                type="function",
                function=ToolCallFunction(name="exec_sql", arguments={"query": "SELECT 1"}),
            )
        ],
    )
    ollama_dict = assistant_msg.to_ollama_dict()
    assert ollama_dict["role"] == "assistant"
    assert ollama_dict["content"] == "I will query the database."
    assert "tool_calls" in ollama_dict
    assert len(ollama_dict["tool_calls"]) == 1
    tc = ollama_dict["tool_calls"][0]
    assert tc["id"] == "call_db_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "exec_sql"
    assert tc["function"]["arguments"] == {"query": "SELECT 1"}

    tool_msg = TrajectoryMessage(
        role="tool",
        content="[{'result': 1}]",
        tool_call_id="call_db_1",
        name="exec_sql",
    )
    tool_dict = tool_msg.to_ollama_dict()
    assert tool_dict["role"] == "tool"
    assert tool_dict["content"] == "[{'result': 1}]"
    assert tool_dict["tool_call_id"] == "call_db_1"
    assert tool_dict["name"] == "exec_sql"


# ---------------------------------------------------------------------------
# 3. Trajectory Reduction Rules Tests
# ---------------------------------------------------------------------------

def test_immutable_anchors_preserved():
    """Root system instructions and the latest active user task must always be preserved."""
    trajectory = _create_sample_trajectory()
    req = ReduceTrajectoryRequest(
        messages=trajectory,
        preserve_recent_turns=1,
    )
    res = TrajectoryReductionService.reduce_trajectory(req)

    # First message must be root system prompt
    assert res.reduced_messages[0].role == "system"
    assert res.reduced_messages[0].content == trajectory[0].content

    # Last message must be active user task
    assert res.reduced_messages[-1].role == "user"
    assert res.reduced_messages[-1].content == trajectory[-1].content

    # Decisions must log anchor preservation
    actions = {d.message_index: d.action for d in res.telemetry.decisions}
    action_details = {d.message_index: d.action_detail for d in res.telemetry.decisions}
    assert actions[0] == "retained"
    assert action_details[0] == "preserved_anchor"
    assert actions[len(trajectory) - 1] == "retained"
    assert action_details[len(trajectory) - 1] == "preserved_anchor"


def test_historical_user_constraints_preserved():
    """Historical user messages containing explicit negative constraints must be safeguarded."""
    trajectory = _create_sample_trajectory()
    req = ReduceTrajectoryRequest(
        messages=trajectory,
        preserve_recent_turns=1,
    )
    res = TrajectoryReductionService.reduce_trajectory(req)

    constraint_msgs = [
        m for m in res.reduced_messages
        if m.role == "user" and "locale" in (m.content or "")
    ]
    assert len(constraint_msgs) == 1
    assert constraint_msgs[0].content is not None
    assert "Do not use the locale module" in constraint_msgs[0].content

    constraint_decisions = [
        d for d in res.telemetry.decisions
        if d.action_detail == "preserved_constraint"
    ]
    assert len(constraint_decisions) >= 1
    assert res.telemetry.constraints_preserved_count >= 1


def test_constraints_in_varied_natural_language():
    """Verify detection of constraints in varied forms: passive voice, colloquial, and implicit bounds."""
    varied_phrases = [
        "The external network access is disallowed for this component.",
        "Under any circumstance, we cannot exceed the 16GB RAM quota.",
        "Make sure you never import third-party packages.",
        "The client strictly requires zero decimal places for all JPY transactions.",
        "It must run air-gapped without using cloud dependencies.",
    ]
    for phrase in varied_phrases:
        assert TrajectoryReductionService.has_explicit_constraints(phrase), f"Failed to detect constraint in: '{phrase}'"


def test_architectural_decisions_preservation():
    """Verify that historical architectural decisions and technology choices are safeguarded."""
    trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="We decided on SQLite-VSS for our embedded vector database architecture."),
        TrajectoryMessage(role="assistant", content="Acknowledged SQLite-VSS architecture."),
        TrajectoryMessage(role="user", content="Intermediate status check."),
        TrajectoryMessage(role="assistant", content="Working on schema."),
        TrajectoryMessage(role="user", content="What were our decisions?"),
    ]
    req = ReduceTrajectoryRequest(messages=trajectory, preserve_recent_turns=1)
    res = TrajectoryReductionService.reduce_trajectory(req)

    decisions = [d for d in res.telemetry.decisions if d.action_detail == "preserved_decision"]
    assert len(decisions) >= 1
    assert res.telemetry.decisions_preserved_count >= 1
    assert any("SQLite-VSS" in (m.content or "") for m in res.reduced_messages)


def test_unrecognized_statements_conservative_retention():
    """Statements without recognized keywords are conservatively retained when importance is uncertain."""
    trajectory = [
        TrajectoryMessage(role="system", content="System prompt"),
        TrajectoryMessage(role="user", content="The accounting department uses GAAP revenue recognition rules with quarterly adjustments."),
        TrajectoryMessage(role="assistant", content="Understood."),
        TrajectoryMessage(role="user", content="Intermediate question."),
        TrajectoryMessage(role="assistant", content="Answer."),
        TrajectoryMessage(role="user", content="Active final prompt."),
    ]
    req = ReduceTrajectoryRequest(messages=trajectory, preserve_recent_turns=1)
    res = TrajectoryReductionService.reduce_trajectory(req)

    # Must be retained under conservative retention policy
    uncertain_decisions = [d for d in res.telemetry.decisions if d.action_detail == "preserved_uncertain"]
    assert len(uncertain_decisions) >= 1
    assert any("GAAP revenue recognition" in (m.content or "") for m in res.reduced_messages)


def test_tool_output_compaction_preserves_tool_call_id():
    """Compacting large tool results must preserve tool_call_id, name, and valid message schema."""
    large_payload = "ROW " * 500  # 2000 characters
    trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="Query data"),
        TrajectoryMessage(
            role="assistant",
            content="Executing query",
            tool_calls=[
                ToolCall(
                    id="call_large_1",
                    type="function",
                    function=ToolCallFunction(name="db_query", arguments={"q": "SELECT * FROM large"}),
                )
            ],
        ),
        TrajectoryMessage(
            role="tool",
            tool_call_id="call_large_1",
            name="db_query",
            content=large_payload,
        ),
        TrajectoryMessage(role="user", content="Intermediate request"),
        TrajectoryMessage(role="assistant", content="Working..."),
        TrajectoryMessage(role="user", content="Final active query"),
    ]

    req = ReduceTrajectoryRequest(
        messages=trajectory,
        preserve_recent_turns=1,
        max_tool_output_chars=300,
    )
    res = TrajectoryReductionService.reduce_trajectory(req)

    compacted_tools = [m for m in res.reduced_messages if m.role == "tool"]
    assert len(compacted_tools) == 1
    compacted = compacted_tools[0]

    assert compacted.tool_call_id == "call_large_1"
    assert compacted.name == "db_query"
    assert compacted.content is not None
    assert len(compacted.content) < len(large_payload)
    assert "characters omitted" in compacted.content.lower()

    # Integrity check must succeed on the reduced trajectory
    validate_trajectory_integrity(res.reduced_messages)
    assert res.telemetry.tools_compacted_count == 1

    # Check dependency info in decision
    compacted_decisions = [d for d in res.telemetry.decisions if d.action == "compacted"]
    assert len(compacted_decisions) == 1
    assert compacted_decisions[0].dependency_info == {"tool_call_id": "call_large_1", "name": "db_query"}


def test_exact_financial_calculations_preservation():
    """Verify that historical messages containing exact financial figures are preserved as evidence when required."""
    trajectory = [
        TrajectoryMessage(role="system", content="You are a financial analysis assistant."),
        TrajectoryMessage(
            role="user",
            content="Here are our quarterly settlement amounts: Q1: $14,250.50, Q2: $18,920.00, Q3: $12,110.25.",
        ),
        TrajectoryMessage(role="assistant", content="Recorded settlement amounts."),
        TrajectoryMessage(role="user", content="Intermediate conversational check."),
        TrajectoryMessage(role="assistant", content="Acknowledged."),
        TrajectoryMessage(role="user", content="Calculate the total sum and average quarterly settlement amount."),
    ]
    req = ReduceTrajectoryRequest(messages=trajectory, preserve_recent_turns=1)
    res = TrajectoryReductionService.reduce_trajectory(req)

    # The financial evidence message must be retained
    retained_contents = [m.content for m in res.reduced_messages if m.role == "user"]
    assert any("$14,250.50" in (c or "") for c in retained_contents)

    evidence_decisions = [
        d for d in res.telemetry.decisions
        if d.action_detail == "preserved_numerical_evidence"
    ]
    assert len(evidence_decisions) >= 1
    assert "Retained source numerical facts" in evidence_decisions[0].reason


def test_large_tabular_tool_outputs_structured_compaction():
    """Verify that large JSON tabular outputs (>8 records) undergo structured record compaction preserving intact rows."""
    records = [{"order_id": 1000 + i, "amount": 100.0 + i, "status": "completed"} for i in range(20)]
    large_json_table = json.dumps(records)

    compacted_text, was_compacted = TrajectoryReductionService.compact_tool_output(
        large_json_table,
        max_chars=300,
        preserve_numerical_evidence=False,
    )
    assert was_compacted is True
    # Must parse back into valid JSON
    compacted_parsed = json.loads(compacted_text)
    assert isinstance(compacted_parsed, list)
    # Must retain first 3 and last 2 records intact without number slicing
    assert compacted_parsed[0]["order_id"] == 1000
    assert compacted_parsed[1]["order_id"] == 1001
    assert compacted_parsed[2]["order_id"] == 1002
    assert compacted_parsed[-2]["order_id"] == 1018
    assert compacted_parsed[-1]["order_id"] == 1019
    # Must contain omission indicator record
    assert any("__omitted_records__" in r for r in compacted_parsed)


def test_small_tabular_outputs_never_compacted():
    """Verify that small tabular outputs (<=8 records) are never compacted to prevent loss of exact records."""
    small_records = [{"id": i, "val": round(i * 123.45, 2)} for i in range(5)]
    small_json_table = json.dumps(small_records)

    compacted_text, was_compacted = TrajectoryReductionService.compact_tool_output(
        small_json_table,
        max_chars=50,  # even if max_chars is smaller than length
        preserve_numerical_evidence=True,
    )
    assert was_compacted is False
    assert compacted_text == small_json_table


def test_aggregation_and_filtering_requirements_preservation():
    """Verify that historical query filter criteria ('West', '2025') are retained when active task asks for aggregation."""
    trajectory = [
        TrajectoryMessage(role="system", content="You are a data analytics assistant."),
        TrajectoryMessage(
            role="user",
            content="Filter database records where region = 'West' and order_date between '2025-01-01' and '2025-12-31'.",
        ),
        TrajectoryMessage(role="assistant", content="Applied filter for West region in 2025."),
        TrajectoryMessage(role="user", content="Status update."),
        TrajectoryMessage(role="assistant", content="All filtered records ready."),
        TrajectoryMessage(role="user", content="Calculate the average revenue of these filtered records."),
    ]
    req = ReduceTrajectoryRequest(messages=trajectory, preserve_recent_turns=1)
    res = TrajectoryReductionService.reduce_trajectory(req)

    retained_texts = [m.content or "" for m in res.reduced_messages]
    assert any("region = 'West'" in t for t in retained_texts)
    filter_decisions = [
        d for d in res.telemetry.decisions
        if d.action_detail == "preserved_numerical_evidence"
    ]
    assert len(filter_decisions) >= 1


def test_missing_numerical_evidence_triggers_conservative_retention():
    """Verify that when context may contain relevant evidence without exact keywords, conservative retention preserves it."""
    trajectory = [
        TrajectoryMessage(role="system", content="You are an audit assistant."),
        TrajectoryMessage(
            role="user",
            content="Maintain ledger alignment with compliance protocol Section 4.2 pending reconciliation.",
        ),
        TrajectoryMessage(role="assistant", content="Understood."),
        TrajectoryMessage(role="user", content="Check current status."),
        TrajectoryMessage(role="assistant", content="Status ok."),
        TrajectoryMessage(role="user", content="Summarize compliance stance."),
    ]
    req = ReduceTrajectoryRequest(messages=trajectory, preserve_recent_turns=1)
    res = TrajectoryReductionService.reduce_trajectory(req)

    # Conservative retention must safeguard the historical user statement
    assert any("compliance protocol Section 4.2" in (m.content or "") for m in res.reduced_messages)
    uncertain_decisions = [
        d for d in res.telemetry.decisions
        if d.action_detail == "preserved_uncertain"
    ]
    assert len(uncertain_decisions) >= 1


def test_safe_fallback_on_validation_failure():
    """If an invalid trajectory is passed, safe fallback returns the original messages without crashing."""
    invalid_trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="Run a tool"),
        TrajectoryMessage(
            role="tool",
            tool_call_id="orphaned_call_id",
            name="some_tool",
            content="orphaned output",
        ),
        TrajectoryMessage(role="user", content="Next user query"),
    ]

    req = ReduceTrajectoryRequest(
        messages=invalid_trajectory,
        preserve_recent_turns=1,
    )
    res = TrajectoryReductionService.reduce_trajectory(req)

    assert res.fallback_used is True
    assert "Orphaned tool result" in (res.fallback_reason or "")
    assert len(res.reduced_messages) == len(invalid_trajectory)
    assert res.telemetry.estimated_token_savings == 0


def test_disabled_configuration_preserves_original(monkeypatch):
    """When TRAJECTORY_REDUCTION_ENABLED=False, trajectory is returned unmodified."""
    monkeypatch.setattr(settings, "TRAJECTORY_REDUCTION_ENABLED", False)

    trajectory = _create_sample_trajectory()
    req = ReduceTrajectoryRequest(messages=trajectory, max_tool_output_chars=50)
    res = TrajectoryReductionService.reduce_trajectory(req)

    assert res.fallback_used is False
    assert len(res.reduced_messages) == len(trajectory)
    assert res.telemetry.tools_compacted_count == 0
    assert all("disabled via configuration" in d.reason for d in res.telemetry.decisions)


def test_context_budgeting_integration():
    """Verify that AdaptiveContextManager computes context tier and headroom for trajectory."""
    trajectory = _create_sample_trajectory()
    req = ReduceTrajectoryRequest(messages=trajectory, include_context_budget=True)
    res = TrajectoryReductionService.reduce_trajectory(req)

    assert res.context_budget is not None
    assert res.context_budget.selected_num_ctx >= settings.OLLAMA_NUM_CTX
    assert res.context_budget.total_required_context > 0
    assert "Snapped to stable tier" in res.context_budget.selection_reason or "Restricted" in res.context_budget.selection_reason


# ---------------------------------------------------------------------------
# 4. FastAPI Endpoint Integration Tests (POST /api/trajectory/reduce)
# ---------------------------------------------------------------------------

def test_api_reduce_trajectory_success(app_client: TestClient):
    """API endpoint returns 200 OK with structured telemetry and reduced messages."""
    trajectory = _create_sample_trajectory()
    payload = {
        "messages": [m.model_dump() for m in trajectory],
        "preserve_recent_turns": 1,
        "max_tool_output_chars": 200,
        "include_context_budget": True,
    }
    resp = app_client.post("/api/trajectory/reduce", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "reduced_messages" in data
    assert "original_messages" in data
    assert "telemetry" in data
    assert "context_budget" in data
    assert data["fallback_used"] is False
    assert data["telemetry"]["token_estimation_source"] == "estimated_conservative_heuristic"
    assert len(data["telemetry"]["decisions"]) == len(trajectory)


def test_api_reduce_trajectory_empty_messages_validation_error(app_client: TestClient):
    """Empty messages array must fail request validation with HTTP 422."""
    payload = {"messages": []}
    resp = app_client.post("/api/trajectory/reduce", json=payload)
    assert resp.status_code == 422


def test_api_reduce_trajectory_invalid_role_validation_error(app_client: TestClient):
    """Unsupported role (e.g. 'thought') must fail validation with HTTP 422."""
    payload = {
        "messages": [
            {"role": "thought", "content": "hidden chain of thought"},
            {"role": "user", "content": "What is the answer?"},
        ]
    }
    resp = app_client.post("/api/trajectory/reduce", json=payload)
    assert resp.status_code == 422


def test_api_reduce_trajectory_orphaned_tool_graceful_fallback(app_client: TestClient):
    """Orphaned tool result does not crash API; triggers safe fallback returning original messages."""
    payload = {
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Run action"},
            {"role": "tool", "tool_call_id": "orphaned_call", "name": "test", "content": "result"},
            {"role": "user", "content": "Next step"},
        ]
    }
    resp = app_client.post("/api/trajectory/reduce", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback_used"] is True
    assert "Orphaned tool result" in data["fallback_reason"]
    assert len(data["reduced_messages"]) == 4


def test_api_generic_error_response_no_leakage(app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """API endpoint returns a generic 500 message without leaking internal exceptions or secrets."""
    def _exploding_reduce(_request):
        raise RuntimeError("CRITICAL_INTERNAL_DB_SECRET_XYZ: connection failed at /var/secret/credentials.key")

    monkeypatch.setattr(TrajectoryReductionService, "reduce_trajectory", _exploding_reduce)

    payload = {
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Execute operation"},
        ]
    }
    resp = app_client.post("/api/trajectory/reduce", json=payload)
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"]["message"] == "Failed to process trajectory reduction."
    assert "CRITICAL_INTERNAL_DB_SECRET_XYZ" not in resp.text
    assert "credentials.key" not in resp.text


def test_existing_optimizer_endpoint_regression(app_client: TestClient):
    """Verify that adding trajectory reduction does not break existing /api/optimize endpoint."""
    # Test with empty prompt to verify standard 422 validation behavior on existing endpoint
    resp = app_client.post("/api/optimize", json={"prompt": ""})
    # Optimizer schema rejects empty prompt with 422
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 5. Live Ollama Integration Test (Conditional on Local Daemon)
# ---------------------------------------------------------------------------

@pytest.mark.ollama_integration
@pytest.mark.asyncio
async def test_live_ollama_reduced_trajectory_execution():
    """Verify that a reduced trajectory executes cleanly against the live local Ollama daemon.

    Skips gracefully if Ollama is not active on the machine.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            v_resp = await client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/version")
            if v_resp.status_code != 200:
                pytest.skip("Local Ollama daemon not reachable.")
    except Exception:
        pytest.skip("Local Ollama daemon not reachable.")

    # Execute reduction
    trajectory = _create_sample_trajectory()
    req = ReduceTrajectoryRequest(
        messages=trajectory,
        preserve_recent_turns=1,
        max_tool_output_chars=250,
    )
    res = TrajectoryReductionService.reduce_trajectory(req)
    assert res.fallback_used is False

    # Send reduced trajectory to live Ollama /api/chat
    ollama_messages = [m.to_ollama_dict() for m in res.reduced_messages]
    async with httpx.AsyncClient(timeout=60.0) as client:
        chat_resp = await client.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": ollama_messages,
                "options": {
                    "num_ctx": 4096,
                    "num_predict": 128,
                    "temperature": 0.2,
                },
                "stream": False,
            },
        )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    msg = chat_data.get("message", {})
    assert msg.get("role") == "assistant"
    assert len(msg.get("content", "")) > 0


# ---------------------------------------------------------------------------
# 6. Stage 5 Comprehensive Validation Tests
# ---------------------------------------------------------------------------

def test_long_trajectory_reduction_scalability():
    """Verify reduction scalability and latency (< 50ms) across a deep 50+ message trajectory."""
    messages = [
        TrajectoryMessage(role="system", content="You are an autonomous engineering assistant."),
        TrajectoryMessage(
            role="user",
            content="Under any circumstance, we cannot exceed the 16GB RAM quota.",
        ),
    ]

    # Generate 10 superseded error cycles + verified fixes (50 messages total)
    for i in range(10):
        err_id = f"call_err_{i}"
        ok_id = f"call_ok_{i}"
        messages.extend([
            TrajectoryMessage(
                role="assistant",
                content=f"Attempting step {i} with initial code.",
                tool_calls=[
                    ToolCall(
                        id=err_id,
                        type="function",
                        function=ToolCallFunction(name="exec_task", arguments={"step": i}),
                    )
                ],
            ),
            TrajectoryMessage(
                role="tool",
                tool_call_id=err_id,
                name="exec_task",
                content=f"Traceback (most recent call last):\nIndexError: list index {i} out of range\nexit code 1",
            ),
            TrajectoryMessage(role="user", content=f"Fix the error in step {i}."),
            TrajectoryMessage(
                role="assistant",
                content=f"Fixed step {i}.",
                tool_calls=[
                    ToolCall(
                        id=ok_id,
                        type="function",
                        function=ToolCallFunction(name="exec_task", arguments={"step": i, "fixed": True}),
                    )
                ],
            ),
            TrajectoryMessage(
                role="tool",
                tool_call_id=ok_id,
                name="exec_task",
                content="OK: All tests passed. returncode 0",
            ),
        ])

    # Add active user task
    messages.append(TrajectoryMessage(role="user", content="Deploy the verified pipeline."))

    assert len(messages) >= 50

    req = ReduceTrajectoryRequest(
        messages=messages,
        preserve_recent_turns=2,
        reduction_profile="balanced",
    )
    res = TrajectoryReductionService.reduce_trajectory(req)

    # Must not trigger fallback
    assert res.fallback_used is False
    # Must preserve root system prompt and active user task
    assert res.reduced_messages[0].role == "system"
    assert res.reduced_messages[-1].content == "Deploy the verified pipeline."
    # Must preserve historical constraint
    assert any("16GB RAM quota" in (m.content or "") for m in res.reduced_messages)
    # Must prune superseded error cycles
    assert res.telemetry.superseded_errors_pruned > 0
    assert len(res.reduced_messages) < len(messages)
    # Must maintain strict tool correlation
    validate_trajectory_integrity(res.reduced_messages)
    # Scalability: processing duration must be sub-50ms (typically < 5ms)
    assert res.telemetry.processing_duration_ms < 50.0


def test_context_budget_large_trajectory_tier_snapping():
    """Verify that AdaptiveContextManager correctly sizes context and snaps tiers for trajectories."""
    trajectory = [
        TrajectoryMessage(role="system", content="You are a data assistant."),
        TrajectoryMessage(role="user", content="Here is a large dataset query log."),
        TrajectoryMessage(
            role="assistant",
            content="Running initial query.",
            tool_calls=[
                ToolCall(
                    id="call_big_1",
                    type="function",
                    function=ToolCallFunction(name="query_db", arguments={"q": "SELECT * FROM logs"}),
                )
            ],
        ),
        TrajectoryMessage(
            role="tool",
            tool_call_id="call_big_1",
            name="query_db",
            content="ROW DATA: " + ("x" * 5000),
        ),
        TrajectoryMessage(role="user", content="What was the summary?"),
    ]

    req = ReduceTrajectoryRequest(
        messages=trajectory,
        include_context_budget=True,
        max_tool_output_chars=400,
    )
    res = TrajectoryReductionService.reduce_trajectory(req)

    assert res.context_budget is not None
    assert res.context_budget.selected_num_ctx in (2048, 4096, 6144, 8192)
    assert res.context_budget.total_required_context > 0
    assert res.context_budget.available_headroom >= 0
    assert len(res.context_budget.eligible_tiers) > 0


def test_context_budget_dynamic_tier_escalation_and_overflow():
    """Demonstrate that tier selection dynamically steps through AdaptiveContextManager tiers."""
    # 1. Base small trajectory: total required context fits in configured baseline tier (OLLAMA_NUM_CTX=4096)
    base_messages = [
        TrajectoryMessage(role="system", content="Concise assistant."),
        TrajectoryMessage(role="user", content="Brief status check."),
    ]
    req_small = ReduceTrajectoryRequest(messages=base_messages, include_context_budget=True)
    res_small = TrajectoryReductionService.reduce_trajectory(req_small)
    assert res_small.context_budget is not None
    assert res_small.context_budget.selected_num_ctx == 4096
    assert res_small.context_budget.eligible_tiers == [4096, 6144, 8192]

    # 2. Medium trajectory: ~14,000 characters (~3500 tokens + reserve 256 + predict 512 = ~4268 tokens)
    # Exceeds 4096, so AdaptiveContextManager dynamically snaps to 6144
    med_messages = [
        TrajectoryMessage(role="system", content="System instruction."),
        TrajectoryMessage(role="user", content="Data block: " + ("abc " * 3500)),
        TrajectoryMessage(role="assistant", content="Acknowledged data."),
        TrajectoryMessage(role="user", content="Analyze the data."),
    ]
    req_med = ReduceTrajectoryRequest(messages=med_messages, include_context_budget=True)
    res_med = TrajectoryReductionService.reduce_trajectory(req_med)
    assert res_med.context_budget is not None
    assert res_med.context_budget.selected_num_ctx == 6144
    assert res_med.context_budget.available_headroom == (
        settings.OLLAMA_MAX_NUM_CTX
        - res_med.context_budget.estimated_input_tokens
        - res_med.context_budget.reserve_tokens
    )

    # 3. Massive trajectory: ~40,000 characters (~10,000 tokens)
    # Exceeds max_num_ctx (8192). AdaptiveContextManager raises ContextOverflowError,
    # and _calculate_context_budget safely returns None without crashing trajectory reduction.
    huge_messages = [
        TrajectoryMessage(role="system", content="System instruction."),
        TrajectoryMessage(role="user", content="Huge data: " + ("bigword " * 6000)),
        TrajectoryMessage(role="user", content="Analyze all of this."),
    ]
    req_huge = ReduceTrajectoryRequest(messages=huge_messages, include_context_budget=True)
    res_huge = TrajectoryReductionService.reduce_trajectory(req_huge)
    # Reduction executes cleanly
    assert res_huge.fallback_used is False
    # Budget safely returns None due to overflow exceeding max context
    assert res_huge.context_budget is None


def test_deterministic_reproducibility():
    """Verify that calling reduce_trajectory multiple times produces identical output and decisions."""
    trajectory = _create_sample_trajectory()
    req = ReduceTrajectoryRequest(
        messages=trajectory,
        preserve_recent_turns=1,
        max_tool_output_chars=200,
    )

    first_res = TrajectoryReductionService.reduce_trajectory(req)
    for _ in range(4):
        subsequent_res = TrajectoryReductionService.reduce_trajectory(req)
        assert len(subsequent_res.reduced_messages) == len(first_res.reduced_messages)
        for m1, m2 in zip(first_res.reduced_messages, subsequent_res.reduced_messages):
            assert m1.role == m2.role
            assert m1.content == m2.content
            assert m1.tool_call_id == m2.tool_call_id
        assert subsequent_res.telemetry.estimated_token_savings == first_res.telemetry.estimated_token_savings
        assert len(subsequent_res.telemetry.decisions) == len(first_res.telemetry.decisions)


def test_compression_orchestrator_subsystem_independence():
    """Verify that CompressionOrchestrator operates independently without interference from trajectory reduction."""
    from app.services.compression.orchestrator import CompressionOrchestrator

    single_turn_prompt = (
        "Here is a payload: "
        '{"name": "test", "items": [1, 2, 3], "metadata": {"active": true}}\n'
        "Please summarize the structure."
    )
    result = CompressionOrchestrator.compress(single_turn_prompt)

    # Subsystem produces its own independent output and telemetry
    assert result is not None
    assert isinstance(result.compressed_text, str)
    assert result.original_text == single_turn_prompt
    assert result.compressed_tokens <= result.original_tokens


def test_empty_and_single_turn_boundaries():
    """Verify boundary handling for short trajectories (<=2 messages) without unnecessary reduction overhead."""
    single_turn = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="Hello, help me with a task."),
    ]
    req = ReduceTrajectoryRequest(messages=single_turn)
    res = TrajectoryReductionService.reduce_trajectory(req)

    assert res.fallback_used is False
    assert len(res.reduced_messages) == 2
    assert res.telemetry.messages_pruned_count == 0
    assert res.telemetry.estimated_token_savings == 0
    assert all(d.action == "retained" for d in res.telemetry.decisions)


def test_trajectory_ending_with_tool_result():
    """A trajectory ending with a tool result must preserve the active user task and trailing tool event."""
    trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="Can we explore various possibilities and brainstorm?"),
        TrajectoryMessage(role="assistant", content="Sure, here are some preliminary thoughts."),
        TrajectoryMessage(role="user", content="Second user greeting"),
        TrajectoryMessage(role="assistant", content="Hello again!"),
        TrajectoryMessage(role="user", content="Calculate discount for active orders."),
        TrajectoryMessage(
            role="assistant",
            content="Querying orders",
            tool_calls=[
                ToolCall(
                    id="call_orders_latest",
                    type="function",
                    function=ToolCallFunction(name="fetch_orders", arguments={}),
                )
            ],
        ),
        TrajectoryMessage(
            role="tool",
            tool_call_id="call_orders_latest",
            name="fetch_orders",
            content='[{"id": 1, "amount": 100}]',
        ),
    ]
    validate_trajectory_integrity(trajectory)
    req = ReduceTrajectoryRequest(messages=trajectory, preserve_recent_turns=1)
    res = TrajectoryReductionService.reduce_trajectory(req)

    assert res.fallback_used is False
    decisions = {d.message_index: d for d in res.telemetry.decisions}

    # Active user task at index 5 is preserved as anchor
    assert decisions[5].action == "retained"
    assert decisions[5].action_detail == "preserved_anchor"
    assert "Active user prompt must be preserved" in decisions[5].reason

    # Trailing assistant and tool events must be preserved
    assert decisions[6].action == "retained"
    assert decisions[6].action_detail == "preserved_trailing_event"
    assert decisions[7].action == "retained"
    assert decisions[7].action_detail == "preserved_trailing_event"

    # End of reduced trajectory must be the tool result
    assert res.reduced_messages[-1].role == "tool"
    assert res.reduced_messages[-1].tool_call_id == "call_orders_latest"


def test_latest_user_task_preservation_when_conservative_retention_disabled(monkeypatch: pytest.MonkeyPatch):
    """Latest user request must be preserved as anchor even when conservative retention is disabled."""
    monkeypatch.setattr(settings, "TRAJECTORY_CONSERVATIVE_RETENTION", False)
    trajectory = [
        TrajectoryMessage(role="system", content="System instruction"),
        TrajectoryMessage(role="user", content="Can we explore various possibilities and brainstorm?"),
        TrajectoryMessage(role="assistant", content="Sure, here are some preliminary thoughts."),
        TrajectoryMessage(role="user", content="Second turn without explicit constraints"),
        TrajectoryMessage(role="assistant", content="Acknowledged, let me know when ready."),
        TrajectoryMessage(role="user", content="Now please implement the final sorting function."),
        TrajectoryMessage(
            role="assistant",
            content="Here is the implementation of the sorting function.",
        ),
    ]
    req = ReduceTrajectoryRequest(messages=trajectory, preserve_recent_turns=1)
    res = TrajectoryReductionService.reduce_trajectory(req)

    assert res.fallback_used is False
    decisions = {d.message_index: d for d in res.telemetry.decisions}

    # Historical unconstrained user message at index 1 is pruned
    assert decisions[1].action == "removed"
    assert decisions[1].action_detail == "pruned_redundant_turn"
    assert "conservative retention disabled" in decisions[1].reason

    # Active user task at index 5 MUST be preserved as anchor
    assert decisions[5].action == "retained"
    assert decisions[5].action_detail == "preserved_anchor"

    # Trailing assistant response at index 6 is preserved
    assert decisions[6].action == "retained"
    assert decisions[6].action_detail == "preserved_trailing_event"

    # Verify reduced trajectory contains active task and trailing assistant response
    roles = [m.role for m in res.reduced_messages]
    assert "system" in roles
    assert res.reduced_messages[-2].content == "Now please implement the final sorting function."
    assert res.reduced_messages[-1].content == "Here is the implementation of the sorting function."

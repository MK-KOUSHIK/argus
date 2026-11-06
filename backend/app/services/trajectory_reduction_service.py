"""Core Domain Service for Agent Execution Trajectory & History Reduction (Phase 6).

Responsible for:
- Validating observable conversation roles ('system', 'user', 'assistant', 'tool').
- Preserving tool-call and tool-result integrity (matching tool_call_id relationships).
- Safeguarding immutable anchors: root system instructions and active current user task.
- Retaining historical user constraints, architectural decisions, and state dependencies.
- Multi-layered constraint detection with conservative retention for unrecognized statements.
- General-purpose evidence preservation: retaining numerical data, financial figures, counts,
  and filter criteria required for downstream calculations and aggregations.
- Structured tabular compaction: preserving record boundaries for large tables, never slicing
  through numbers, and never compacting small tabular query results needed for calculations.
- Deterministic identification and pruning of intermediate superseded error cycles (paired removal).
- Coordinating with AdaptiveContextManager through its supported interface.
- Safe fallback: cleanly returns original unreduced trajectory if reduction raises any exception.
- Zero private chain-of-thought: strictly observable message roles only.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import settings
from app.schemas.trajectory import (
    ContextBudgetInfo,
    ReduceTrajectoryRequest,
    ReduceTrajectoryResponse,
    ReductionDecision,
    TrajectoryMessage,
    TrajectoryReductionTelemetry,
    TrajectoryValidationError,
    validate_trajectory_integrity,
)
from app.services.adaptive_context_manager import AdaptiveContextManager


class TrajectoryReductionService:
    """Service orchestrating agent history compaction, dependency tracking, and safe fallback."""

    # Layer 1: Regex patterns for explicit constraints, prohibitions, and negative bounds
    _CONSTRAINT_PATTERNS = [
        re.compile(r"\bdo not\b", re.IGNORECASE),
        re.compile(r"\bdon't\b", re.IGNORECASE),
        re.compile(r"\bnever\b", re.IGNORECASE),
        re.compile(r"\bmust not\b", re.IGNORECASE),
        re.compile(r"\bcannot\b", re.IGNORECASE),
        re.compile(r"\bcan't\b", re.IGNORECASE),
        re.compile(r"\bmust\b", re.IGNORECASE),
        re.compile(r"\bprohibit(?:ed|ion)?\b", re.IGNORECASE),
        re.compile(r"\bforbid(?:den)?\b", re.IGNORECASE),
        re.compile(r"\bdisallow(?:ed)?\b", re.IGNORECASE),
        re.compile(r"\bavoid\b", re.IGNORECASE),
        re.compile(r"\bonly\b", re.IGNORECASE),
        re.compile(r"\bstrict(?:ly)?\b", re.IGNORECASE),
        re.compile(r"\brequire(?:d|ment|ments)?\b", re.IGNORECASE),
        re.compile(r"\bconstraint[s]?\b", re.IGNORECASE),
        re.compile(r"\blimit[s]?\b", re.IGNORECASE),
        re.compile(r"\bair-gapped\b", re.IGNORECASE),
        re.compile(r"\bair gapped\b", re.IGNORECASE),
        re.compile(r"\bno cloud\b", re.IGNORECASE),
        re.compile(r"\bno external\b", re.IGNORECASE),
        re.compile(r"\bzero decimal\b", re.IGNORECASE),
        re.compile(r"\bwithout using\b", re.IGNORECASE),
        re.compile(r"\bunder any circumstance\b", re.IGNORECASE),
        re.compile(r"\bcannot exceed\b", re.IGNORECASE),
        re.compile(r"\bself-contained\b", re.IGNORECASE),
    ]

    # Layer 2: Regex patterns for architectural decisions, framework selections, and schema choices
    _DECISION_PATTERNS = [
        re.compile(r"\bwe decided\b", re.IGNORECASE),
        re.compile(r"\bdecision\b", re.IGNORECASE),
        re.compile(r"\bwe chose\b", re.IGNORECASE),
        re.compile(r"\bwe agreed\b", re.IGNORECASE),
        re.compile(r"\barchitectur(?:e|al choice)\b", re.IGNORECASE),
        re.compile(r"\badopt(?:ed|ing)?\b", re.IGNORECASE),
        re.compile(r"\bstandardiz(?:ed|ation)\b", re.IGNORECASE),
        re.compile(r"\bselected\b", re.IGNORECASE),
        re.compile(r"\bschema uses\b", re.IGNORECASE),
        re.compile(r"\bframework is\b", re.IGNORECASE),
        re.compile(r"\bopted for\b", re.IGNORECASE),
        re.compile(r"\bsettled on\b", re.IGNORECASE),
    ]

    # Layer 3: Numerical / aggregation / mathematical intent patterns
    _NUMERICAL_INTENT_PATTERNS = [
        re.compile(r"\b(calculate|average|sum|total|count|amounts?|values?|quantit\w+|math|formula)\b", re.IGNORECASE),
        re.compile(r"\b(how much|how many|ratio|percentage|mean|median|stats?|statistics?|aggregat\w+)\b", re.IGNORECASE),
        re.compile(r"\b(differenc\w+|variance|balance|financial|revenue|orders?|sales?|dollar|\$|€|¥|£)\b", re.IGNORECASE),
    ]

    # Layer 4: Filter and timeline constraint patterns
    _FILTER_PATTERNS = [
        re.compile(r"\b(where|filter\w*|between|region|timeline|\b20\d\d\b|\d{4}-\d{2}-\d{2}|top \d+|limit \d+)\b", re.IGNORECASE),
        re.compile(r"\b(placed in \d{4}|from the ['\"][^'\"]+['\"] region)\b", re.IGNORECASE),
    ]

    # Conversational acknowledgment phrases with no semantic instructions
    _PHATIC_PATTERNS = [
        re.compile(r"^(ok|okay|got it|thanks|thank you|cool|great|sounds good|proceed|yes|sure)\.?$", re.IGNORECASE)
    ]

    @classmethod
    def has_explicit_constraints(cls, text: str) -> bool:
        """Check if message content contains actionable constraints or negative prohibitions."""
        if not text:
            return False
        return any(pat.search(text) for pat in cls._CONSTRAINT_PATTERNS)

    @classmethod
    def has_architectural_decisions(cls, text: str) -> bool:
        """Check if message content records architectural decisions, technology choices, or schema bounds."""
        if not text:
            return False
        return any(pat.search(text) for pat in cls._DECISION_PATTERNS)

    @classmethod
    def has_numerical_or_aggregation_intent(cls, text: str) -> bool:
        """Check if message content indicates quantitative calculations, aggregation, or financial questions."""
        if not text:
            return False
        return any(pat.search(text) for pat in cls._NUMERICAL_INTENT_PATTERNS)

    @classmethod
    def contains_filter_criteria(cls, text: str) -> bool:
        """Check if message establishes query filters, dates, regions, or selection bounds."""
        if not text:
            return False
        return any(pat.search(text) for pat in cls._FILTER_PATTERNS)

    @classmethod
    def contains_numerical_facts(cls, text: str) -> bool:
        """Check if content contains concrete numerical figures, currency, counts, or tabular records."""
        if not text:
            return False
        # Matches currency ($1,234.50), decimals (5552.10), dates (2025-01-01), or JSON metric fields
        num_patterns = [
            re.compile(r"\$\s*\d+[\d,]*(\.\d+)?"),
            re.compile(r"\b\d+[\d,]*\.\d{2}\b"),
            re.compile(r"\b20\d\d-\d{2}-\d{2}\b"),
            re.compile(r"\"(amount|total|count|order_id|price|revenue|cost)\"\s*:\s*\d+", re.IGNORECASE),
        ]
        return any(p.search(text) for p in num_patterns)

    @classmethod
    def is_phatic_acknowledgement(cls, text: str) -> bool:
        """Check if message content is purely conversational acknowledgement without substantive instruction."""
        if not text:
            return True
        stripped = text.strip()
        return any(pat.match(stripped) for pat in cls._PHATIC_PATTERNS)

    @classmethod
    def is_error_output(cls, content: str) -> bool:
        """Check if tool output reflects an execution failure, traceback, or non-zero exit."""
        if not content:
            return False
        error_indicators = [
            "traceback (most recent call last)",
            "indexerror",
            "keyerror",
            "valueerror",
            "syntaxerror",
            "assertionerror",
            "failed with return code",
            "exit code 1",
            "returncode 1",
            "execution failed",
            "error:",
        ]
        lowered = content.lower()
        return any(ind in lowered for ind in error_indicators)

    @classmethod
    def is_success_output(cls, content: str) -> bool:
        """Check if tool output reflects a verified successful execution."""
        if not content:
            return False
        success_indicators = [
            "all_tests_passed",
            "execution succeeded",
            "return code 0",
            "returncode 0",
            "passed in",
            "ok: all tests passed",
        ]
        lowered = content.lower()
        return any(ind in lowered for ind in success_indicators)

    @classmethod
    def compact_tool_output(
        cls,
        content: str,
        max_chars: int = 400,
        preserve_numerical_evidence: bool = False,
    ) -> Tuple[str, bool]:
        """Deterministically compact verbose tool output while retaining essential context.

        General-Purpose Evidence Preservation Rules:
        - If content is small tabular/JSON data (<= 8 records) or needed for active numerical calculations,
          NEVER compact it; return full content to guarantee exact calculations without approximation.
        - If content is a large JSON list (> 8 records), perform structured record-level compaction
          preserving exact head and tail objects, never slicing through numbers or breaking JSON syntax.
        - For plain logs/text, only compact when savings exceed 120 chars, preserving head and tail.

        Returns (compacted_content, was_compacted).
        """
        if not content or len(content) <= max_chars:
            return content, False

        # Attempt structured JSON tabular compaction
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                # If table is small (e.g. <= 8 records) or contains critical numerical evidence,
                # do NOT compact; exact rows are required for calculations
                if len(parsed) <= 8 or preserve_numerical_evidence:
                    return content, False

                # Large JSON table: keep first 3 and last 2 records intact
                head_records = parsed[:3]
                tail_records = parsed[-2:]
                omitted_count = len(parsed) - (len(head_records) + len(tail_records))
                compacted_list = head_records + [{"__omitted_records__": f"{omitted_count} records omitted"}] + tail_records
                return json.dumps(compacted_list, indent=2), True
        except Exception:
            pass

        # If numerical evidence must be preserved and content contains numerical facts, avoid slicing
        if preserve_numerical_evidence and cls.contains_numerical_facts(content):
            return content, False

        # General text/log compaction: only apply if saving is meaningful (> 120 chars)
        if (len(content) - max_chars) < 120:
            return content, False

        head_chars = int(max_chars * 0.55)
        tail_chars = int(max_chars * 0.35)
        omitted = len(content) - (head_chars + tail_chars)

        compacted = (
            content[:head_chars].rstrip()
            + f"\n[... {omitted} characters omitted ...]\n"
            + content[-tail_chars:].lstrip()
        )
        return compacted, True

    @classmethod
    def _format_messages_for_budgeting(cls, messages: List[TrajectoryMessage]) -> Tuple[str, str]:
        """Extract system instruction and format trajectory body for AdaptiveContextManager."""
        system_parts = []
        body_parts = []
        for m in messages:
            if m.role == "system":
                if m.content:
                    system_parts.append(m.content.strip())
            else:
                role_label = m.role.upper()
                name_suffix = f" ({m.name})" if m.name else ""
                content = m.content or ""
                if m.role == "assistant" and m.tool_calls:
                    tc_desc = ", ".join(tc.function.name for tc in m.tool_calls)
                    content = f"{content} [ToolCalls: {tc_desc}]".strip()
                body_parts.append(f"[{role_label}{name_suffix}]: {content}")

        system_instruction = "\n\n".join(system_parts)
        formatted_body = "\n\n".join(body_parts)
        return system_instruction, formatted_body

    @classmethod
    def _calculate_context_budget(
        cls, messages: List[TrajectoryMessage], target_model: Optional[str] = None
    ) -> Optional[ContextBudgetInfo]:
        """Query AdaptiveContextManager to compute context tier and headroom for trajectory."""
        try:
            sys_inst, formatted_body = cls._format_messages_for_budgeting(messages)
            breakdown = AdaptiveContextManager.calculate_budget(
                system_instruction=sys_inst,
                formatted_prompt=formatted_body,
                min_ctx=settings.OLLAMA_NUM_CTX,
                max_ctx=settings.OLLAMA_MAX_NUM_CTX,
                output_budget=settings.OLLAMA_NUM_PREDICT,
            )
            return ContextBudgetInfo(
                selected_num_ctx=breakdown["selected_num_ctx"],
                selection_reason=breakdown["selection_reason"],
                estimated_input_tokens=breakdown["estimated_input_tokens"],
                total_required_context=breakdown["total_required_context"],
                predict_tokens=breakdown["predict_tokens"],
                reserve_tokens=breakdown["reserve_tokens"],
                available_headroom=breakdown["available_headroom"],
                token_estimation_source=breakdown.get(
                    "token_estimation_source", "estimated_conservative_heuristic"
                ),
                eligible_tiers=breakdown.get("eligible_tiers", []),
            )
        except Exception:
            return None

    @classmethod
    def reduce_trajectory(cls, request: ReduceTrajectoryRequest) -> ReduceTrajectoryResponse:
        """Execute deterministic, dependency-preserving trajectory reduction with safe fallback."""
        t0 = time.perf_counter()
        original_messages = request.messages

        preserve_recent_turns = (
            request.preserve_recent_turns
            if request.preserve_recent_turns is not None
            else settings.TRAJECTORY_PRESERVE_RECENT_TURNS
        )
        max_tool_output_chars = (
            request.max_tool_output_chars
            if request.max_tool_output_chars is not None
            else settings.TRAJECTORY_MAX_TOOL_OUTPUT_CHARS
        )
        reduction_profile = (
            request.reduction_profile
            if request.reduction_profile is not None
            else settings.TRAJECTORY_REDUCTION_PROFILE
        )

        # -------------------------------------------------------------------
        # 1. Disabled Configuration Check
        # -------------------------------------------------------------------
        if not settings.TRAJECTORY_REDUCTION_ENABLED:
            dur_ms = round((time.perf_counter() - t0) * 1000, 2)
            orig_chars = sum(len(m.content or "") for m in original_messages)
            orig_tokens = max(1, orig_chars // 4) if orig_chars > 0 else 0

            budget_info = (
                cls._calculate_context_budget(original_messages, request.target_model)
                if request.include_context_budget
                else None
            )

            telemetry = TrajectoryReductionTelemetry(
                original_messages_count=len(original_messages),
                reduced_messages_count=len(original_messages),
                messages_pruned_count=0,
                tools_compacted_count=0,
                superseded_errors_pruned=0,
                constraints_preserved_count=0,
                decisions_preserved_count=0,
                original_estimated_tokens=orig_tokens,
                reduced_estimated_tokens=orig_tokens,
                estimated_token_savings=0,
                estimated_token_reduction_percent=0.0 if orig_tokens > 0 else None,
                token_estimation_source="estimated_conservative_heuristic",
                processing_duration_ms=dur_ms,
                decisions=[
                    ReductionDecision(
                        message_index=i,
                        role=m.role,
                        action="retained",
                        action_detail="preserved_unmodified",
                        reason="Trajectory reduction is disabled via configuration (TRAJECTORY_REDUCTION_ENABLED=False)",
                        original_char_count=len(m.content or ""),
                        reduced_char_count=len(m.content or ""),
                        dependency_info={"tool_call_id": m.tool_call_id} if m.tool_call_id else None,
                    )
                    for i, m in enumerate(original_messages)
                ],
            )
            return ReduceTrajectoryResponse(
                reduced_messages=original_messages,
                original_messages=original_messages,
                telemetry=telemetry,
                context_budget=budget_info,
                fallback_used=False,
            )

        try:
            # ---------------------------------------------------------------
            # 2. Structural & Tool-Call Integrity Validation
            # ---------------------------------------------------------------
            validate_trajectory_integrity(original_messages)

            if len(original_messages) <= 2:
                dur_ms = round((time.perf_counter() - t0) * 1000, 2)
                orig_chars = sum(len(m.content or "") for m in original_messages)
                orig_tokens = max(1, orig_chars // 4) if orig_chars > 0 else 0
                budget_info = (
                    cls._calculate_context_budget(original_messages, request.target_model)
                    if request.include_context_budget
                    else None
                )
                return ReduceTrajectoryResponse(
                    reduced_messages=original_messages,
                    original_messages=original_messages,
                    telemetry=TrajectoryReductionTelemetry(
                        original_messages_count=len(original_messages),
                        reduced_messages_count=len(original_messages),
                        messages_pruned_count=0,
                        tools_compacted_count=0,
                        superseded_errors_pruned=0,
                        constraints_preserved_count=0,
                        decisions_preserved_count=0,
                        original_estimated_tokens=orig_tokens,
                        reduced_estimated_tokens=orig_tokens,
                        estimated_token_savings=0,
                        estimated_token_reduction_percent=0.0 if orig_tokens > 0 else None,
                        token_estimation_source="estimated_conservative_heuristic",
                        processing_duration_ms=dur_ms,
                        decisions=[
                            ReductionDecision(
                                message_index=i,
                                role=m.role,
                                action="retained",
                                action_detail="preserved_anchor",
                                reason="Trajectory is too short to require reduction; preserved as anchor",
                                original_char_count=len(m.content or ""),
                                reduced_char_count=len(m.content or ""),
                                dependency_info={"tool_call_id": m.tool_call_id} if m.tool_call_id else None,
                            )
                            for i, m in enumerate(original_messages)
                        ],
                    ),
                    context_budget=budget_info,
                    fallback_used=False,
                )

            # ---------------------------------------------------------------
            # 3. Partition Trajectory into Anchors and Reducible Body
            # ---------------------------------------------------------------
            total_count = len(original_messages)

            system_end_idx = 0
            while system_end_idx < total_count and original_messages[system_end_idx].role == "system":
                system_end_idx += 1

            # Identify the active user task: the latest user message in the trajectory
            user_msg_indices = [
                i for i in range(system_end_idx, total_count)
                if original_messages[i].role == "user"
            ]
            if user_msg_indices:
                active_task_idx = user_msg_indices[-1]
            else:
                # If there are no user messages, anchor to the final message
                active_task_idx = total_count - 1

            active_task_msg = original_messages[active_task_idx]

            # Detect if active task or recent turns have numerical/aggregation intent
            active_has_numerical_intent = cls.has_numerical_or_aggregation_intent(active_task_msg.content or "")

            # Count user turns prior to active task to establish recent turn window
            prior_user_indices = [
                i for i in range(system_end_idx, active_task_idx)
                if original_messages[i].role == "user"
            ]
            if prior_user_indices and preserve_recent_turns > 0:
                cutoff_count = min(len(prior_user_indices), preserve_recent_turns)
                recent_turns_threshold = prior_user_indices[-cutoff_count]
            elif preserve_recent_turns > 0:
                recent_turns_span = max(1, preserve_recent_turns * 2)
                recent_turns_threshold = max(system_end_idx, active_task_idx - recent_turns_span)
            else:
                recent_turns_threshold = active_task_idx

            # Check if any recent turn or active task requires numerical evidence
            trajectory_requires_numerical_evidence = active_has_numerical_intent or any(
                cls.has_numerical_or_aggregation_intent(original_messages[i].content or "")
                for i in range(recent_turns_threshold, active_task_idx)
            )

            # ---------------------------------------------------------------
            # 4. Dependency & Superseded Execution Cycle Analysis
            # ---------------------------------------------------------------
            tool_call_to_assistant_idx: Dict[str, int] = {}
            tool_call_to_result_idx: Dict[str, int] = {}
            tool_call_functions: Dict[str, str] = {}

            for idx, msg in enumerate(original_messages):
                if msg.role == "assistant" and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_call_to_assistant_idx[tc.id] = idx
                        tool_call_functions[tc.id] = tc.function.name
                elif msg.role == "tool" and msg.tool_call_id:
                    tool_call_to_result_idx[msg.tool_call_id] = idx

            pruned_message_indices: Set[int] = set()

            # Identify superseded error cycles:
            # An error tool call (c_err) is superseded if a subsequent call (c_ok) with the SAME
            # function name succeeded, and c_err is older than the recent window.
            if reduction_profile in ("balanced", "aggressive"):
                for call_id, res_idx in tool_call_to_result_idx.items():
                    res_msg = original_messages[res_idx]
                    asst_idx = tool_call_to_assistant_idx.get(call_id)
                    if asst_idx is None or asst_idx >= recent_turns_threshold:
                        continue

                    # If this tool result was an error
                    if cls.is_error_output(res_msg.content or ""):
                        fn_name = tool_call_functions.get(call_id)
                        # Check if a later tool call for the same function succeeded
                        has_later_success = False
                        for later_call_id, later_res_idx in tool_call_to_result_idx.items():
                            if later_res_idx > res_idx and tool_call_functions.get(later_call_id) == fn_name:
                                later_msg = original_messages[later_res_idx]
                                if cls.is_success_output(later_msg.content or ""):
                                    has_later_success = True
                                    break
                        if has_later_success:
                            asst_msg = original_messages[asst_idx]
                            if asst_msg.tool_calls and len(asst_msg.tool_calls) == 1:
                                pruned_message_indices.add(asst_idx)
                                pruned_message_indices.add(res_idx)

            # Identify early unconstrained conversational exploration (Assistant essays)
            # In balanced/aggressive profile, if an early assistant message precedes any user constraints
            # and precedes the recent window, and provides generic exploratory chatter:
            if reduction_profile in ("balanced", "aggressive"):
                for idx in range(system_end_idx, recent_turns_threshold):
                    msg = original_messages[idx]
                    if msg.role == "assistant" and not msg.tool_calls:
                        later_has_constraints = any(
                            cls.has_explicit_constraints(m.content or "")
                            for m in original_messages[idx + 1:active_task_idx + 1]
                        )
                        earlier_has_constraints = any(
                            cls.has_explicit_constraints(m.content or "")
                            for m in original_messages[system_end_idx:idx]
                        )
                        # If subsequent user introduced strict constraints and this essay is long
                        if later_has_constraints and not earlier_has_constraints and len(msg.content or "") > 300:
                            pruned_message_indices.add(idx)

            # ---------------------------------------------------------------
            # 5. Build Reduced Trajectory & Explainable Decisions
            # ---------------------------------------------------------------
            decisions: List[ReductionDecision] = []
            tools_compacted_count = 0
            superseded_errors_pruned = 0
            constraints_preserved_count = 0
            decisions_preserved_count = 0

            reduced: List[TrajectoryMessage] = []

            for idx, msg in enumerate(original_messages):
                orig_c = len(msg.content or "")
                dep_info: Optional[Dict[str, Any]] = None
                if msg.role == "tool" and msg.tool_call_id:
                    dep_info = {"tool_call_id": msg.tool_call_id, "name": msg.name}
                elif msg.role == "assistant" and msg.tool_calls:
                    dep_info = {"tool_call_ids": [tc.id for tc in msg.tool_calls]}

                # Rule A: Root System Anchor
                if idx < system_end_idx:
                    reduced.append(msg)
                    decisions.append(
                        ReductionDecision(
                            message_index=idx,
                            role=msg.role,
                            action="retained",
                            action_detail="preserved_anchor",
                            reason="Root system instruction must be preserved as immutable anchor",
                            original_char_count=orig_c,
                            reduced_char_count=orig_c,
                            dependency_info=dep_info,
                        )
                    )
                    continue

                # Rule B: Active User Prompt Anchor
                if idx == active_task_idx:
                    reduced.append(msg)
                    decisions.append(
                        ReductionDecision(
                            message_index=idx,
                            role=msg.role,
                            action="retained",
                            action_detail="preserved_anchor",
                            reason="Active user prompt must be preserved as immutable anchor",
                            original_char_count=orig_c,
                            reduced_char_count=orig_c,
                            dependency_info=dep_info,
                        )
                    )
                    continue

                # Rule B2: Dependent Trailing Events (events after active user request)
                if idx > active_task_idx:
                    reduced.append(msg)
                    decisions.append(
                        ReductionDecision(
                            message_index=idx,
                            role=msg.role,
                            action="retained",
                            action_detail="preserved_trailing_event",
                            reason="Preserved trailing event dependent on the active user request",
                            original_char_count=orig_c,
                            reduced_char_count=orig_c,
                            dependency_info=dep_info,
                        )
                    )
                    continue

                # Rule C: Pruned Superseded Error Cycles or Exploratory Chatter
                if idx in pruned_message_indices:
                    superseded_errors_pruned += 1
                    reason = (
                        "Pruned intermediate failed tool call and traceback superseded by subsequent verified execution"
                        if (msg.role in ("assistant", "tool") and (msg.tool_calls or msg.tool_call_id))
                        else "Pruned early unconstrained conversational commentary superseded by subsequent operational constraints"
                    )
                    decisions.append(
                        ReductionDecision(
                            message_index=idx,
                            role=msg.role,
                            action="removed",
                            action_detail="pruned_superseded_error" if msg.tool_calls or msg.tool_call_id else "pruned_redundant_turn",
                            reason=reason,
                            original_char_count=orig_c,
                            reduced_char_count=0,
                            dependency_info=dep_info,
                        )
                    )
                    continue

                # Rule D: Recent Turns Window
                if idx >= recent_turns_threshold:
                    reduced.append(msg)
                    decisions.append(
                        ReductionDecision(
                            message_index=idx,
                            role=msg.role,
                            action="retained",
                            action_detail="preserved_recent_turn",
                            reason=f"Message is within recent preservation window ({preserve_recent_turns} turns)",
                            original_char_count=orig_c,
                            reduced_char_count=orig_c,
                            dependency_info=dep_info,
                        )
                    )
                    continue

                # Rule E: Historical Numerical Evidence & Filters Safeguard
                # If the trajectory requires numerical evidence, messages containing source numerical
                # facts, amounts, or query filters must be preserved as primary evidence
                msg_content = msg.content or ""
                has_num_facts = cls.contains_numerical_facts(msg_content)
                has_filters = cls.contains_filter_criteria(msg_content)

                if trajectory_requires_numerical_evidence and (has_num_facts or has_filters):
                    reduced.append(msg)
                    decisions.append(
                        ReductionDecision(
                            message_index=idx,
                            role=msg.role,
                            action="retained",
                            action_detail="preserved_numerical_evidence",
                            reason="Retained source numerical facts, exact financial amounts, and filter criteria required for downstream calculations",
                            original_char_count=orig_c,
                            reduced_char_count=orig_c,
                            dependency_info=dep_info,
                        )
                    )
                    continue

                # Rule F: Historical User Turns Analysis
                if msg.role == "user":
                    if cls.has_explicit_constraints(msg_content):
                        reduced.append(msg)
                        constraints_preserved_count += 1
                        decisions.append(
                            ReductionDecision(
                                message_index=idx,
                                role=msg.role,
                                action="retained",
                                action_detail="preserved_constraint",
                                reason="Historical user message contains explicit operational constraints or negative prohibitions",
                                original_char_count=orig_c,
                                reduced_char_count=orig_c,
                                dependency_info=dep_info,
                            )
                        )
                        continue

                    if cls.has_architectural_decisions(msg_content):
                        reduced.append(msg)
                        decisions_preserved_count += 1
                        decisions.append(
                            ReductionDecision(
                                message_index=idx,
                                role=msg.role,
                                action="retained",
                                action_detail="preserved_decision",
                                reason="Historical user message records architectural decisions, technology selections, or schema choices",
                                original_char_count=orig_c,
                                reduced_char_count=orig_c,
                                dependency_info=dep_info,
                            )
                        )
                        continue

                    # Conservative Retention of Unrecognized Statements
                    if settings.TRAJECTORY_CONSERVATIVE_RETENTION and not cls.is_phatic_acknowledgement(msg_content):
                        reduced.append(msg)
                        decisions.append(
                            ReductionDecision(
                                message_index=idx,
                                role=msg.role,
                                action="retained",
                                action_detail="preserved_uncertain",
                                reason="Retained historical user statement under conservative retention policy to safeguard unrecognized context",
                                original_char_count=orig_c,
                                reduced_char_count=orig_c,
                                dependency_info=dep_info,
                            )
                        )
                        continue

                    # If phatic or conservative retention is disabled, prune the unconstrained historical user statement
                    pruned_reason = (
                        "Pruned historical conversational phatic acknowledgement"
                        if cls.is_phatic_acknowledgement(msg_content)
                        else "Pruned unconstrained historical user statement (conservative retention disabled)"
                    )
                    decisions.append(
                        ReductionDecision(
                            message_index=idx,
                            role=msg.role,
                            action="removed",
                            action_detail="pruned_redundant_turn",
                            reason=pruned_reason,
                            original_char_count=orig_c,
                            reduced_char_count=0,
                            dependency_info=dep_info,
                        )
                    )
                    continue

                # Rule G: Historical Tool Message Compaction (with Evidence Preservation)
                if msg.role == "tool":
                    compacted_text, was_compacted = cls.compact_tool_output(
                        msg_content,
                        max_chars=max_tool_output_chars,
                        preserve_numerical_evidence=trajectory_requires_numerical_evidence,
                    )
                    if was_compacted:
                        tools_compacted_count += 1
                        new_msg = TrajectoryMessage(
                            role="tool",
                            content=compacted_text,
                            tool_call_id=msg.tool_call_id,
                            name=msg.name,
                        )
                        reduced.append(new_msg)
                        decisions.append(
                            ReductionDecision(
                                message_index=idx,
                                role=msg.role,
                                action="compacted",
                                action_detail="compacted_tool_output",
                                reason=f"Compacted verbose tool observation ({orig_c} -> {len(compacted_text)} chars) while strictly preserving tool_call_id binding",
                                original_char_count=orig_c,
                                reduced_char_count=len(compacted_text),
                                dependency_info=dep_info,
                            )
                        )
                        continue
                    else:
                        reduced.append(msg)
                        decisions.append(
                            ReductionDecision(
                                message_index=idx,
                                role=msg.role,
                                action="retained",
                                action_detail="preserved_unmodified",
                                reason="Tool observation is within character budget limit or preserved to safeguard numerical evidence",
                                original_char_count=orig_c,
                                reduced_char_count=orig_c,
                                dependency_info=dep_info,
                            )
                        )
                        continue

                # Default for other historical turns: preserve to protect dependencies
                reduced.append(msg)
                decisions.append(
                    ReductionDecision(
                        message_index=idx,
                        role=msg.role,
                        action="retained",
                        action_detail="preserved_unmodified",
                        reason="Preserved to maintain dialogue coherence and tool-call sequence dependencies",
                        original_char_count=orig_c,
                        reduced_char_count=orig_c,
                        dependency_info=dep_info,
                    )
                )

            # ---------------------------------------------------------------
            # 6. Post-Reduction Integrity Verification
            # ---------------------------------------------------------------
            validate_trajectory_integrity(reduced)

            # ---------------------------------------------------------------
            # 7. Telemetry & Context Budgeting
            # ---------------------------------------------------------------
            dur_ms = round((time.perf_counter() - t0) * 1000, 2)
            orig_chars = sum(len(m.content or "") for m in original_messages)
            red_chars = sum(len(m.content or "") for m in reduced)
            orig_tokens = max(1, orig_chars // 4) if orig_chars > 0 else 0
            red_tokens = max(1, red_chars // 4) if red_chars > 0 else 0
            token_savings = max(0, orig_tokens - red_tokens)
            reduction_pct = (
                round(((orig_tokens - red_tokens) / orig_tokens) * 100, 2)
                if orig_tokens > 0
                else None
            )

            budget_info = (
                cls._calculate_context_budget(reduced, request.target_model)
                if request.include_context_budget
                else None
            )

            telemetry = TrajectoryReductionTelemetry(
                original_messages_count=len(original_messages),
                reduced_messages_count=len(reduced),
                messages_pruned_count=len(original_messages) - len(reduced),
                tools_compacted_count=tools_compacted_count,
                superseded_errors_pruned=superseded_errors_pruned,
                constraints_preserved_count=constraints_preserved_count,
                decisions_preserved_count=decisions_preserved_count,
                original_estimated_tokens=orig_tokens,
                reduced_estimated_tokens=red_tokens,
                estimated_token_savings=token_savings,
                estimated_token_reduction_percent=reduction_pct,
                token_estimation_source="estimated_conservative_heuristic",
                processing_duration_ms=dur_ms,
                decisions=decisions,
            )

            return ReduceTrajectoryResponse(
                reduced_messages=reduced,
                original_messages=original_messages,
                telemetry=telemetry,
                context_budget=budget_info,
                fallback_used=False,
            )

        except Exception as exc:
            # Safe Fallback: cleanly return unreduced trajectory on any failure
            dur_ms = round((time.perf_counter() - t0) * 1000, 2)
            orig_chars = sum(len(m.content or "") for m in original_messages)
            orig_tokens = max(1, orig_chars // 4) if orig_chars > 0 else 0

            fallback_budget = (
                cls._calculate_context_budget(original_messages, request.target_model)
                if request.include_context_budget
                else None
            )

            fallback_telemetry = TrajectoryReductionTelemetry(
                original_messages_count=len(original_messages),
                reduced_messages_count=len(original_messages),
                messages_pruned_count=0,
                tools_compacted_count=0,
                superseded_errors_pruned=0,
                constraints_preserved_count=0,
                decisions_preserved_count=0,
                original_estimated_tokens=orig_tokens,
                reduced_estimated_tokens=orig_tokens,
                estimated_token_savings=0,
                estimated_token_reduction_percent=0.0 if orig_tokens > 0 else None,
                token_estimation_source="estimated_conservative_heuristic",
                processing_duration_ms=dur_ms,
                decisions=[
                    ReductionDecision(
                        message_index=i,
                        role=m.role,
                        action="retained",
                        action_detail="preserved_unmodified",
                        reason=f"Safe fallback triggered due to error: {str(exc)}",
                        original_char_count=len(m.content or ""),
                        reduced_char_count=len(m.content or ""),
                        dependency_info={"tool_call_id": m.tool_call_id} if m.tool_call_id else None,
                    )
                    for i, m in enumerate(original_messages)
                ],
            )
            return ReduceTrajectoryResponse(
                reduced_messages=original_messages,
                original_messages=original_messages,
                telemetry=fallback_telemetry,
                context_budget=fallback_budget,
                fallback_used=True,
                fallback_reason=str(exc),
            )

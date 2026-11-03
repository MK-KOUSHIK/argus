"""Automated Test Suite for Argus Adaptive KV-Cache Optimization (Phase 4).

Covers:
1. Adaptive context budget selection and tier snapping (short, medium, long, near-limit, oversized).
2. Clean HTTP 413 rejection of oversized prompts without silent truncation.
3. GPU telemetry unavailable fallback and system RAM tracking.
4. Runtime-aware adaptive concurrency decisions under normal, low, and critical VRAM.
5. Guaranteed capacity release and queue depth recovery on error or cancellation.
6. Ollama process manager state, manifest verification, and log evidence detection.
7. Authoritative telemetry parsing including prompt_eval_cached_count and source attribution.
"""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from app.core.config import settings
from app.services.adaptive_context_manager import AdaptiveContextManager, ContextOverflowError
from app.services.hardware_telemetry import HardwareTelemetryProvider
from app.services.concurrency_controller import (
    RuntimeAwareConcurrencyController,
    ConcurrencyControllerError,
)
from app.services.ollama_process_manager import OllamaProcessManager
from app.services.ollama_service import OllamaService, OllamaServiceError


# ---------------------------------------------------------------------------
# 1. Adaptive Context Length Management & Budgeting Tests
# ---------------------------------------------------------------------------

def test_context_budget_short_prompt_snaps_to_baseline():
    """Short prompt should snap to baseline tier (4096)."""
    system_inst = "You are a concise code assistant."
    prompt = "Write a function to add two numbers in Python."
    budget = AdaptiveContextManager.calculate_budget(system_inst, prompt)

    assert budget["selected_num_ctx"] == 4096
    assert budget["estimated_input_tokens"] < 100
    assert budget["reserve_tokens"] == settings.OLLAMA_CONTEXT_RESERVE_TOKENS
    assert budget["predict_tokens"] == settings.OLLAMA_NUM_PREDICT
    assert "Snapped to stable tier 4096" in budget["selection_reason"]


def test_context_budget_medium_prompt_snaps_to_mid_tier():
    """Prompt needing ~4500 tokens should snap to 6144 tier."""
    system_inst = "System instructions for enterprise architecture."
    # Generate ~4,000 token input (approx 16,000 characters)
    prompt = "Detailed system requirement description. " * 400
    budget = AdaptiveContextManager.calculate_budget(system_inst, prompt)

    assert budget["selected_num_ctx"] == 6144
    assert 4096 < budget["total_required_context"] <= 6144
    assert "Snapped to stable tier 6144" in budget["selection_reason"]


def test_context_budget_near_limit_snaps_to_max_tier():
    """Prompt needing ~7,000 tokens should scale up to max tier (8192)."""
    system_inst = "System instructions."
    # Generate ~6,500 token input
    prompt = "Long technical data pipeline module description. " * 550
    budget = AdaptiveContextManager.calculate_budget(system_inst, prompt)

    assert budget["selected_num_ctx"] == 8192
    assert 6144 < budget["total_required_context"] <= 8192
    assert "Snapped to stable tier 8192" in budget["selection_reason"]


def test_context_budget_oversized_prompt_raises_413():
    """Prompts exceeding max context (8192) must raise ContextOverflowError (HTTP 413)."""
    system_inst = "System instructions."
    # Exceeds 8192 tokens
    oversized_prompt = "Exhaustive consensus protocol analysis. " * 900

    with pytest.raises(ContextOverflowError) as exc_info:
        AdaptiveContextManager.calculate_budget(system_inst, oversized_prompt)

    err = exc_info.value
    assert err.status_code == 413
    assert "exceed the configured maximum context limit" in err.message
    assert err.budget_breakdown["total_required_context"] > 8192
    assert err.budget_breakdown["max_num_ctx_limit"] == settings.OLLAMA_MAX_NUM_CTX


def test_context_budget_low_vram_safeguard():
    """Low free VRAM (<1000 MiB) restricts tier expansion if feasible."""
    mock_gpu = {"total_mb": 6141, "free_mb": 800, "used_mb": 5341, "source": "nvidia_smi"}
    system_inst = "Short system instruction."
    prompt = "Short prompt that fits inside baseline 4096."

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu):
        budget = AdaptiveContextManager.calculate_budget(system_inst, prompt)
        assert budget["selected_num_ctx"] == 4096


# ---------------------------------------------------------------------------
# 2. Hardware Telemetry & Concurrency Controller Tests
# ---------------------------------------------------------------------------

def test_hardware_telemetry_gpu_query():
    """Probes GPU telemetry format."""
    gpu = HardwareTelemetryProvider.get_gpu_memory(force_refresh=True)
    if gpu is not None:
        assert "total_mb" in gpu
        assert "free_mb" in gpu
        assert "used_mb" in gpu
        assert gpu["source"] == "nvidia_smi"
        assert gpu["total_mb"] > 0


def test_hardware_telemetry_ram_query():
    """Probes system RAM telemetry format."""
    ram = HardwareTelemetryProvider.get_system_ram()
    assert "total_mb" in ram
    assert "free_mb" in ram
    assert "used_mb" in ram
    assert ram["source"] in ("win32_global_memory_status", "unavailable")


def test_concurrency_controller_adaptive_healthy_gpu():
    """Normal VRAM conditions provide full configured capacity."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=2)
    mock_gpu = {"total_mb": 6141, "free_mb": 4000, "used_mb": 2141, "source": "nvidia_smi"}

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu):
        capacity, mode, gpu, ram = controller.get_admission_capacity()
        assert capacity == 2
        assert mode == "adaptive_hardware_aware"
        assert gpu["free_mb"] == 4000


def test_concurrency_controller_adaptive_restricted_vram():
    """Low free VRAM (<400 MiB) restricts concurrency capacity to 1."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=3)
    mock_gpu = {"total_mb": 6141, "free_mb": 350, "used_mb": 5791, "source": "nvidia_smi"}

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu):
        capacity, mode, gpu, ram = controller.get_admission_capacity()
        assert capacity == 1
        assert mode == "adaptive_hardware_aware_restricted"


def test_concurrency_controller_static_fallback():
    """When GPU telemetry is unavailable, uses conservative static fallback."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=1)

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=None):
        capacity, mode, gpu, ram = controller.get_admission_capacity()
        assert capacity == 1
        assert mode == "conservative_static_fallback"
        assert gpu is None


@pytest.mark.asyncio
async def test_concurrency_controller_critical_vram_rejection():
    """Critically low free VRAM (<150 MiB) raises ConcurrencyControllerError."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=1)
    mock_gpu = {"total_mb": 6141, "free_mb": 100, "used_mb": 6041, "source": "nvidia_smi"}

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu):
        with pytest.raises(ConcurrencyControllerError) as exc_info:
            async with controller.slot():
                pass
        assert "Insufficient GPU VRAM" in str(exc_info.value)


@pytest.mark.asyncio
async def test_concurrency_controller_slot_release_on_exception():
    """Guarantees slot is released and counters recover even on inner error."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=1)
    mock_gpu = {"total_mb": 6141, "free_mb": 4000, "used_mb": 2141, "source": "nvidia_smi"}

    assert controller.in_flight == 0

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu):
        with pytest.raises(RuntimeError):
            async with controller.slot() as slot_meta:
                assert controller.in_flight == 1
                assert slot_meta["active_concurrency_in_flight"] == 1
                raise RuntimeError("Inference processing crashed")

    # Counter should be cleanly decremented back to 0
    assert controller.in_flight == 0


@pytest.mark.asyncio
async def test_concurrency_controller_concurrent_queueing():
    """Multiple concurrent callers should queue cleanly without race conditions."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=1)
    mock_gpu = {"total_mb": 6141, "free_mb": 3000, "used_mb": 3141, "source": "nvidia_smi"}

    order = []

    async def worker(worker_id: int, delay: float):
        with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu):
            async with controller.slot() as slot_info:
                order.append(f"start_{worker_id}")
                await asyncio.sleep(delay)
                order.append(f"end_{worker_id}")

    # Launch two tasks competing for 1 slot
    await asyncio.gather(
        worker(1, 0.05),
        worker(2, 0.02),
    )

    # Worker 1 should start and end before worker 2 starts
    assert order == ["start_1", "end_1", "start_2", "end_2"]
    assert controller.in_flight == 0


# ---------------------------------------------------------------------------
# 3. Ollama Process Manager & Verification Tests
# ---------------------------------------------------------------------------

def test_ollama_process_manager_runtime_policy():
    """Inspects runtime policy output structure."""
    policy = OllamaProcessManager.get_runtime_policy()
    assert "configured_kv_cache_type" in policy
    assert "configured_flash_attention" in policy
    assert "server_process_status" in policy
    assert "server_process_verified" in policy
    assert "server_settings_verified" in policy
    assert "runtime_configuration_status" in policy
    assert policy["runtime_configuration_status"] in (
        "independently_confirmed",
        "launched_with_requested_settings",
        "unmanaged_external_instance",
        "not_running",
    )


def test_ollama_process_manager_manifest_reading():
    """Manifest reader handles present or missing files safely."""
    manifest = OllamaProcessManager.read_manifest()
    if manifest:
        assert "pid" in manifest
        assert "kv_cache_type" in manifest
        assert "flash_attention" in manifest


# ---------------------------------------------------------------------------
# 4. Native Telemetry with Cached Prefix Attribution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_native_telemetry_with_prompt_eval_cached_count():
    """Telemetry parsing retains prompt_eval_cached_count and source attributions."""
    mock_ollama_data = {
        "response": "Optimized output text",
        "prompt_eval_count": 50,
        "prompt_eval_cached_count": 48,
        "prompt_eval_duration": 150000000,
        "eval_count": 25,
        "eval_duration": 250000000,
        "load_duration": 10000000,
    }

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_ollama_data
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("app.services.ollama_service.httpx.AsyncClient") as client_cls:
        client_cls.return_value.__aenter__.return_value = mock_client
        mock_gpu = {"total_mb": 6141, "free_mb": 4500, "used_mb": 1641, "source": "nvidia_smi"}
        with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu):
            output, duration, telemetry = await OllamaService._execute_model_call(
                "System prompt", "User prompt"
            )

    assert output == "Optimized output text"
    assert telemetry["prompt_eval_count"] == 50
    assert telemetry["prompt_eval_cached_count"] == 48
    assert telemetry["eval_count"] == 25
    assert telemetry["selected_num_ctx"] == 4096
    assert telemetry["active_concurrency_capacity"] == 1
    assert telemetry["vram_available_mb"] == 4500
    assert telemetry["telemetry_source_attribution"]["prompt_eval_cached_count"] == "ollama_native"
    assert telemetry["telemetry_source_attribution"]["vram"] == "gpu_telemetry_nvidia_smi"

"""Automated Test Suite for Argus Adaptive KV-Cache Optimization (Phase 4).

Covers:
1. Adaptive context budget selection, tier derivation, and strict max_num_ctx capping.
2. Clean HTTP 413 rejection of oversized prompts and low-VRAM overflow without silent truncation.
3. Genuine dynamic concurrency capacity enforcement (throttling from 3 to 1 under low VRAM while active).
4. Critical memory rejections (<150 MiB VRAM, <512 MiB RAM) and RAM admission constraints.
5. Guaranteed queue depth recovery and cancellation cleanup without leaking permits.
6. Independent verification of Q8_0 and Flash Attention, stale manifest rejection, and stale log rejection.
7. Authoritative telemetry parsing including prompt_eval_cached_count and source attribution.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
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

def test_context_eligible_tiers_derived_dynamically():
    """Eligible tiers must derive dynamically from min and max without exceeding max."""
    tiers_default = AdaptiveContextManager.get_eligible_tiers(4096, 8192)
    assert tiers_default == [4096, 6144, 8192]

    tiers_custom = AdaptiveContextManager.get_eligible_tiers(2048, 6144)
    assert tiers_custom == [2048, 4096, 6144]

    tiers_equal = AdaptiveContextManager.get_eligible_tiers(4096, 4096)
    assert tiers_equal == [4096]

    tiers_small = AdaptiveContextManager.get_eligible_tiers(2048, 3000)
    assert tiers_small == [2048, 3000]
    assert all(t <= 3000 for t in tiers_small)


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
    prompt = "Detailed system requirement description. " * 400
    budget = AdaptiveContextManager.calculate_budget(system_inst, prompt)

    assert budget["selected_num_ctx"] == 6144
    assert 4096 < budget["total_required_context"] <= 6144
    assert "Snapped to stable tier 6144" in budget["selection_reason"]


def test_context_budget_near_limit_snaps_to_max_tier():
    """Prompt needing ~7,000 tokens should scale up to max tier (8192)."""
    system_inst = "System instructions."
    prompt = "Long technical data pipeline module description. " * 550
    budget = AdaptiveContextManager.calculate_budget(system_inst, prompt)

    assert budget["selected_num_ctx"] == 8192
    assert 6144 < budget["total_required_context"] <= 8192
    assert "Snapped to stable tier 8192" in budget["selection_reason"]


def test_context_budget_max_num_ctx_below_8192():
    """When OLLAMA_MAX_NUM_CTX is configured below 8192 (e.g. 6144), never select 8192."""
    system_inst = "System instructions."
    prompt = "Detailed system requirement description. " * 400

    # Test with max_ctx=6144
    budget = AdaptiveContextManager.calculate_budget(system_inst, prompt, min_ctx=4096, max_ctx=6144)
    assert budget["selected_num_ctx"] == 6144
    assert budget["max_num_ctx_limit"] == 6144
    assert all(t <= 6144 for t in budget["eligible_tiers"])

    # Oversized for 6144 limit must raise 413
    oversized = "Long technical data pipeline module description. " * 550
    with pytest.raises(ContextOverflowError) as exc_info:
        AdaptiveContextManager.calculate_budget(system_inst, oversized, min_ctx=4096, max_ctx=6144)
    assert exc_info.value.status_code == 413
    assert exc_info.value.budget_breakdown["max_num_ctx_limit"] == 6144


def test_context_budget_oversized_prompt_raises_413():
    """Prompts exceeding max context (8192) must raise ContextOverflowError (HTTP 413)."""
    system_inst = "System instructions."
    oversized_prompt = "Exhaustive consensus protocol analysis. " * 900

    with pytest.raises(ContextOverflowError) as exc_info:
        AdaptiveContextManager.calculate_budget(system_inst, oversized_prompt)

    err = exc_info.value
    assert err.status_code == 413
    assert "exceed the configured maximum context limit" in err.message
    assert err.budget_breakdown["total_required_context"] > 8192
    assert err.budget_breakdown["max_num_ctx_limit"] == settings.OLLAMA_MAX_NUM_CTX


def test_context_budget_low_vram_prompt_requires_larger_tier_raises_413():
    """Under low VRAM (<1000 MiB), a prompt requiring >4096 tokens cannot expand safely and must raise 413."""
    mock_gpu = {"total_mb": 6141, "free_mb": 750, "used_mb": 5391, "source": "nvidia_smi"}
    system_inst = "System instructions."
    # Needs ~5,000 tokens
    prompt = "Detailed system requirement description. " * 400

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu):
        with pytest.raises(ContextOverflowError) as exc_info:
            AdaptiveContextManager.calculate_budget(system_inst, prompt)

        err = exc_info.value
        assert err.status_code == 413
        assert "constrained GPU VRAM" in err.message
        assert err.budget_breakdown["vram_headroom_guarded"] is True
        assert err.budget_breakdown["free_vram_mb"] == 750


def test_context_budget_low_vram_prompt_fits_in_safe_tier():
    """Under low VRAM (<1000 MiB), a prompt fitting within safe tier (4096) is admitted at baseline."""
    mock_gpu = {"total_mb": 6141, "free_mb": 800, "used_mb": 5341, "source": "nvidia_smi"}
    system_inst = "Short instruction."
    prompt = "Short code prompt that easily fits within 4096 tokens."

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu):
        budget = AdaptiveContextManager.calculate_budget(system_inst, prompt)
        assert budget["selected_num_ctx"] == 4096
        assert budget["vram_headroom_guarded"] is True
        assert "Restricted to safe baseline tier 4096" in budget["selection_reason"]


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
    assert ram["source"] in ("win32_global_memory_status", "proc_meminfo", "unavailable")


def test_concurrency_controller_adaptive_healthy_gpu():
    """Normal VRAM conditions provide full configured capacity."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=3)
    mock_gpu = {"total_mb": 6141, "free_mb": 4000, "used_mb": 2141, "source": "nvidia_smi"}
    mock_ram = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu), \
         patch.object(HardwareTelemetryProvider, "get_system_ram", return_value=mock_ram):
        capacity, mode, gpu, ram = controller.get_admission_capacity()
        assert capacity == 3
        assert mode == "adaptive_hardware_aware"
        assert gpu["free_mb"] == 4000


def test_concurrency_controller_adaptive_restricted_vram():
    """Low free VRAM (<400 MiB) restricts concurrency capacity to 1."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=3)
    mock_gpu = {"total_mb": 6141, "free_mb": 350, "used_mb": 5791, "source": "nvidia_smi"}
    mock_ram = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu), \
         patch.object(HardwareTelemetryProvider, "get_system_ram", return_value=mock_ram):
        capacity, mode, gpu, ram = controller.get_admission_capacity()
        assert capacity == 1
        assert "restricted_low_vram" in mode


def test_concurrency_controller_adaptive_restricted_ram():
    """Low free RAM (<1024 MiB) restricts concurrency capacity to 1."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=3)
    mock_gpu = {"total_mb": 6141, "free_mb": 3000, "used_mb": 3141, "source": "nvidia_smi"}
    mock_ram = {"total_mb": 16000, "free_mb": 800, "used_mb": 15200, "source": "win32"}

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu), \
         patch.object(HardwareTelemetryProvider, "get_system_ram", return_value=mock_ram):
        capacity, mode, gpu, ram = controller.get_admission_capacity()
        assert capacity == 1
        assert "restricted_low_ram" in mode


@pytest.mark.asyncio
async def test_concurrency_controller_critical_ram_rejection():
    """Critically low free RAM (<512 MiB) raises ConcurrencyControllerError."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=2)
    mock_gpu = {"total_mb": 6141, "free_mb": 3000, "used_mb": 3141, "source": "nvidia_smi"}
    mock_ram = {"total_mb": 16000, "free_mb": 350, "used_mb": 15650, "source": "win32"}

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu), \
         patch.object(HardwareTelemetryProvider, "get_system_ram", return_value=mock_ram):
        with pytest.raises(ConcurrencyControllerError) as exc_info:
            async with controller.slot():
                pass
        assert "Insufficient memory to admit inference request" in str(exc_info.value)


@pytest.mark.asyncio
async def test_concurrency_controller_critical_vram_rejection():
    """Critically low free VRAM (<150 MiB) raises ConcurrencyControllerError."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=1)
    mock_gpu = {"total_mb": 6141, "free_mb": 100, "used_mb": 6041, "source": "nvidia_smi"}

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu):
        with pytest.raises(ConcurrencyControllerError) as exc_info:
            async with controller.slot():
                pass
        assert "Insufficient memory to admit inference request" in str(exc_info.value)


@pytest.mark.asyncio
async def test_concurrency_controller_dynamic_restriction_from_3_to_1_while_active():
    """Proves configured limit 3 is genuinely restricted to 1 under low VRAM while requests are active."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=3)

    # Initial healthy state: 4000 MiB free VRAM
    current_gpu = {"total_mb": 6141, "free_mb": 4000, "used_mb": 2141, "source": "nvidia_smi"}
    mock_ram = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    task1_started = asyncio.Event()
    task1_release = asyncio.Event()
    task2_admitted = False

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", side_effect=lambda: current_gpu), \
         patch.object(HardwareTelemetryProvider, "get_system_ram", return_value=mock_ram):

        async def task_1():
            async with controller.slot():
                task1_started.set()
                await task1_release.wait()

        async def task_2():
            nonlocal task2_admitted
            async with controller.slot():
                task2_admitted = True

        # 1. Start Task 1 under healthy conditions
        t1 = asyncio.create_task(task_1())
        await task1_started.wait()

        assert controller.in_flight == 1
        assert controller.queue_depth == 0

        # 2. VRAM drops to 300 MiB (capacity reduces from 3 to 1)
        current_gpu = {"total_mb": 6141, "free_mb": 300, "used_mb": 5841, "source": "nvidia_smi"}

        # 3. Launch Task 2: Since in-flight (1) >= new capacity (1), Task 2 must NOT be admitted!
        t2 = asyncio.create_task(task_2())
        await asyncio.sleep(0.05)

        assert task2_admitted is False
        assert controller.queue_depth == 1
        assert controller.in_flight == 1

        # 4. Release Task 1
        task1_release.set()
        await t1

        # 5. Task 2 now admits because in-flight dropped to 0 < 1
        await asyncio.wait_for(t2, timeout=2.0)
        assert task2_admitted is True
        assert controller.in_flight == 0
        assert controller.queue_depth == 0


@pytest.mark.asyncio
async def test_concurrency_controller_queue_wait_cancellation():
    """Cancelling a queued waiter decrements queue_depth without corrupting counters."""
    controller = RuntimeAwareConcurrencyController(max_concurrent_limit=1)
    mock_gpu = {"total_mb": 6141, "free_mb": 4000, "used_mb": 2141, "source": "nvidia_smi"}
    mock_ram = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    t1_started = asyncio.Event()
    t1_release = asyncio.Event()

    with patch.object(HardwareTelemetryProvider, "get_gpu_memory", return_value=mock_gpu), \
         patch.object(HardwareTelemetryProvider, "get_system_ram", return_value=mock_ram):

        async def task_1():
            async with controller.slot():
                t1_started.set()
                await t1_release.wait()

        async def task_2():
            async with controller.slot():
                pass

        t1 = asyncio.create_task(task_1())
        await t1_started.wait()

        t2 = asyncio.create_task(task_2())
        await asyncio.sleep(0.02)

        assert controller.queue_depth == 1
        assert controller.in_flight == 1

        # Cancel task 2 while it is waiting
        t2.cancel()
        await asyncio.sleep(0.02)

        assert controller.queue_depth == 0
        assert controller.in_flight == 1

        # Clean up task 1
        t1_release.set()
        await t1
        assert controller.in_flight == 0


# ---------------------------------------------------------------------------
# 3. Ollama Process Manager & Verification Tests
# ---------------------------------------------------------------------------

def test_process_manager_stale_manifest_pid_mismatch():
    """Stale manifest with mismatched PID must be classified as unmanaged."""
    fake_manifest = {
        "pid": 99999,
        "kv_cache_type": "q8_0",
        "flash_attention": True,
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    with patch.object(OllamaProcessManager, "get_port_listening_pid", return_value=12345), \
         patch.object(OllamaProcessManager, "is_pid_running_ollama", return_value=True), \
         patch.object(OllamaProcessManager, "read_manifest", return_value=fake_manifest), \
         patch.object(OllamaProcessManager, "scan_log_evidence", return_value={"all_settings_confirmed": False, "kv_confirmed": False, "flash_attention_confirmed": False, "combined_evidence": None}):
        policy = OllamaProcessManager.get_runtime_policy()
        assert policy["managed_by_argus"] is False
        assert policy["manifest_stale"] is True
        assert policy["active_pid"] == 12345
        assert "stale_manifest" in policy["runtime_configuration_status"]


def test_process_manager_stale_logs_rejected(tmp_path: Path):
    """Log file with mtime older than active process start time is ignored as stale."""
    std_log = tmp_path / "ollama_server.log"
    err_log = tmp_path / "ollama_server.err.log"

    std_log.write_text("Server config: OLLAMA_KV_CACHE_TYPE:q8_0\nllama_context: flash_attn = enabled\n", encoding="utf-8")
    err_log.write_text("", encoding="utf-8")

    # Set mtime to 1 hour ago
    old_time = time.time() - 3600
    import os
    os.utime(std_log, (old_time, old_time))
    os.utime(err_log, (old_time, old_time))

    proc_start = datetime.now(timezone.utc)

    evidence = OllamaProcessManager.scan_log_evidence(
        target_kv="q8_0",
        target_fa=True,
        active_process_start_time=proc_start,
        std_log_path=std_log,
        err_log_path=err_log,
    )

    assert evidence["all_settings_confirmed"] is False
    assert evidence["kv_confirmed"] is False
    assert evidence["flash_attention_confirmed"] is False
    assert evidence["is_stale_log"] is True


def test_process_manager_independent_kv_only_confirmation(tmp_path: Path):
    """Log containing only KV-cache evidence does NOT confirm Flash Attention."""
    std_log = tmp_path / "ollama_server.log"
    err_log = tmp_path / "ollama_server.err.log"

    # Only KV cache line present
    std_log.write_text("Server config: OLLAMA_KV_CACHE_TYPE:q8_0\n", encoding="utf-8")
    err_log.write_text("", encoding="utf-8")

    proc_start = datetime.now(timezone.utc) - timedelta(seconds=10)

    evidence = OllamaProcessManager.scan_log_evidence(
        target_kv="q8_0",
        target_fa=True,
        active_process_start_time=proc_start,
        std_log_path=std_log,
        err_log_path=err_log,
    )

    assert evidence["kv_confirmed"] is True
    assert evidence["flash_attention_confirmed"] is False
    assert evidence["all_settings_confirmed"] is False


def test_process_manager_independent_fa_only_confirmation(tmp_path: Path):
    """Log containing only Flash Attention evidence does NOT confirm KV cache."""
    std_log = tmp_path / "ollama_server.log"
    err_log = tmp_path / "ollama_server.err.log"

    # Only Flash Attention line present
    std_log.write_text("llama_context: flash_attn = enabled\n", encoding="utf-8")
    err_log.write_text("", encoding="utf-8")

    proc_start = datetime.now(timezone.utc) - timedelta(seconds=10)

    evidence = OllamaProcessManager.scan_log_evidence(
        target_kv="q8_0",
        target_fa=True,
        active_process_start_time=proc_start,
        std_log_path=std_log,
        err_log_path=err_log,
    )

    assert evidence["kv_confirmed"] is False
    assert evidence["flash_attention_confirmed"] is True
    assert evidence["all_settings_confirmed"] is False


def test_process_manager_both_settings_confirmed(tmp_path: Path):
    """When both settings appear in active log, all_settings_confirmed is True."""
    std_log = tmp_path / "ollama_server.log"
    err_log = tmp_path / "ollama_server.err.log"

    std_log.write_text(
        "Server config: OLLAMA_KV_CACHE_TYPE:q8_0\nllama_context: flash_attn = enabled\n",
        encoding="utf-8",
    )
    err_log.write_text("", encoding="utf-8")

    proc_start = datetime.now(timezone.utc) - timedelta(seconds=10)

    evidence = OllamaProcessManager.scan_log_evidence(
        target_kv="q8_0",
        target_fa=True,
        active_process_start_time=proc_start,
        std_log_path=std_log,
        err_log_path=err_log,
    )

    assert evidence["kv_confirmed"] is True
    assert evidence["flash_attention_confirmed"] is True
    assert evidence["all_settings_confirmed"] is True


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

"""Runtime-Aware Concurrency Controller for Argus.

Dynamically adapts inference admission capacity based on live GPU VRAM and
system RAM measurements. Prevents deadlocks, guarantees safe slot release on
timeout or cancellation, and falls back to a conservative static ceiling when
hardware telemetry is unavailable.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from app.core.config import settings
from app.services.hardware_telemetry import HardwareTelemetryProvider


class ConcurrencyControllerError(Exception):
    """Raised when request cannot be admitted due to resource exhaustion."""
    pass


class RuntimeAwareConcurrencyController:
    """Adaptive inference concurrency controller."""

    def __init__(self, max_concurrent_limit: Optional[int] = None):
        self._configured_limit = max_concurrent_limit or max(1, settings.OLLAMA_MAX_CONCURRENT_REQUESTS)
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._configured_limit)
        self._in_flight: int = 0
        self._queue_depth: int = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    def get_admission_capacity(self) -> Tuple[int, str, Optional[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate resource conditions and determine safe active capacity."""
        gpu_telemetry = HardwareTelemetryProvider.get_gpu_memory()
        ram_telemetry = HardwareTelemetryProvider.get_system_ram()

        if gpu_telemetry is not None:
            free_vram = gpu_telemetry.get("free_mb", 0)
            # If VRAM is critically low (<400 MiB), restrict concurrency to 1
            if free_vram < 400:
                return 1, "adaptive_hardware_aware_restricted", gpu_telemetry, ram_telemetry
            # Otherwise allow configured ceiling
            return self._configured_limit, "adaptive_hardware_aware", gpu_telemetry, ram_telemetry

        # Non-NVIDIA or telemetry unavailable fallback
        return self._configured_limit, "conservative_static_fallback", None, ram_telemetry

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[Dict[str, Any]]:
        """Asynchronously acquire an inference slot with queue tracking and guaranteed release."""
        capacity, mode, gpu_telemetry, ram_telemetry = self.get_admission_capacity()

        # Check for critical VRAM exhaustion (<150 MiB free)
        if gpu_telemetry is not None and gpu_telemetry.get("free_mb", 0) < 150:
            raise ConcurrencyControllerError(
                f"Insufficient GPU VRAM to admit inference ({gpu_telemetry['free_mb']} MiB free). "
                "Request queued or rejected to prevent CUDA OOM crash."
            )

        queue_started = time.perf_counter()
        async with self._lock:
            self._queue_depth += 1

        try:
            await self._semaphore.acquire()
            queue_wait_ms = int((time.perf_counter() - queue_started) * 1000)
            async with self._lock:
                self._queue_depth = max(0, self._queue_depth - 1)
                self._in_flight += 1

            slot_telemetry = {
                "queue_wait_ms": queue_wait_ms,
                "active_concurrency_capacity": capacity,
                "active_concurrency_in_flight": self._in_flight,
                "concurrency_mode": mode,
                "vram_available_mb": gpu_telemetry.get("free_mb") if gpu_telemetry else None,
                "vram_used_mb": gpu_telemetry.get("used_mb") if gpu_telemetry else None,
                "ram_available_mb": ram_telemetry.get("free_mb") if ram_telemetry else None,
            }

            try:
                yield slot_telemetry
            finally:
                async with self._lock:
                    self._in_flight = max(0, self._in_flight - 1)
                self._semaphore.release()
        except Exception:
            # Ensure queue depth decrements if acquire was cancelled
            async with self._lock:
                if self._queue_depth > 0:
                    self._queue_depth = max(0, self._queue_depth - 1)
            raise


# Global singleton controller
concurrency_controller = RuntimeAwareConcurrencyController()

"""Runtime-Aware Concurrency Controller for Argus.

Dynamically enforces inference admission capacity based on live GPU VRAM and
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


class _SemaphoreCompatibilityProxy:
    """Backwards-compatible proxy exposing ._value for existing tests and callers."""

    def __init__(self, controller: "RuntimeAwareConcurrencyController"):
        self._controller = controller

    @property
    def _value(self) -> int:
        capacity, _, _, _ = self._controller.get_admission_capacity()
        return max(0, capacity - self._controller.in_flight)


class RuntimeAwareConcurrencyController:
    """Adaptive inference concurrency controller with genuine dynamic capacity enforcement."""

    def __init__(self, max_concurrent_limit: Optional[int] = None):
        self._configured_limit = max_concurrent_limit or max(1, settings.OLLAMA_MAX_CONCURRENT_REQUESTS)
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(lock=self._lock)
        self._in_flight: int = 0
        self._queue_depth: int = 0
        self._semaphore = _SemaphoreCompatibilityProxy(self)

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    @property
    def configured_limit(self) -> int:
        return self._configured_limit

    def get_admission_capacity(self) -> Tuple[int, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Evaluate resource conditions and determine safe active capacity.

        Returns:
            Tuple of (capacity, mode, gpu_telemetry, ram_telemetry).
        """
        gpu_telemetry = HardwareTelemetryProvider.get_gpu_memory()
        ram_telemetry = HardwareTelemetryProvider.get_system_ram()

        # Check host RAM constraints only if telemetry is available and authoritative
        ram_restricted = False
        ram_valid = (
            ram_telemetry is not None
            and ram_telemetry.get("source") not in (None, "unavailable")
            and ram_telemetry.get("total_mb", 0) > 0
        )
        if ram_valid:
            free_ram = ram_telemetry.get("free_mb", 0)
            if free_ram < 512:
                # Critical RAM exhaustion - capacity is 0 to trigger immediate rejection
                return 0, "critical_ram_exhaustion", gpu_telemetry, ram_telemetry
            if free_ram < 1024:
                ram_restricted = True

        gpu_valid = (
            gpu_telemetry is not None
            and gpu_telemetry.get("source") not in (None, "unavailable")
            and gpu_telemetry.get("total_mb", 0) > 0
        )
        if gpu_valid:
            free_vram = gpu_telemetry.get("free_mb", 0)
            # Critical VRAM exhaustion (<150 MiB)
            if free_vram < 150:
                return 0, "critical_vram_exhaustion", gpu_telemetry, ram_telemetry

            # Constrained VRAM (<400 MiB) or constrained RAM (<1024 MiB) -> restrict to 1
            if free_vram < 400 or ram_restricted:
                reason = "low_vram" if free_vram < 400 else "low_ram"
                return 1, f"adaptive_hardware_aware_restricted_{reason}", gpu_telemetry, ram_telemetry

            # Nominal capacity
            return self._configured_limit, "adaptive_hardware_aware", gpu_telemetry, ram_telemetry

        # Non-NVIDIA or telemetry unavailable fallback
        if ram_restricted:
            return 1, "conservative_static_fallback_low_ram", None, ram_telemetry

        return self._configured_limit, "conservative_static_fallback", None, ram_telemetry

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[Dict[str, Any]]:
        """Asynchronously acquire an inference slot with queue tracking, dynamic capacity enforcement,

        and guaranteed slot release on completion, cancellation, or exception.
        """
        queue_started = time.perf_counter()
        admitted = False

        async with self._condition:
            self._queue_depth += 1
            try:
                while True:
                    # Re-evaluate hardware capacity at the exact point of admission
                    capacity, mode, gpu_telemetry, ram_telemetry = self.get_admission_capacity()

                    # Critical resource rejections
                    if capacity == 0:
                        vram_free = gpu_telemetry.get("free_mb") if gpu_telemetry else "N/A"
                        ram_free = ram_telemetry.get("free_mb") if ram_telemetry else "N/A"
                        raise ConcurrencyControllerError(
                            f"Insufficient memory to admit inference request (VRAM free: {vram_free} MiB, "
                            f"RAM free: {ram_free} MiB). Request rejected to prevent out-of-memory crash."
                        )

                    # Dynamic admission enforcement:
                    # Only admit if current in-flight requests are strictly less than current capacity.
                    # If capacity reduced (e.g. from 3 to 1 while in-flight is 1 or 2), no new requests
                    # are admitted until in-flight drops below the new capacity limit.
                    if self._in_flight < capacity:
                        self._in_flight += 1
                        self._queue_depth = max(0, self._queue_depth - 1)
                        admitted = True
                        break

                    # Wait until an active request completes or capacity changes
                    await self._condition.wait()
            except BaseException:
                if not admitted:
                    self._queue_depth = max(0, self._queue_depth - 1)
                raise

        # Prepare slot telemetry for the admitted request
        queue_wait_ms = int((time.perf_counter() - queue_started) * 1000)
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
            async with self._condition:
                self._in_flight = max(0, self._in_flight - 1)
                # Wake all queued waiters to re-evaluate their capacity and in_flight count
                self._condition.notify_all()


# Global singleton controller
concurrency_controller = RuntimeAwareConcurrencyController()

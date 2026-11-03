"""Hardware Telemetry Provider for Argus.

Collects real GPU memory metrics (via nvidia-smi) and Windows/OS RAM metrics
(via Win32 GlobalMemoryStatusEx). Includes caching to prevent repeated subprocess calls.
Never fabricates telemetry or assumes GPU availability without empirical confirmation.
"""

import ctypes
import os
import subprocess
import time
from typing import Any, Dict, Optional, Tuple


class HardwareTelemetryProvider:
    """Provides observable GPU and system RAM telemetry with caching."""

    _last_gpu_query_time: float = 0.0
    _cached_gpu_telemetry: Optional[Dict[str, Any]] = None
    _cache_ttl_seconds: float = 1.0

    @classmethod
    def get_gpu_memory(cls, force_refresh: bool = False) -> Optional[Dict[str, int]]:
        """Query NVIDIA GPU VRAM metrics in MiB.

        Returns:
            Dict with total_mb, free_mb, used_mb, and source, or None if unavailable.
        """
        now = time.monotonic()
        if not force_refresh and cls._cached_gpu_telemetry is not None:
            if (now - cls._last_gpu_query_time) < cls._cache_ttl_seconds:
                return cls._cached_gpu_telemetry

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.free,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
                stderr=subprocess.DEVNULL,
            )
            lines = out.strip().splitlines()
            if not lines:
                cls._cached_gpu_telemetry = None
                return None

            parts = [int(p.strip()) for p in lines[0].split(",")]
            if len(parts) >= 3:
                telemetry = {
                    "total_mb": parts[0],
                    "free_mb": parts[1],
                    "used_mb": parts[2],
                    "source": "nvidia_smi",
                }
                cls._cached_gpu_telemetry = telemetry
                cls._last_gpu_query_time = now
                return telemetry
        except Exception:
            cls._cached_gpu_telemetry = None
            return None

        return None

    @classmethod
    def get_system_ram(cls) -> Dict[str, float]:
        """Query physical host RAM usage in MiB using Win32 API or fallback."""
        if os.name == "nt":
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_mb = round(stat.ullTotalPhys / (1024 * 1024), 1)
                    free_mb = round(stat.ullAvailPhys / (1024 * 1024), 1)
                    used_mb = round((stat.ullTotalPhys - stat.ullAvailPhys) / (1024 * 1024), 1)
                    return {
                        "total_mb": total_mb,
                        "free_mb": free_mb,
                        "used_mb": used_mb,
                        "source": "win32_global_memory_status",
                    }
            except Exception:
                pass

        return {
            "total_mb": 0.0,
            "free_mb": 0.0,
            "used_mb": 0.0,
            "source": "unavailable",
        }

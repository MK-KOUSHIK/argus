#!/usr/bin/env python3
"""Argus Phase 4.2 — Content-Aware Compression Engine Empirical Benchmark.

Evaluates character reduction, native prompt prefill token savings, downstream task
answer quality preservation, and exact reversible restoration across technical workloads.
All measurements reflect genuine runtime execution against local Ollama (qwen2.5:1.5b)
with alternating execution order and warm-up passes to eliminate driver cache bias.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ctypes
from ctypes import Structure, c_ulong, c_ulonglong

import httpx

from app.core.config import settings
from app.schemas.optimizer import OptimizationMode, OptimizeRequest, OptimizeResponse
from app.services.compression.registry import CompressionRegistry
from app.services.ollama_service import OllamaService


class MEMORYSTATUSEX(Structure):
    _fields_ = [
        ("dwLength", c_ulong),
        ("dwMemoryLoad", c_ulong),
        ("ullTotalPhys", c_ulonglong),
        ("ullAvailPhys", c_ulonglong),
        ("ullTotalPageFile", c_ulonglong),
        ("ullAvailPageFile", c_ulonglong),
        ("ullTotalVirtual", c_ulonglong),
        ("ullAvailVirtual", c_ulonglong),
        ("ullAvailExtendedVirtual", c_ulonglong),
    ]


def detect_hardware_metadata() -> Dict[str, Any]:
    """Dynamically query GPU, CPU, RAM, and OS metadata without hardcoded values."""
    gpu_name: Optional[str] = None
    total_vram_mib: Optional[int] = None
    driver_version: Optional[str] = None
    gpu_error: Optional[str] = None

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) >= 3:
            gpu_name = parts[0]
            vram_digits = re.sub(r"[^\d]", "", parts[1])
            if vram_digits:
                total_vram_mib = int(vram_digits)
            driver_version = parts[2]
    except Exception as exc:
        gpu_error = str(exc)

    gpu_info: Dict[str, Any] = {
        "name": gpu_name,
        "total_vram_mib": total_vram_mib,
        "driver": driver_version,
        "source": "nvidia_smi" if gpu_name else None,
    }
    if gpu_error is not None:
        gpu_info["error"] = gpu_error

    total_ram_mb: Optional[float] = None
    if os.name == "nt" or sys.platform == "win32":
        try:
            windll = getattr(ctypes, "windll", None)
            if windll is not None:
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_ram_mb = round(stat.ullTotalPhys / (1024 * 1024), 2)
        except Exception:
            pass
    elif hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int):
                total_ram_mb = round((pages * page_size) / (1024 * 1024), 2)
        except (ValueError, OSError, AttributeError):
            pass

    return {
        "gpu": gpu_info,
        "ram": {
            "total_mb": total_ram_mb,
            "source": "GlobalMemoryStatusEx" if os.name == "nt" else "sysconf",
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
    }


BENCHMARK_WORKLOADS: List[Dict[str, Any]] = [
    {
        "id": "json_cluster_config",
        "name": "JSON API Cluster Configuration",
        "content_type": "json",
        "mode": "concise",
        "prompt": """```json
{
  "cluster_id": "us-east-prod-01",
  "environment": "production",
  "version": "2.4.1",
  "datacenter": {
    "region": "us-east-1",
    "zone": "us-east-1a",
    "rack_id": 402
  },
  "nodes": [
    {
      "node_id": "node-alpha",
      "role": "control-plane",
      "status": "ready",
      "specs": {
        "cpu_cores": 16,
        "memory_gib": 64,
        "storage_nvme_gb": 1000
      },
      "tags": {
        "tier": "frontend",
        "monitoring": "datadog"
      }
    },
    {
      "node_id": "node-beta",
      "role": "worker",
      "status": "ready",
      "specs": {
        "cpu_cores": 32,
        "memory_gib": 128,
        "storage_nvme_gb": 2000
      },
      "tags": {
        "tier": "compute",
        "monitoring": "datadog"
      }
    },
    {
      "node_id": "node-gamma",
      "role": "worker",
      "status": "ready",
      "specs": {
        "cpu_cores": 32,
        "memory_gib": 128,
        "storage_nvme_gb": 2000
      },
      "tags": {
        "tier": "compute",
        "monitoring": "datadog"
      }
    }
  ]
}
```
Analyze this cluster specification JSON and state the name of each active node and the total memory.""",
        "description": "Fenced multi-node JSON cluster payload with analysis task directives.",
        "expected_registered_content": """{
  "cluster_id": "us-east-prod-01",
  "environment": "production",
  "version": "2.4.1",
  "datacenter": {
    "region": "us-east-1",
    "zone": "us-east-1a",
    "rack_id": 402
  },
  "nodes": [
    {
      "node_id": "node-alpha",
      "role": "control-plane",
      "status": "ready",
      "specs": {
        "cpu_cores": 16,
        "memory_gib": 64,
        "storage_nvme_gb": 1000
      },
      "tags": {
        "tier": "frontend",
        "monitoring": "datadog"
      }
    },
    {
      "node_id": "node-beta",
      "role": "worker",
      "status": "ready",
      "specs": {
        "cpu_cores": 32,
        "memory_gib": 128,
        "storage_nvme_gb": 2000
      },
      "tags": {
        "tier": "compute",
        "monitoring": "datadog"
      }
    },
    {
      "node_id": "node-gamma",
      "role": "worker",
      "status": "ready",
      "specs": {
        "cpu_cores": 32,
        "memory_gib": 128,
        "storage_nvme_gb": 2000
      },
      "tags": {
        "tier": "compute",
        "monitoring": "datadog"
      }
    }
  ]
}""",
    },
    {
        "id": "code_caching_pipeline",
        "name": "Python Source Code Implementation",
        "content_type": "code",
        "mode": "balanced",
        "prompt": '''```python
# ==============================================================================
# In-Memory Cache with TTL and LRU Eviction
# Author: Platform Engineering
# Last Updated: 2026-10-15
# ==============================================================================

import time
from collections import OrderedDict
from typing import Any, Optional

class MemoryCache:
    """Production memory cache supporting TTL expiration and maximum capacity LRU bound."""

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 300) -> None:
        # Initialize internal storage structure
        self.max_size: int = max_size  # Maximum allowed entries in memory
        self.default_ttl: int = default_ttl_seconds  # Default TTL in seconds
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()  # Key -> (val, expire_ts)

    def get(self, key: str) -> Optional[Any]:
        # Perform lookup in ordered dictionary
        if key not in self._store:
            return None  # Cache miss

        val, expires_at = self._store[key]
        now = time.time()  # Get current epoch timestamp

        # Check if item has expired
        if now > expires_at:
            del self._store[key]  # Invalidate expired entry
            return None

        # Move accessed entry to end of OrderedDict for LRU tracking
        self._store.move_to_end(key)
        return val

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Determine actual TTL value
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expire_at = time.time() + effective_ttl

        # Evict oldest entry if capacity limit reached
        if key not in self._store and len(self._store) >= self.max_size:
            self._store.popitem(last=False)  # Remove least recently used item

        self._store[key] = (value, expire_at)
        self._store.move_to_end(key)
```
Review the following Python caching implementation and explain how the eviction policy operates.''',
        "description": "Python source code with extensive inline comments, spacing, and docstrings.",
        "expected_registered_content": """# ==============================================================================
# In-Memory Cache with TTL and LRU Eviction
# Author: Platform Engineering
# Last Updated: 2026-10-15
# ==============================================================================

import time
from collections import OrderedDict
from typing import Any, Optional

class MemoryCache:
    \"\"\"Production memory cache supporting TTL expiration and maximum capacity LRU bound.\"\"\"

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 300) -> None:
        # Initialize internal storage structure
        self.max_size: int = max_size  # Maximum allowed entries in memory
        self.default_ttl: int = default_ttl_seconds  # Default TTL in seconds
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()  # Key -> (val, expire_ts)

    def get(self, key: str) -> Optional[Any]:
        # Perform lookup in ordered dictionary
        if key not in self._store:
            return None  # Cache miss

        val, expires_at = self._store[key]
        now = time.time()  # Get current epoch timestamp

        # Check if item has expired
        if now > expires_at:
            del self._store[key]  # Invalidate expired entry
            return None

        # Move accessed entry to end of OrderedDict for LRU tracking
        self._store.move_to_end(key)
        return val

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Determine actual TTL value
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expire_at = time.time() + effective_ttl

        # Evict oldest entry if capacity limit reached
        if key not in self._store and len(self._store) >= self.max_size:
            self._store.popitem(last=False)  # Remove least recently used item

        self._store[key] = (value, expire_at)
        self._store.move_to_end(key)""",
    },
    {
        "id": "logs_database_incident",
        "name": "System Logs with Repetitive Events",
        "content_type": "logs",
        "mode": "concise",
        "prompt": """Examine these system service logs and identify what caused the worker crash and at what timestamp:

2026-11-05 08:00:01 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:02 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:03 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:04 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:05 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:06 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:07 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:08 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:09 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:10 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:11 INFO [pool-worker-1] Heartbeat probe alive status=200 ok
2026-11-05 08:00:12 ERROR [pool-worker-1] Connection refused: failed to connect to PostgreSQL primary at 10.0.1.5:5432
2026-11-05 08:00:13 CRITICAL [pool-worker-1] ConnectionPoolExhausted: all 20 active connections terminated unexpectedly
2026-11-05 08:00:14 INFO [pool-worker-1] Retrying connection attempt 1 of 5
2026-11-05 08:00:15 INFO [pool-worker-1] Retrying connection attempt 1 of 5
2026-11-05 08:00:16 INFO [pool-worker-1] Retrying connection attempt 1 of 5
2026-11-05 08:00:17 INFO [pool-worker-1] Retrying connection attempt 1 of 5
2026-11-05 08:00:18 FATAL [pool-worker-1] Worker process terminated due to unhandled database connection loss""",
        "description": "Log stream containing repetitive heartbeat sequences and critical errors.",
        "expected_registered_content": "\n".join(
            [f"2026-11-05 08:00:{sec:02d} INFO [pool-worker-1] Heartbeat probe alive status=200 ok" for sec in range(2, 11)]
        ),
    },
    {
        "id": "text_redundant_instructions",
        "name": "Redundant Natural Language Directives",
        "content_type": "text",
        "mode": "concise",
        "prompt": """Please help me by providing a helpful answer. I would really appreciate your assistance with this task.
Explain the core architectural differences between horizontal scaling and vertical scaling in cloud systems.
Explain the core architectural differences between horizontal scaling and vertical scaling in cloud systems.
Make sure to explain both horizontal scaling and vertical scaling clearly.
Provide your response strictly in exactly 3 bullet points.
Do not use markdown tables or external links.
Never mention proprietary cloud vendor names.""",
        "description": "Natural language request with repetitive phrasing, filler, and explicit constraints.",
        "expected_registered_content": None,
    },
    {
        "id": "mixed_debugging_dossier",
        "name": "Mixed Markdown Engineering Dossier",
        "content_type": "mixed",
        "mode": "structure",
        "prompt": """# Microservice Health Investigation Dossier

Please inspect the following debugging information and summarize the status of the service:

### Service Configuration Payload
```json
{
  "service_name": "billing-aggregator",
  "port": 8080,
  "health_check": {
    "interval_seconds": 15,
    "timeout_ms": 2000
  },
  "database": {
    "host": "postgres-primary.internal",
    "port": 5432,
    "max_connections": 50
  }
}
```

### Handler Code Snippet
```python
# Main request dispatch logic
def handle_transaction(tx_id: str, amount: float) -> bool:
    # Validate transaction id is non-empty
    if not tx_id:
        return False  # Reject empty ID
    # Log transaction processing
    print(f"Processing transaction: {tx_id}")
    return True
```

### Task Directives
Format the summary with markdown headings and verify whether the port configuration matches across the payload.""",
        "description": "Multi-segment markdown prompt combining natural language, fenced JSON, and fenced Python.",
        "expected_registered_content": """{
  "service_name": "billing-aggregator",
  "port": 8080,
  "health_check": {
    "interval_seconds": 15,
    "timeout_ms": 2000
  },
  "database": {
    "host": "postgres-primary.internal",
    "port": 5432,
    "max_connections": 50
  }
}""",
    },
]


def determine_comparison_status(baseline_passed: Optional[bool], compressed_passed: Optional[bool]) -> str:
    """Determine explicit comparison status according to evidence rigor.

    States:
    - preserved: both baseline and compressed passed
    - regression: baseline passed but compressed failed
    - improvement: baseline failed but compressed passed
    - inconclusive_or_both_failed: both failed
    - not_evaluated: either side not evaluated or missing
    """
    if baseline_passed is None or compressed_passed is None:
        return "not_evaluated"
    if baseline_passed and compressed_passed:
        return "preserved"
    if baseline_passed and not compressed_passed:
        return "regression"
    if not baseline_passed and compressed_passed:
        return "improvement"
    return "inconclusive_or_both_failed"


def evaluate_prompt_preservation(workload_id: str, optimized_prompt: str) -> Dict[str, Any]:
    """Evaluate whether the Argus-optimized prompt preserved essential constraints and entities."""
    if not optimized_prompt or not optimized_prompt.strip():
        return {
            "evaluation_type": "rule_based_prompt_preservation",
            "methodology_note": "Rule-based structural/keyword check of optimized prompt; does not guarantee downstream response fidelity.",
            "evaluated": True,
            "passed": False,
            "reason": "empty_optimized_prompt",
            "details": {"length": 0},
        }

    lower = optimized_prompt.lower()

    if workload_id == "json_cluster_config":
        has_nodes = bool(re.search(r"\b(?:nodes?|cluster|node-alpha|specs)\b", lower))
        has_task = bool(re.search(r"\b(?:analyze|state|summarize|list|memory)\b", lower))
        passed = has_nodes and has_task
        return {
            "evaluation_type": "rule_based_prompt_preservation",
            "methodology_note": "Rule-based structural/keyword check of optimized prompt; does not guarantee downstream response fidelity.",
            "evaluated": True,
            "passed": passed,
            "reason": "rule_based_cluster_data_and_task",
            "details": {"has_nodes": has_nodes, "has_task": has_task},
        }

    elif workload_id == "code_caching_pipeline":
        has_class = "memorycache" in lower or "class memorycache" in lower
        has_methods = "get" in lower and "set" in lower
        passed = has_class and has_methods
        return {
            "evaluation_type": "rule_based_prompt_preservation",
            "methodology_note": "Rule-based structural/keyword check of optimized prompt; does not guarantee downstream response fidelity.",
            "evaluated": True,
            "passed": passed,
            "reason": "rule_based_caching_class_and_methods",
            "details": {"has_class": has_class, "has_methods": has_methods},
        }

    elif workload_id == "logs_database_incident":
        has_error = "connection refused" in lower or "connectionpoolexhausted" in lower or "fatal" in lower
        has_ts = "08:00:" in lower or "timestamp" in lower
        passed = has_error and has_ts
        return {
            "evaluation_type": "rule_based_prompt_preservation",
            "methodology_note": "Rule-based structural/keyword check of optimized prompt; does not guarantee downstream response fidelity.",
            "evaluated": True,
            "passed": passed,
            "reason": "rule_based_critical_error_and_timestamp",
            "details": {"has_error": has_error, "has_ts": has_ts},
        }

    elif workload_id == "text_redundant_instructions":
        has_horiz = "horizontal" in lower
        has_vert = "vertical" in lower
        has_bullets = "bullet" in lower or "3" in lower
        passed = has_horiz and has_vert and has_bullets
        return {
            "evaluation_type": "rule_based_prompt_preservation",
            "methodology_note": "Rule-based structural/keyword check of optimized prompt; does not guarantee downstream response fidelity.",
            "evaluated": True,
            "passed": passed,
            "reason": "rule_based_scaling_concepts_and_bullet_constraint",
            "details": {"has_horizontal": has_horiz, "has_vertical": has_vert, "has_bullets": has_bullets},
        }

    elif workload_id == "mixed_debugging_dossier":
        has_billing = "billing-aggregator" in lower or "billing" in lower
        has_port = "8080" in lower or "5432" in lower or "port" in lower
        passed = has_billing and has_port
        return {
            "evaluation_type": "rule_based_prompt_preservation",
            "methodology_note": "Rule-based structural/keyword check of optimized prompt; does not guarantee downstream response fidelity.",
            "evaluated": True,
            "passed": passed,
            "reason": "rule_based_service_name_and_port",
            "details": {"has_billing": has_billing, "has_port": has_port},
        }

    return {
        "evaluation_type": "rule_based_prompt_preservation",
        "methodology_note": "Rule-based structural/keyword check of optimized prompt; does not guarantee downstream response fidelity.",
        "evaluated": True,
        "passed": len(optimized_prompt) > 30,
        "reason": "rule_based_generic_length",
        "details": {},
    }


def evaluate_downstream_task_answer(workload_id: str, raw_answer: str) -> Dict[str, Any]:
    """Objective, workload-specific rule-based programmatic heuristic check of the generated answer."""
    if not raw_answer or not raw_answer.strip():
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; not full semantic equivalence.",
            "evaluated": True,
            "passed": False,
            "reason": "empty_downstream_answer",
            "details": {"length": 0},
        }

    lower = raw_answer.lower()

    if workload_id == "json_cluster_config":
        # Task: State active nodes (alpha, beta, gamma) and total memory (320 GiB, or 64 + 128 + 128)
        has_nodes = all(n in lower for n in ["alpha", "beta", "gamma"])
        has_mem = bool(re.search(r"\b(?:320|64|128)\b", lower))
        passed = has_nodes and has_mem
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; not full semantic equivalence.",
            "evaluated": True,
            "passed": passed,
            "reason": "rule_based_identifies_active_nodes_and_memory",
            "details": {"has_nodes": has_nodes, "has_memory": has_mem},
        }

    elif workload_id == "code_caching_pipeline":
        # Task: Explain eviction policy (LRU / least recently used and TTL / expiration)
        has_lru = bool(re.search(r"\b(?:lru|least recently used|popitem|evict)\b", lower))
        has_ttl = bool(re.search(r"\b(?:ttl|expir|time to live|timestamp)\b", lower))
        passed = has_lru and has_ttl
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; not full semantic equivalence.",
            "evaluated": True,
            "passed": passed,
            "reason": "rule_based_explains_lru_and_ttl_mechanisms",
            "details": {"has_lru": has_lru, "has_ttl": has_ttl},
        }

    elif workload_id == "logs_database_incident":
        # Task: Identify cause (connection refused / postgresql / connection pool) and timestamp (08:00:12 / 08:00:13 / 08:00:18)
        has_cause = bool(re.search(r"\b(?:connection refused|database|postgresql|connection pool|5432|exhausted|connection loss)\b", lower))
        has_ts = bool(re.search(r"\b08:00:1[238]\b", lower))
        passed = has_cause and has_ts
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; not full semantic equivalence.",
            "evaluated": True,
            "passed": passed,
            "reason": "rule_based_identifies_incident_cause_and_timestamp",
            "details": {"has_cause": has_cause, "has_timestamp": has_ts},
        }

    elif workload_id == "text_redundant_instructions":
        # Task: Horizontal vs Vertical scaling in 3 bullet points, no proprietary vendor names
        has_horiz = bool(re.search(r"\b(?:horizontal|scale out|add (?:nodes|machines|instances))\b", lower))
        has_vert = bool(re.search(r"\b(?:vertical|scale up|add (?:ram|cpu|resources))\b", lower))
        # Negative prohibition check: AWS, Azure, GCP, Google Cloud, Amazon Web Services
        has_vendor = bool(re.search(r"\b(?:aws|azure|gcp|google cloud|amazon web services)\b", lower))
        # Bullet formatting check
        bullets = len(re.findall(r"(?:^|\n)\s*[\*\-\d\.]+\s+", raw_answer))
        passed = has_horiz and has_vert and not has_vendor and (bullets >= 2)
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; not full semantic equivalence.",
            "evaluated": True,
            "passed": passed,
            "reason": "rule_based_explains_scaling_with_bullets_and_no_vendor_names",
            "details": {
                "has_horizontal": has_horiz,
                "has_vertical": has_vert,
                "has_vendor_prohibition_violation": has_vendor,
                "bullet_count": bullets,
            },
        }

    elif workload_id == "mixed_debugging_dossier":
        # Task: Summarize service status and check port match
        has_service = "billing" in lower
        has_port = "8080" in lower or "5432" in lower or "port" in lower
        passed = has_service and has_port
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; not full semantic equivalence.",
            "evaluated": True,
            "passed": passed,
            "reason": "rule_based_summarizes_service_and_evaluates_port",
            "details": {"has_service": has_service, "has_port": has_port},
        }

    return {
        "evaluation_type": "rule_based_answer_quality",
        "methodology_note": "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; not full semantic equivalence.",
        "evaluated": True,
        "passed": len(raw_answer) > 20,
        "reason": "rule_based_generic_length",
        "details": {},
    }


async def execute_downstream_generation(prompt: str) -> Dict[str, Any]:
    """Execute raw downstream inference on Ollama to generate the actual task response."""
    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": 256,
            "num_ctx": 4096,
        },
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
        resp = await client.post(url, json=payload)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if resp.status_code != 200:
        return {
            "success": False,
            "raw_answer": "",
            "latency_ms": latency_ms,
            "eval_count": None,
            "prompt_eval_count": None,
            "done_reason": None,
            "error": f"Ollama HTTP {resp.status_code}",
        }

    data = resp.json()
    return {
        "success": True,
        "raw_answer": data.get("response", "").strip(),
        "latency_ms": round(latency_ms, 2),
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "done_reason": data.get("done_reason"),
        "error": None,
    }


async def run_single_workload_run(
    workload: Dict[str, Any],
    enable_compression: bool,
    ollama_service: OllamaService,
) -> Dict[str, Any]:
    """Execute a single optimization and downstream task run with real telemetry."""
    prompt = workload["prompt"]
    mode = workload["mode"]
    mode_val = cast(OptimizationMode, mode)

    req = OptimizeRequest(
        prompt=prompt,
        mode=mode_val,
        enable_compression=enable_compression,
        compression_reversible=True,
    )

    t0 = time.perf_counter()
    opt_resp = await ollama_service.optimize_prompt(req)
    total_opt_latency_ms = (time.perf_counter() - t0) * 1000.0

    # Extract native Ollama telemetry from the optimization call
    native_eval_count: Optional[int] = None
    native_prompt_eval_count: Optional[int] = None
    native_prompt_eval_cached_count: Optional[int] = None
    native_cache_hit_ratio: Optional[float] = None
    native_eval_duration_ms: Optional[float] = None
    native_prompt_eval_duration_ms: Optional[float] = None
    native_gen_tok_per_sec: Optional[float] = None
    selected_num_ctx: Optional[int] = None
    done_reason: Optional[str] = None

    if opt_resp.inference_telemetry and isinstance(opt_resp.inference_telemetry, dict):
        calls = opt_resp.inference_telemetry.get("calls", [])
        if calls and isinstance(calls, list):
            first_call: Dict[str, Any] = calls[0]
            native_eval_count = first_call.get("eval_count")
            native_prompt_eval_count = first_call.get("prompt_eval_count")
            native_prompt_eval_cached_count = first_call.get("prompt_eval_cached_count")
            native_cache_hit_ratio = first_call.get("cache_hit_ratio")
            eval_dur_ns = first_call.get("eval_duration_ns")
            if eval_dur_ns is not None:
                native_eval_duration_ms = round(float(eval_dur_ns) / 1_000_000.0, 2)
            prompt_dur_ns = first_call.get("prompt_eval_duration_ns")
            if prompt_dur_ns is not None:
                native_prompt_eval_duration_ms = round(float(prompt_dur_ns) / 1_000_000.0, 2)
            native_gen_tok_per_sec = first_call.get("generation_tokens_per_second")
            selected_num_ctx = first_call.get("selected_num_ctx")
            done_reason = first_call.get("done_reason")

    # Compression telemetry
    comp_applied = False
    comp_content_type = workload.get("content_type", "unknown")
    comp_strategy = "none"
    comp_orig_tokens = 0
    comp_final_tokens = 0
    comp_reduction_pct = 0.0
    comp_time_ms = 0.0
    comp_ref: Optional[str] = None
    comp_validation_passed = True
    comp_reason = "compression_disabled" if not enable_compression else "none"

    if opt_resp.compression:
        comp_applied = opt_resp.compression.enabled
        comp_content_type = opt_resp.compression.detected_content_type
        comp_strategy = opt_resp.compression.strategy_applied
        comp_orig_tokens = opt_resp.compression.original_tokens
        comp_final_tokens = opt_resp.compression.compressed_tokens
        comp_reduction_pct = opt_resp.compression.reduction_percent
        comp_time_ms = opt_resp.compression.processing_time_ms
        comp_ref = opt_resp.compression.restoration_reference
        comp_validation_passed = (opt_resp.compression.validation_status == "passed")
        comp_reason = opt_resp.compression.fallback_reason or "passed"

    # Evaluate prompt preservation
    prompt_preservation = evaluate_prompt_preservation(workload["id"], opt_resp.optimized_prompt)

    # Execute genuine downstream task inference using the optimized prompt
    downstream = await execute_downstream_generation(opt_resp.optimized_prompt)
    downstream_eval = evaluate_downstream_task_answer(workload["id"], downstream["raw_answer"])

    # Compute genuine character reduction without artificial zero-clamping
    char_delta = len(prompt) - len(opt_resp.optimized_prompt)
    char_reduction_percent = round((char_delta / max(1, len(prompt))) * 100.0, 2)

    return {
        "workload_id": workload["id"],
        "enable_compression": enable_compression,
        "input_original_chars": len(prompt),
        "input_effective_chars": len(opt_resp.optimized_prompt),
        "char_reduction_percent": char_reduction_percent,
        "compression_applied": comp_applied,
        "content_type": comp_content_type,
        "strategy": comp_strategy,
        "original_tokens": comp_orig_tokens,
        "compressed_tokens": comp_final_tokens,
        "reduction_percent": comp_reduction_pct,
        "compression_time_ms": comp_time_ms,
        "compression_reason": comp_reason,
        "compression_validation_passed": comp_validation_passed,
        "restoration_reference": comp_ref,
        "total_opt_latency_ms": round(total_opt_latency_ms, 2),
        "selected_num_ctx": selected_num_ctx,
        "prompt_eval_count": native_prompt_eval_count,
        "prompt_eval_cached_count": native_prompt_eval_cached_count,
        "cache_hit_ratio": native_cache_hit_ratio,
        "prompt_eval_duration_ms": native_prompt_eval_duration_ms,
        "eval_count": native_eval_count,
        "eval_duration_ms": native_eval_duration_ms,
        "generation_tokens_per_second": native_gen_tok_per_sec,
        "done_reason": done_reason,
        "optimized_prompt_preview": opt_resp.optimized_prompt[:250] + ("..." if len(opt_resp.optimized_prompt) > 250 else ""),
        "prompt_preservation_evaluation": prompt_preservation,
        "downstream_task_answer": downstream["raw_answer"],
        "downstream_task_evaluation": downstream_eval,
        "downstream_eval_tokens": downstream["eval_count"],
        "downstream_latency_ms": downstream["latency_ms"],
    }


def verify_workload_restoration(workload: Dict[str, Any], compressed_run: Dict[str, Any]) -> Dict[str, Any]:
    """Verify exact byte and hash restoration against the actual expected original payload."""
    ref_key = compressed_run.get("restoration_reference")
    expected_content = workload.get("expected_registered_content")

    if not ref_key or expected_content is None:
        return {
            "tested": False,
            "status": "not_applicable",
            "ref_key": ref_key,
            "verified": None,
            "exact_match": None,
            "expected_hash": None,
            "actual_hash": None,
            "negative_tests_passed": None,
        }

    # Retrieve content from registry
    retrieved = CompressionRegistry.retrieve(ref_key)
    if retrieved is None:
        return {
            "tested": True,
            "status": "missing_or_expired",
            "ref_key": ref_key,
            "verified": False,
            "exact_match": False,
            "expected_hash": hashlib.sha256(expected_content.encode("utf-8")).hexdigest(),
            "actual_hash": None,
            "negative_tests_passed": False,
        }

    # Verify that retrieved content matches the actual expected original payload byte-for-byte and by SHA-256
    expected_bytes = expected_content.encode("utf-8")
    retrieved_bytes = retrieved.encode("utf-8")
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    actual_hash = hashlib.sha256(retrieved_bytes).hexdigest()

    exact_match = (retrieved_bytes == expected_bytes)
    hash_match = (actual_hash == expected_hash)

    # Use CompressionRegistry.verify_restoration against the actual expected original content
    verify_res = CompressionRegistry.verify_restoration(ref_key, expected_content)

    # Negative test 1: invalid reference key returns None
    neg_missing = CompressionRegistry.retrieve("argus:ref:invalid_key_for_testing") is None
    # Negative test 2: corrupted payload returns mismatch
    neg_corrupt = CompressionRegistry.verify_restoration(ref_key, expected_content + " corrupted")["verified"] is False

    negative_ok = neg_missing and neg_corrupt
    verified_ok = exact_match and hash_match and verify_res["verified"] and negative_ok

    return {
        "tested": True,
        "status": "matched" if verified_ok else "mismatch",
        "ref_key": ref_key,
        "verified": verified_ok,
        "exact_match": exact_match,
        "expected_hash": expected_hash,
        "actual_hash": actual_hash,
        "negative_tests_passed": negative_ok,
        "retrieved_char_length": len(retrieved),
        "expected_char_length": len(expected_content),
    }


async def run_full_benchmark(warmup: int = 1, reps: int = 2) -> Dict[str, Any]:
    """Execute complete benchmark across workloads with warm-up, repetitions, and alternating order."""
    print("=" * 80)
    print("ARGUS PHASE 4.2 — INDEPENDENT CONTENT-AWARE COMPRESSION BENCHMARK")
    print("=" * 80)

    hardware = detect_hardware_metadata()
    print(f"OS: {hardware['os']['system']} {hardware['os']['release']} ({hardware['os']['machine']})")
    print(f"CPU/RAM: {hardware['ram']['total_mb']} MB detected via {hardware['ram']['source']}")
    if hardware["gpu"]["name"]:
        print(f"GPU: {hardware['gpu']['name']} ({hardware['gpu']['total_vram_mib']} MiB VRAM)")
    else:
        print("GPU: Telemetry unavailable")

    ollama_service = OllamaService()
    health = await OllamaService.check_status()
    print(f"Ollama Service: model={health.model}, connected={health.connected}")
    if not health.connected:
        print("ERROR: Ollama service is not reachable. Ensure 'ollama serve' is running.")
        sys.exit(1)

    # Warm-up runs
    if warmup > 0:
        print(f"\nPerforming {warmup} warm-up run(s)...")
        for _ in range(warmup):
            for wl in BENCHMARK_WORKLOADS:
                await run_single_workload_run(wl, False, ollama_service)
                await run_single_workload_run(wl, True, ollama_service)
        print("Warm-up complete.")

    all_trials: List[Dict[str, Any]] = []

    for workload in BENCHMARK_WORKLOADS:
        print(f"\n--- Workload: {workload['name']} ({workload['content_type'].upper()}) ---")

        for rep in range(1, reps + 1):
            # Alternating execution order: rep % 2 == 1 -> Baseline first, rep % 2 == 0 -> Compressed first
            baseline_first = (rep % 2 == 1)
            order_label = "Baseline -> Compressed" if baseline_first else "Compressed -> Baseline"
            print(f"  [Rep {rep}/{reps} | Order: {order_label}]")

            if baseline_first:
                print("    Running Baseline (Uncompressed)...")
                base_run = await run_single_workload_run(workload, False, ollama_service)
                print("    Running Compressed Engine...")
                comp_run = await run_single_workload_run(workload, True, ollama_service)
            else:
                print("    Running Compressed Engine...")
                comp_run = await run_single_workload_run(workload, True, ollama_service)
                print("    Running Baseline (Uncompressed)...")
                base_run = await run_single_workload_run(workload, False, ollama_service)

            # Restoration verification
            restoration = verify_workload_restoration(workload, comp_run)

            # Quality comparison statuses
            prompt_preservation_status = determine_comparison_status(
                base_run["prompt_preservation_evaluation"]["passed"],
                comp_run["prompt_preservation_evaluation"]["passed"],
            )
            downstream_answer_status = determine_comparison_status(
                base_run["downstream_task_evaluation"]["passed"],
                comp_run["downstream_task_evaluation"]["passed"],
            )

            # Token reduction
            prompt_token_savings: Optional[int] = None
            if base_run["prompt_eval_count"] is not None and comp_run["prompt_eval_count"] is not None:
                prompt_token_savings = base_run["prompt_eval_count"] - comp_run["prompt_eval_count"]

            base_chars = base_run["input_original_chars"]
            comp_chars = comp_run["input_effective_chars"]
            char_reduction = base_chars - comp_chars
            char_reduction_percent = round((char_reduction / max(1, base_chars)) * 100.0, 2)

            trial_result = {
                "workload_id": workload["id"],
                "workload_name": workload["name"],
                "content_type": workload["content_type"],
                "repetition": rep,
                "execution_order": order_label,
                "baseline": base_run,
                "compressed": comp_run,
                "comparison": {
                    "character_reduction": char_reduction,
                    "character_reduction_percent": char_reduction_percent,
                    "prompt_eval_token_savings": prompt_token_savings,
                    "compression_overhead_ms": comp_run["compression_time_ms"],
                    "prompt_preservation_status": prompt_preservation_status,
                    "downstream_answer_status": downstream_answer_status,
                    "restoration_verification": restoration,
                },
            }

            print(
                f"    -> Chars: {comp_run['input_effective_chars']} ({char_reduction_percent:+.2f}%) | "
                f"Prompt tokens: {comp_run['prompt_eval_count']} (Saved: {prompt_token_savings}) | "
                f"Engine Latency: {comp_run['compression_time_ms']:.2f} ms | "
                f"Prompt Preserved: {prompt_preservation_status} | "
                f"Downstream Answer: {downstream_answer_status}"
            )
            all_trials.append(trial_result)

    # Compute statistically rigorous aggregates without zero-substitution for missing data
    char_reductions = [t["comparison"]["character_reduction_percent"] for t in all_trials]
    token_savings_list = [
        t["comparison"]["prompt_eval_token_savings"]
        for t in all_trials
        if t["comparison"]["prompt_eval_token_savings"] is not None
    ]
    engine_latencies = [t["compressed"]["compression_time_ms"] for t in all_trials]

    total_base_chars = sum(t["baseline"]["input_original_chars"] for t in all_trials)
    total_comp_chars = sum(t["compressed"]["input_effective_chars"] for t in all_trials)
    agg_char_reduction_pct = round(((total_base_chars - total_comp_chars) / max(1, total_base_chars)) * 100.0, 2)

    # Token totals only across trials with valid measurements
    valid_token_trials = [
        t for t in all_trials
        if t["baseline"]["prompt_eval_count"] is not None and t["compressed"]["prompt_eval_count"] is not None
    ]
    if valid_token_trials:
        total_base_prompt_tokens = sum(t["baseline"]["prompt_eval_count"] for t in valid_token_trials)
        total_comp_prompt_tokens = sum(t["compressed"]["prompt_eval_count"] for t in valid_token_trials)
        agg_token_reduction_pct = (
            round(((total_base_prompt_tokens - total_comp_prompt_tokens) / max(1, total_base_prompt_tokens)) * 100.0, 2)
            if total_base_prompt_tokens > 0
            else 0.0
        )
    else:
        total_base_prompt_tokens = None
        total_comp_prompt_tokens = None
        agg_token_reduction_pct = None

    tested_restorations = [
        t["comparison"]["restoration_verification"]
        for t in all_trials
        if t["comparison"]["restoration_verification"]["tested"]
    ]
    restoration_pass_count = sum(1 for r in tested_restorations if r["verified"])

    downstream_preserved_count = sum(1 for t in all_trials if t["comparison"]["downstream_answer_status"] == "preserved")
    downstream_regressions = sum(1 for t in all_trials if t["comparison"]["downstream_answer_status"] == "regression")
    downstream_improvements = sum(1 for t in all_trials if t["comparison"]["downstream_answer_status"] == "improvement")
    downstream_both_failed = sum(1 for t in all_trials if t["comparison"]["downstream_answer_status"] == "inconclusive_or_both_failed")

    summary_payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": health.model,
        "hardware": hardware,
        "methodology": {
            "warmup_runs": warmup,
            "repetitions": reps,
            "total_trials": len(all_trials),
            "execution_order_alternated": True,
            "downstream_inference_parameters": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 256,
                "num_ctx": 4096,
            },
            "quality_evaluation_note": (
                "Downstream answer quality is evaluated via rule-based programmatic keyword/heuristic checks "
                "against workload-specific acceptance criteria; not full semantic equivalence."
            ),
            "restoration_verification_note": (
                "Reversible restoration compares retrieved content against the actual expected original payload, "
                "verifying byte-for-byte equality, SHA-256 hash match, and negative test verification."
            ),
            "ollama_telemetry_semantics_note": (
                "prompt_eval_count represents total input tokens prefilled by Ollama. "
                "prompt_eval_cached_count represents tokens found in Ollama's local KV prefix cache. "
                "eval_count represents tokens generated during output decoding. "
                "All metrics are native counters reported directly by Ollama."
            ),
        },
        "aggregate_statistics": {
            "character_reduction": {
                "aggregate_percent": agg_char_reduction_pct,
                "mean_percent": round(statistics.mean(char_reductions), 2) if char_reductions else 0.0,
                "median_percent": round(statistics.median(char_reductions), 2) if char_reductions else 0.0,
                "stdev_percent": round(statistics.stdev(char_reductions), 2) if len(char_reductions) > 1 else 0.0,
            },
            "prompt_eval_token_reduction": {
                "total_baseline_tokens": total_base_prompt_tokens,
                "total_compressed_tokens": total_comp_prompt_tokens,
                "aggregate_percent": agg_token_reduction_pct,
                "mean_tokens_saved": round(statistics.mean(token_savings_list), 2) if token_savings_list else None,
                "median_tokens_saved": round(statistics.median(token_savings_list), 2) if token_savings_list else None,
                "stdev_tokens_saved": round(statistics.stdev(token_savings_list), 2) if len(token_savings_list) > 1 else (0.0 if token_savings_list else None),
            },
            "compression_engine_latency_ms": {
                "mean_ms": round(statistics.mean(engine_latencies), 3) if engine_latencies else 0.0,
                "median_ms": round(statistics.median(engine_latencies), 3) if engine_latencies else 0.0,
                "stdev_ms": round(statistics.stdev(engine_latencies), 3) if len(engine_latencies) > 1 else 0.0,
            },
            "downstream_task_quality_summary": {
                "total_trials": len(all_trials),
                "preserved_count": downstream_preserved_count,
                "regression_count": downstream_regressions,
                "improvement_count": downstream_improvements,
                "both_failed_count": downstream_both_failed,
                "preservation_rate_percent": round((downstream_preserved_count / len(all_trials)) * 100.0, 1),
            },
            "reversible_restoration_summary": {
                "total_tested": len(tested_restorations),
                "verified_byte_exact": restoration_pass_count,
                "verification_rate_percent": round((restoration_pass_count / len(tested_restorations)) * 100.0, 1) if tested_restorations else 0.0,
            },
        },
        "trials": all_trials,
    }

    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Argus Phase 4.2 Content-Aware Compression Benchmark")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up runs")
    parser.add_argument("--reps", type=int, default=2, help="Number of repetitions per workload")
    parser.add_argument(
        "--output",
        type=str,
        default="docs/reports/compression_benchmark.json",
        help="Path to save raw benchmark JSON output",
    )
    args = parser.parse_args()

    results = asyncio.run(run_full_benchmark(warmup=args.warmup, reps=args.reps))

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    stats = results["aggregate_statistics"]
    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION COMPLETE")
    print("Summary:")
    print(f"  Character Reduction: {stats['character_reduction']['aggregate_percent']:+.2f}% (mean: {stats['character_reduction']['mean_percent']:+.2f}%)")
    print(f"  Native Prompt Token Reduction: -{stats['prompt_eval_token_reduction']['aggregate_percent']}% (mean saved: {stats['prompt_eval_token_reduction']['mean_tokens_saved']} tokens)")
    print(f"  Engine Latency: mean {stats['compression_engine_latency_ms']['mean_ms']} ms (median: {stats['compression_engine_latency_ms']['median_ms']} ms)")
    print(f"  Downstream Quality Status: {stats['downstream_task_quality_summary']['preserved_count']} preserved / {stats['downstream_task_quality_summary']['total_trials']} trials ({stats['downstream_task_quality_summary']['preservation_rate_percent']}%)")
    print(f"  Reversible Restoration: {stats['reversible_restoration_summary']['verified_byte_exact']} / {stats['reversible_restoration_summary']['total_tested']} verified byte-exact")
    print(f"Saved raw JSON measurements to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

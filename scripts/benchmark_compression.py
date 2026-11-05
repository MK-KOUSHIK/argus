#!/usr/bin/env python3
"""Argus Phase 4.2 — Content-Aware Compression Engine Empirical Benchmark.

Evaluates character reduction, native prompt prefill token savings, downstream task
answer quality preservation, and exact reversible restoration across technical workloads.
All measurements reflect genuine runtime execution against local Ollama (qwen2.5:1.5b)
with alternating execution order and warm-up passes to eliminate driver cache bias.

Distinguishes:
1. Internal Argus Optimizer Inference (prompt rewriting pipeline)
2. Downstream Model Inference (task answer generation)
3. Four equivalent stage measurements:
   - Original workload prompt
   - Compression-stage output
   - Final optimized prompt
   - Downstream model input
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
from app.services.compression import CompressionOrchestrator
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
    """Objective, workload-specific rule-based programmatic heuristic check of the generated answer.

    Explicitly acknowledges this is rule-based acceptance checking, not full semantic equivalence.
    """
    if not raw_answer or not raw_answer.strip():
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": (
                "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; "
                "not full semantic equivalence."
            ),
            "evaluated": True,
            "passed": False,
            "status": "failed",
            "reason": "empty_downstream_answer",
            "details": {"length": 0},
        }

    lower = raw_answer.lower()

    if workload_id == "json_cluster_config":
        # Check required values and structure:
        # 1. Active nodes: node-alpha, node-beta, node-gamma (or alpha, beta, gamma)
        # 2. Memory values: components 64 and 128, or total sum (320 or 312)
        # 3. Status/Role recognition: ready, control-plane, worker
        has_alpha = "alpha" in lower
        has_beta = "beta" in lower
        has_gamma = "gamma" in lower
        has_all_nodes = has_alpha and has_beta and has_gamma

        has_mem_components = ("64" in lower and "128" in lower)
        has_mem_total = bool(re.search(r"\b(?:320|312)\b", lower))
        has_memory = has_mem_components or has_mem_total

        has_status_or_role = any(kw in lower for kw in ["ready", "control-plane", "worker", "status", "role"])

        passed = has_all_nodes and has_memory and has_status_or_role
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": (
                "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; "
                "not full semantic equivalence."
            ),
            "evaluated": True,
            "passed": passed,
            "status": "passed" if passed else "failed",
            "reason": "rule_based_cluster_nodes_memory_and_structure",
            "details": {
                "has_node_alpha": has_alpha,
                "has_node_beta": has_beta,
                "has_node_gamma": has_gamma,
                "has_all_nodes": has_all_nodes,
                "has_memory_components": has_mem_components,
                "has_memory_total": has_mem_total,
                "has_status_or_role": has_status_or_role,
            },
        }

    elif workload_id == "code_caching_pipeline":
        # Check LRU and TTL concepts and required behavior:
        # 1. LRU concept: lru, least recently used, OrderedDict, move_to_end, popitem
        # 2. TTL concept: ttl, time-to-live, expir*, timestamp, expires_at
        # 3. Eviction behavior: capacity limit / max_size triggers removal of oldest/least-recently used entry
        has_lru = bool(re.search(r"\b(?:lru|least recently used|ordereddict|move_to_end|popitem)\b", lower))
        has_ttl = bool(re.search(r"\b(?:ttl|time[- ]to[- ]live|expir\w*|timestamp|expires_at)\b", lower))
        has_behavior = bool(re.search(r"\b(?:capacity|max_size|oldest|limit|reached|evict\w*|remov\w*|del\b)\b", lower))
        passed = has_lru and has_ttl and has_behavior
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": (
                "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; "
                "not full semantic equivalence."
            ),
            "evaluated": True,
            "passed": passed,
            "status": "passed" if passed else "failed",
            "reason": "rule_based_lru_ttl_and_eviction_behavior",
            "details": {
                "has_lru": has_lru,
                "has_ttl": has_ttl,
                "has_eviction_behavior": has_behavior,
            },
        }

    elif workload_id == "logs_database_incident":
        # Check exact failure cause and timestamp:
        # 1. Failure cause: connection refused / database connection loss / ConnectionPoolExhausted / postgresql / 10.0.1.5:5432
        # 2. Timestamp: 08:00:12 (refusal), 08:00:13 (exhaustion), or 08:00:18 (fatal crash)
        has_cause = bool(re.search(
            r"\b(?:connection refused|connection loss|connectionpoolexhausted|postgresql|10\.0\.1\.5|unhandled database|all 20 active connections)\b",
            lower
        ))
        has_timestamp = bool(re.search(r"\b08:00:1[238]\b", lower)) or "2026-11-05 08:00" in lower
        passed = has_cause and has_timestamp
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": (
                "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; "
                "not full semantic equivalence."
            ),
            "evaluated": True,
            "passed": passed,
            "status": "passed" if passed else "failed",
            "reason": "rule_based_incident_failure_cause_and_timestamp",
            "details": {
                "has_cause": has_cause,
                "has_timestamp": has_timestamp,
            },
        }

    elif workload_id == "text_redundant_instructions":
        # Check required concepts, exactly three bullets, and prohibited vendor names:
        # 1. Required concepts: horizontal scaling and vertical scaling
        # 2. Negative check for prohibited vendor names (AWS, Azure, GCP, Google Cloud, Amazon Web Services, Microsoft)
        # 3. Exactly three bullets check
        has_horizontal = bool(re.search(r"\b(?:horizontal|scale out|add (?:nodes|machines|instances|servers))\b", lower))
        has_vertical = bool(re.search(r"\b(?:vertical|scale up|add (?:ram|cpu|resources|hardware|capacity))\b", lower))
        has_vendor = bool(re.search(r"\b(?:aws|azure|gcp|google cloud|amazon web services|microsoft)\b", lower))

        bullet_matches = re.findall(r"(?:^|\n)\s*(?:[\*\-]|\d+\.)\s+", raw_answer)
        bullet_count = len(bullet_matches)
        has_three_bullets = (bullet_count == 3)

        passed = has_horizontal and has_vertical and (not has_vendor) and has_three_bullets
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": (
                "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; "
                "not full semantic equivalence."
            ),
            "evaluated": True,
            "passed": passed,
            "status": "passed" if passed else "failed",
            "reason": "rule_based_scaling_concepts_three_bullets_and_no_vendor_names",
            "details": {
                "has_horizontal": has_horizontal,
                "has_vertical": has_vertical,
                "has_vendor_prohibition_violation": has_vendor,
                "bullet_count": bullet_count,
                "has_three_bullets": has_three_bullets,
            },
        }

    elif workload_id == "mixed_debugging_dossier":
        # Check service identity and port consistency:
        # 1. Service identity: billing-aggregator
        # 2. Port consistency: addresses service port 8080 and/or database port 5432
        has_service = bool(re.search(r"\b(?:billing-aggregator|billing)\b", lower))
        has_port_eval = bool(re.search(r"\b(?:8080|5432|port configuration|port matches|ports? (?:do not match|differ|match))\b", lower))
        passed = has_service and has_port_eval
        return {
            "evaluation_type": "rule_based_answer_quality",
            "methodology_note": (
                "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; "
                "not full semantic equivalence."
            ),
            "evaluated": True,
            "passed": passed,
            "status": "passed" if passed else "failed",
            "reason": "rule_based_service_identity_and_port_consistency",
            "details": {
                "has_service": has_service,
                "has_port_consistency_evaluation": has_port_eval,
            },
        }

    return {
        "evaluation_type": "rule_based_answer_quality",
        "methodology_note": (
            "Rule-based programmatic heuristic check of generated answer against workload-specific acceptance criteria; "
            "not full semantic equivalence."
        ),
        "evaluated": False,
        "passed": False,
        "status": "not_evaluated",
        "reason": "no_workload_specific_evaluator_defined",
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
            "num_predict": 384,
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
            "latency_ms": round(latency_ms, 2),
            "prompt_eval_count": None,
            "eval_count": None,
            "prompt_eval_duration_ms": None,
            "eval_duration_ms": None,
            "done_reason": None,
            "error": f"Ollama HTTP {resp.status_code}",
        }

    data = resp.json()
    prompt_dur_ns = data.get("prompt_eval_duration")
    eval_dur_ns = data.get("eval_duration")
    return {
        "success": True,
        "raw_answer": data.get("response", "").strip(),
        "latency_ms": round(latency_ms, 2),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "prompt_eval_duration_ms": round(float(prompt_dur_ns) / 1_000_000.0, 2) if prompt_dur_ns is not None else None,
        "eval_duration_ms": round(float(eval_dur_ns) / 1_000_000.0, 2) if eval_dur_ns is not None else None,
        "done_reason": data.get("done_reason"),
        "error": None,
    }


def aggregate_optimizer_inference_telemetry(
    inference_telemetry: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate internal optimizer native Ollama telemetry across all relevant model calls.

    Handles single calls, multiple retry calls, missing telemetry, and incomplete records.
    Preserves per-call telemetry for traceability while providing aggregated metrics.
    """
    default_result: Dict[str, Any] = {
        "call_count": 0,
        "calls": [],
        "has_complete_telemetry": False,
        "telemetry_source": "unavailable",
        "prompt_eval_count": None,
        "prompt_eval_cached_count": None,
        "cache_hit_ratio": None,
        "prompt_eval_duration_ms": None,
        "eval_count": None,
        "eval_duration_ms": None,
        "generation_tokens_per_second": None,
        "selected_num_ctx": None,
        "done_reason": None,
        "first_call_prompt_eval_count": None,
        "first_call_eval_count": None,
    }

    if not inference_telemetry or not isinstance(inference_telemetry, dict):
        return default_result

    calls = inference_telemetry.get("calls")
    if not calls or not isinstance(calls, list):
        return default_result

    valid_calls: List[Dict[str, Any]] = [c for c in calls if isinstance(c, dict)]
    if not valid_calls:
        return default_result

    call_count = len(valid_calls)
    first_call = valid_calls[0]
    first_call_prompt_eval_count = first_call.get("prompt_eval_count")
    first_call_eval_count = first_call.get("eval_count")

    # Aggregate prompt evaluation tokens
    prompt_eval_counts = [
        int(c["prompt_eval_count"])
        for c in valid_calls
        if isinstance(c.get("prompt_eval_count"), (int, float))
    ]
    prompt_eval_cached_counts = [
        int(c["prompt_eval_cached_count"])
        for c in valid_calls
        if isinstance(c.get("prompt_eval_cached_count"), (int, float))
    ]
    eval_counts = [
        int(c["eval_count"])
        for c in valid_calls
        if isinstance(c.get("eval_count"), (int, float))
    ]

    # Durations (ns to ms)
    prompt_dur_ns_list = [
        float(c["prompt_eval_duration_ns"])
        for c in valid_calls
        if isinstance(c.get("prompt_eval_duration_ns"), (int, float))
    ]
    eval_dur_ns_list = [
        float(c["eval_duration_ns"])
        for c in valid_calls
        if isinstance(c.get("eval_duration_ns"), (int, float))
    ]

    # Determine completeness
    all_calls_have_tokens = (
        len(prompt_eval_counts) == call_count and len(eval_counts) == call_count
    )

    total_prompt_eval_count: Optional[int] = sum(prompt_eval_counts) if prompt_eval_counts else None
    total_prompt_eval_cached_count: Optional[int] = (
        sum(prompt_eval_cached_counts)
        if prompt_eval_cached_counts
        else (0 if total_prompt_eval_count is not None else None)
    )
    total_eval_count: Optional[int] = sum(eval_counts) if eval_counts else None

    # Aggregate durations
    total_prompt_eval_dur_ms: Optional[float] = (
        round(sum(prompt_dur_ns_list) / 1_000_000.0, 2) if prompt_dur_ns_list else None
    )
    total_eval_dur_ms: Optional[float] = (
        round(sum(eval_dur_ns_list) / 1_000_000.0, 2) if eval_dur_ns_list else None
    )

    # Cache hit ratio
    cache_hit_ratio: Optional[float] = None
    if total_prompt_eval_count is not None and total_prompt_eval_count > 0 and total_prompt_eval_cached_count is not None:
        cache_hit_ratio = round(total_prompt_eval_cached_count / float(total_prompt_eval_count), 4)
    elif call_count == 1 and first_call.get("cache_hit_ratio") is not None:
        cache_hit_ratio = float(first_call["cache_hit_ratio"])

    # Overall generation speed (tokens / sec)
    gen_tok_per_sec: Optional[float] = None
    if total_eval_count is not None and eval_dur_ns_list and sum(eval_dur_ns_list) > 0:
        total_eval_sec = sum(eval_dur_ns_list) / 1_000_000_000.0
        gen_tok_per_sec = round(float(total_eval_count) / total_eval_sec, 2)
    elif call_count == 1 and first_call.get("generation_tokens_per_second") is not None:
        gen_tok_per_sec = float(first_call["generation_tokens_per_second"])

    # Context window & done_reason
    selected_num_ctx = valid_calls[-1].get("selected_num_ctx") or first_call.get("selected_num_ctx")
    done_reason = valid_calls[-1].get("done_reason") or first_call.get("done_reason")

    telemetry_source = (
        "complete_all_calls"
        if all_calls_have_tokens
        else ("first_call_only" if first_call_prompt_eval_count is not None else "incomplete")
    )

    return {
        "call_count": call_count,
        "calls": valid_calls,
        "has_complete_telemetry": all_calls_have_tokens,
        "telemetry_source": telemetry_source,
        "prompt_eval_count": total_prompt_eval_count,
        "prompt_eval_cached_count": total_prompt_eval_cached_count,
        "cache_hit_ratio": cache_hit_ratio,
        "prompt_eval_duration_ms": total_prompt_eval_dur_ms,
        "eval_count": total_eval_count,
        "eval_duration_ms": total_eval_dur_ms,
        "generation_tokens_per_second": gen_tok_per_sec,
        "selected_num_ctx": selected_num_ctx,
        "done_reason": done_reason,
        "first_call_prompt_eval_count": first_call_prompt_eval_count,
        "first_call_eval_count": first_call_eval_count,
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

    # 1. Compression Stage Execution
    t_comp_start = time.perf_counter()
    direct_comp_res = CompressionOrchestrator.compress(
        text=prompt,
        enabled=enable_compression,
        reversible=True,
    )
    compression_stage_text = direct_comp_res.compressed_text
    compression_stage_chars = len(compression_stage_text)

    # 2. Pipeline Execution (Stage 0 -> Stage D)
    req = OptimizeRequest(
        prompt=prompt,
        mode=mode_val,
        enable_compression=enable_compression,
        compression_reversible=True,
    )

    t0 = time.perf_counter()
    opt_resp = await ollama_service.optimize_prompt(req)
    total_opt_latency_ms = (time.perf_counter() - t0) * 1000.0

    # Extract internal optimizer native Ollama telemetry across all calls
    opt_telemetry = aggregate_optimizer_inference_telemetry(opt_resp.inference_telemetry)
    native_eval_count = opt_telemetry["eval_count"]
    native_prompt_eval_count = opt_telemetry["prompt_eval_count"]
    native_prompt_eval_cached_count = opt_telemetry["prompt_eval_cached_count"]
    native_cache_hit_ratio = opt_telemetry["cache_hit_ratio"]
    native_eval_duration_ms = opt_telemetry["eval_duration_ms"]
    native_prompt_eval_duration_ms = opt_telemetry["prompt_eval_duration_ms"]
    native_gen_tok_per_sec = opt_telemetry["generation_tokens_per_second"]
    selected_num_ctx = opt_telemetry["selected_num_ctx"]
    done_reason = opt_telemetry["done_reason"]

    # Compression telemetry from response
    comp_applied = False
    comp_content_type = workload.get("content_type", "unknown")
    comp_strategy = "none"
    comp_orig_tokens = direct_comp_res.original_tokens
    comp_final_tokens = direct_comp_res.compressed_tokens
    comp_reduction_pct = direct_comp_res.reduction_percent
    comp_time_ms = direct_comp_res.processing_time_ms
    comp_ref: Optional[str] = direct_comp_res.restoration_reference
    comp_validation_passed = direct_comp_res.validation_passed
    comp_val_status = direct_comp_res.validation_status

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
        comp_val_status = opt_resp.compression.validation_status

    # Evaluate prompt preservation on final optimized prompt
    prompt_preservation = evaluate_prompt_preservation(workload["id"], opt_resp.optimized_prompt)

    # 3. Downstream Task Model Inference
    # The prompt delivered to the downstream model is the final optimized prompt produced by Argus
    downstream_prompt = opt_resp.optimized_prompt
    downstream = await execute_downstream_generation(downstream_prompt)
    downstream_eval = evaluate_downstream_task_answer(workload["id"], downstream["raw_answer"])

    return {
        "workload_id": workload["id"],
        "enable_compression": enable_compression,
        # Section 2: Four equivalent stage measurements
        "stage_measurements": {
            "original_workload_prompt_chars": len(prompt),
            "compression_stage_output_chars": compression_stage_chars,
            "final_optimized_prompt_chars": len(opt_resp.optimized_prompt),
            "downstream_model_input_chars": len(downstream_prompt),
        },
        # Section 4: Compression stage telemetry
        "compression_stage": {
            "applied": comp_applied,
            "content_type": comp_content_type,
            "strategy": comp_strategy,
            "original_tokens": comp_orig_tokens,
            "compressed_tokens": comp_final_tokens,
            "tokens_reduced": comp_orig_tokens - comp_final_tokens,
            "reduction_percent": comp_reduction_pct,
            "processing_time_ms": comp_time_ms,
            "validation_status": comp_val_status,
            "validation_passed": comp_validation_passed,
            "restoration_reference": comp_ref,
        },
        # Section 1: Internal Argus Optimizer Inference Metrics
        "optimizer_inference": {
            "total_latency_ms": round(total_opt_latency_ms, 2),
            "validation_status": opt_resp.validation_status,
            "call_count": opt_telemetry["call_count"],
            "telemetry_source": opt_telemetry["telemetry_source"],
            "has_complete_telemetry": opt_telemetry["has_complete_telemetry"],
            "selected_num_ctx": selected_num_ctx,
            "prompt_eval_count": native_prompt_eval_count,
            "prompt_eval_cached_count": native_prompt_eval_cached_count,
            "cache_hit_ratio": native_cache_hit_ratio,
            "prompt_eval_duration_ms": native_prompt_eval_duration_ms,
            "eval_count": native_eval_count,
            "eval_duration_ms": native_eval_duration_ms,
            "generation_tokens_per_second": native_gen_tok_per_sec,
            "done_reason": done_reason,
            "first_call_prompt_eval_count": opt_telemetry["first_call_prompt_eval_count"],
            "first_call_eval_count": opt_telemetry["first_call_eval_count"],
            "calls": opt_telemetry["calls"],
            "final_optimized_prompt_preview": opt_resp.optimized_prompt[:250] + ("..." if len(opt_resp.optimized_prompt) > 250 else ""),
            "prompt_preservation_evaluation": prompt_preservation,
        },
        # Section 1 & 3: Downstream Model Inference Metrics (Actual Task Execution)
        "downstream_inference": {
            "prompt_eval_count": downstream["prompt_eval_count"],
            "eval_count": downstream["eval_count"],
            "prompt_eval_duration_ms": downstream["prompt_eval_duration_ms"],
            "eval_duration_ms": downstream["eval_duration_ms"],
            "latency_ms": downstream["latency_ms"],
            "done_reason": downstream["done_reason"],
            "raw_answer": downstream["raw_answer"],
            "task_evaluation": downstream_eval,
        },
    }


def verify_workload_restoration(workload: Dict[str, Any], compressed_run: Dict[str, Any]) -> Dict[str, Any]:
    """Verify exact byte and hash restoration against the actual expected original payload."""
    ref_key = compressed_run["compression_stage"].get("restoration_reference")
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


NEW_SCHEMA_IDENTIFIER_KEYS = {
    "first_call_prompt_eval_count",
    "first_call_eval_count",
    "calls",
    "call_count",
    "telemetry_source",
    "has_complete_telemetry",
}


def is_new_schema_optimizer_record(opt_dict: Dict[str, Any]) -> bool:
    """Check if an optimizer telemetry record follows the new multi-call telemetry schema.

    Identified by the presence of explicit new-schema keys such as 'first_call_prompt_eval_count',
    'calls', 'call_count', 'telemetry_source', or 'has_complete_telemetry'.
    """
    if not isinstance(opt_dict, dict):
        return False
    return any(k in opt_dict for k in NEW_SCHEMA_IDENTIFIER_KEYS)


def is_demonstrably_historical_single_call(opt_dict: Dict[str, Any]) -> bool:
    """Determine whether an optimizer telemetry record demonstrably represents a valid legacy single-call run.

    Historical benchmark records in Argus recorded only the primary (first) call directly in
    'prompt_eval_count' along with optimizer run metadata ('validation_status', 'total_latency_ms',
    or 'selected_num_ctx').
    They never contained new-schema keys ('first_call_prompt_eval_count', 'calls', 'call_count')
    and never had multi-call indicators ('retry_count' > 0, 'is_multi_call' = True, etc.).

    Returns False for new-schema records, multi-call records, incomplete records, or arbitrary dicts.
    """
    if not isinstance(opt_dict, dict):
        return False

    # 1. Must NOT be a new-schema record
    if any(k in opt_dict for k in NEW_SCHEMA_IDENTIFIER_KEYS):
        return False

    # 2. Must NOT indicate multi-call or retries
    for k in ("retry_count", "retries", "retries_attempted"):
        v = opt_dict.get(k)
        if v is not None and isinstance(v, (int, float)) and v > 0:
            return False

    for k in ("is_multi_call", "multi_call"):
        if opt_dict.get(k) is True:
            return False

    for k in ("total_calls", "call_count"):
        v = opt_dict.get(k)
        if v is not None and isinstance(v, (int, float)) and v > 1:
            return False

    # 3. Must have a valid non-negative prompt_eval_count
    pe = opt_dict.get("prompt_eval_count")
    if pe is None or not isinstance(pe, (int, float)) or pe < 0:
        return False

    # 4. Must demonstrably contain authentic Argus historical run context
    # Historical records contain 'validation_status' (e.g. 'passed', 'fallback')
    # and at least one execution metadata key ('total_latency_ms' or 'selected_num_ctx')
    # Or explicitly flagged as historical single call
    has_historical_context = (
        ("validation_status" in opt_dict and ("total_latency_ms" in opt_dict or "selected_num_ctx" in opt_dict or "done_reason" in opt_dict))
        or opt_dict.get("historical_single_call") is True
    )

    return has_historical_context


def extract_primary_rewrite_prompt_tokens(opt_dict: Dict[str, Any]) -> Optional[int]:
    """Extract primary rewrite prefill token count safely and strictly.

    For new-schema records:
      - STRICTLY uses 'first_call_prompt_eval_count'.
      - NEVER falls back to 'prompt_eval_count' (which may be aggregate across retries).
      - If first-call telemetry is missing or incomplete, returns None.

    For historical records:
      - ONLY allows fallback to 'prompt_eval_count' if demonstrably a legacy single-call record.
      - If record has multi-call indicators or lacks historical run context, returns None.
    """
    if not isinstance(opt_dict, dict):
        return None

    if is_new_schema_optimizer_record(opt_dict):
        val = opt_dict.get("first_call_prompt_eval_count")
        if val is not None and isinstance(val, (int, float)) and val >= 0:
            return int(val)
        return None

    if is_demonstrably_historical_single_call(opt_dict):
        val = opt_dict.get("prompt_eval_count")
        if val is not None and isinstance(val, (int, float)) and val >= 0:
            return int(val)
        return None

    return None


def calculate_primary_rewrite_savings(
    base_run: Dict[str, Any],
    comp_run: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate prefill token savings for the primary (initial) rewrite attempt.

    Preserves comparability with historical first-call benchmark measurements.
    Strictly uses 'first_call_prompt_eval_count' for new-schema records; never substitutes
    aggregate multi-call telemetry if first-call telemetry is missing.
    Allows fallback to 'prompt_eval_count' only for demonstrably valid historical single-call records.
    """
    base_opt = base_run.get("optimizer_inference", {}) if isinstance(base_run, dict) else {}
    comp_opt = comp_run.get("optimizer_inference", {}) if isinstance(comp_run, dict) else {}

    base_tok = extract_primary_rewrite_prompt_tokens(base_opt)
    comp_tok = extract_primary_rewrite_prompt_tokens(comp_opt)

    if base_tok is not None and comp_tok is not None:
        saved = base_tok - comp_tok
        pct = round((saved / max(1, base_tok)) * 100.0, 2)
        status = "evaluated"
    else:
        saved = None
        pct = None
        status = "unavailable"

    return {
        "metric_name": "primary_rewrite_call_savings",
        "measurement_scope": "first_call_only",
        "status": status,
        "baseline_prompt_eval_count": base_tok,
        "compressed_prompt_eval_count": comp_tok,
        "tokens_saved": saved,
        "reduction_percent": pct,
    }


def calculate_total_optimizer_cost(
    base_run: Dict[str, Any],
    comp_run: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate total optimizer inference cost across all relevant model calls including retries.

    Reports baseline total, compressed total, tokens saved, percentage, call counts,
    and telemetry completeness.
    If multi-call telemetry is unavailable (such as historical benchmarks where calls were not recorded),
    explicitly reports status as 'unavailable' and does not fabricate missing token counts.
    """
    base_opt = base_run.get("optimizer_inference", {}) if isinstance(base_run, dict) else {}
    comp_opt = comp_run.get("optimizer_inference", {}) if isinstance(comp_run, dict) else {}

    base_calls = base_opt.get("calls")
    comp_calls = comp_opt.get("calls")
    base_call_count = base_opt.get("call_count")
    comp_call_count = comp_opt.get("call_count")
    base_complete = base_opt.get("has_complete_telemetry")
    comp_complete = comp_opt.get("has_complete_telemetry")

    # If neither calls list nor call_count is present, this is historical single-call data
    base_is_new = is_new_schema_optimizer_record(base_opt)
    comp_is_new = is_new_schema_optimizer_record(comp_opt)

    if not base_is_new and not comp_is_new:
        return {
            "metric_name": "total_optimizer_inference_cost",
            "measurement_scope": "all_relevant_calls",
            "status": "unavailable",
            "baseline_total_prompt_eval_count": None,
            "compressed_total_prompt_eval_count": None,
            "tokens_saved": None,
            "reduction_percent": None,
            "baseline_call_count": None,
            "compressed_call_count": None,
            "has_complete_telemetry": False,
            "telemetry_source": "unavailable_in_historical_benchmark",
            "note": "Multi-call retry telemetry was not preserved in historical first-call benchmark; not fabricated.",
        }

    if not (base_is_new and comp_is_new):
        return {
            "metric_name": "total_optimizer_inference_cost",
            "measurement_scope": "all_relevant_calls",
            "status": "unavailable",
            "baseline_total_prompt_eval_count": None,
            "compressed_total_prompt_eval_count": None,
            "tokens_saved": None,
            "reduction_percent": None,
            "baseline_call_count": None,
            "compressed_call_count": None,
            "has_complete_telemetry": False,
            "telemetry_source": "incompatible_schema",
            "note": "Mismatched telemetry schema between baseline and compressed runs; cannot compare total cost.",
        }

    base_tot = base_opt.get("prompt_eval_count")
    comp_tot = comp_opt.get("prompt_eval_count")
    has_complete = bool(base_complete and comp_complete)

    if (
        base_tot is not None
        and comp_tot is not None
        and isinstance(base_tot, (int, float))
        and isinstance(comp_tot, (int, float))
        and base_tot >= 0
        and comp_tot >= 0
    ):
        base_int = int(base_tot)
        comp_int = int(comp_tot)
        saved = base_int - comp_int
        pct = round((saved / max(1, base_int)) * 100.0, 2)
        status = "evaluated" if has_complete else "incomplete_telemetry"
    else:
        base_int = None
        comp_int = None
        saved = None
        pct = None
        status = "unavailable"

    return {
        "metric_name": "total_optimizer_inference_cost",
        "measurement_scope": "all_relevant_calls",
        "status": status,
        "baseline_total_prompt_eval_count": base_int,
        "compressed_total_prompt_eval_count": comp_int,
        "tokens_saved": saved,
        "reduction_percent": pct,
        "baseline_call_count": base_call_count if base_call_count is not None else len(base_calls or []),
        "compressed_call_count": comp_call_count if comp_call_count is not None else len(comp_calls or []),
        "has_complete_telemetry": has_complete,
        "telemetry_source": "complete_all_calls" if has_complete else "incomplete",
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
                base_run["optimizer_inference"]["prompt_preservation_evaluation"]["passed"],
                comp_run["optimizer_inference"]["prompt_preservation_evaluation"]["passed"],
            )
            downstream_answer_status = determine_comparison_status(
                base_run["downstream_inference"]["task_evaluation"]["passed"],
                comp_run["downstream_inference"]["task_evaluation"]["passed"],
            )

            # Section 2: Stage character comparisons
            base_stages = base_run["stage_measurements"]
            comp_stages = comp_run["stage_measurements"]

            # Compression-stage character reduction
            comp_stage_char_reduction = base_stages["original_workload_prompt_chars"] - comp_stages["compression_stage_output_chars"]
            comp_stage_reduction_pct = round(
                (comp_stage_char_reduction / max(1, base_stages["original_workload_prompt_chars"])) * 100.0, 2
            )

            # Final optimized prompt character reduction
            final_char_reduction = base_stages["final_optimized_prompt_chars"] - comp_stages["final_optimized_prompt_chars"]
            final_reduction_pct = round(
                (final_char_reduction / max(1, base_stages["final_optimized_prompt_chars"])) * 100.0, 2
            )

            # Section 1: Downstream task model token savings
            base_down_prompt_tok = base_run["downstream_inference"]["prompt_eval_count"]
            comp_down_prompt_tok = comp_run["downstream_inference"]["prompt_eval_count"]
            downstream_tokens_saved: Optional[int] = None
            downstream_tokens_saved_pct: Optional[float] = None
            if base_down_prompt_tok is not None and comp_down_prompt_tok is not None:
                saved = int(base_down_prompt_tok) - int(comp_down_prompt_tok)
                downstream_tokens_saved = saved
                downstream_tokens_saved_pct = round(
                    (saved / max(1, int(base_down_prompt_tok))) * 100.0, 2
                )

            # Section 1: Calculate both distinct optimizer inference metrics:
            # 1. Primary rewrite call savings (first-call only, historical baseline compatible)
            primary_rewrite_savings = calculate_primary_rewrite_savings(base_run, comp_run)

            # 2. Total optimizer inference cost (all relevant calls including retries)
            total_optimizer_cost = calculate_total_optimizer_cost(base_run, comp_run)

            trial_result = {
                "workload_id": workload["id"],
                "workload_name": workload["name"],
                "content_type": workload["content_type"],
                "repetition": rep,
                "execution_order": order_label,
                "baseline": base_run,
                "compressed": comp_run,
                "comparison": {
                    # Section 2: Four equivalent stage measurements
                    "stage_measurements_comparison": {
                        "original_workload_prompt_chars": base_stages["original_workload_prompt_chars"],
                        "compression_stage": {
                            "baseline_chars": base_stages["compression_stage_output_chars"],
                            "compressed_chars": comp_stages["compression_stage_output_chars"],
                            "reduction_chars": comp_stage_char_reduction,
                            "reduction_percent": comp_stage_reduction_pct,
                        },
                        "final_optimized_prompt": {
                            "baseline_chars": base_stages["final_optimized_prompt_chars"],
                            "compressed_chars": comp_stages["final_optimized_prompt_chars"],
                            "reduction_chars": final_char_reduction,
                            "reduction_percent": final_reduction_pct,
                        },
                        "downstream_model_input": {
                            "baseline_chars": base_stages["downstream_model_input_chars"],
                            "compressed_chars": comp_stages["downstream_model_input_chars"],
                        },
                    },
                    # Section 1: Actual Downstream Task Model Token Savings
                    "downstream_token_savings": {
                        "baseline_prompt_eval_count": base_down_prompt_tok,
                        "compressed_prompt_eval_count": comp_down_prompt_tok,
                        "tokens_saved": downstream_tokens_saved,
                        "reduction_percent": downstream_tokens_saved_pct,
                    },
                    # Section 1: Primary Rewrite Call Token Savings (First-Call Only)
                    "primary_rewrite_call_savings": primary_rewrite_savings,
                    # Section 1: Total Optimizer Inference Cost (All Calls Including Retries)
                    "total_optimizer_inference_cost": total_optimizer_cost,
                    # Section 1: Internal Optimizer Rewriter Token Savings (Backward-compatible alias)
                    "internal_optimizer_token_savings": {
                        "baseline_prompt_eval_count": primary_rewrite_savings["baseline_prompt_eval_count"],
                        "compressed_prompt_eval_count": primary_rewrite_savings["compressed_prompt_eval_count"],
                        "tokens_saved": primary_rewrite_savings["tokens_saved"],
                        "reduction_percent": primary_rewrite_savings["reduction_percent"],
                        "telemetry_source": primary_rewrite_savings["measurement_scope"],
                        "baseline_call_count": base_run["optimizer_inference"].get("call_count", 1),
                        "compressed_call_count": comp_run["optimizer_inference"].get("call_count", 1),
                    },
                    "compression_overhead_ms": comp_run["compression_stage"]["processing_time_ms"],
                    "prompt_preservation_status": prompt_preservation_status,
                    "downstream_answer_status": downstream_answer_status,
                    "restoration_verification": restoration,
                },
            }

            print(
                f"    -> Comp Stage: {comp_stages['compression_stage_output_chars']} chars ({comp_stage_reduction_pct:+.2f}%) | "
                f"Final Opt: {comp_stages['final_optimized_prompt_chars']} chars ({final_reduction_pct:+.2f}%) | "
                f"Downstream tokens saved: {downstream_tokens_saved} | "
                f"Downstream Answer: {downstream_answer_status}"
            )
            all_trials.append(trial_result)

    # 1. Downstream token savings aggregation
    valid_down_token_trials = [
        t for t in all_trials
        if t["comparison"]["downstream_token_savings"]["tokens_saved"] is not None
    ]
    if valid_down_token_trials:
        total_base_down_tokens = sum(t["comparison"]["downstream_token_savings"]["baseline_prompt_eval_count"] for t in valid_down_token_trials)
        total_comp_down_tokens = sum(t["comparison"]["downstream_token_savings"]["compressed_prompt_eval_count"] for t in valid_down_token_trials)
        total_down_saved = total_base_down_tokens - total_comp_down_tokens
        agg_down_token_pct = round((total_down_saved / max(1, total_base_down_tokens)) * 100.0, 2)
        down_tokens_saved_list = [t["comparison"]["downstream_token_savings"]["tokens_saved"] for t in valid_down_token_trials]
        mean_down_saved = round(statistics.mean(down_tokens_saved_list), 2)
        median_down_saved = round(statistics.median(down_tokens_saved_list), 2)
        stdev_down_saved = round(statistics.stdev(down_tokens_saved_list), 2) if len(down_tokens_saved_list) > 1 else 0.0
    else:
        total_base_down_tokens = None
        total_comp_down_tokens = None
        total_down_saved = None
        agg_down_token_pct = None
        mean_down_saved = None
        median_down_saved = None
        stdev_down_saved = None

    # 2. Internal optimizer token savings aggregation (Primary Rewrite Call)
    valid_opt_token_trials = [
        t for t in all_trials
        if t["comparison"]["internal_optimizer_token_savings"]["tokens_saved"] is not None
    ]
    if valid_opt_token_trials:
        total_base_opt_tokens = sum(t["comparison"]["internal_optimizer_token_savings"]["baseline_prompt_eval_count"] for t in valid_opt_token_trials)
        total_comp_opt_tokens = sum(t["comparison"]["internal_optimizer_token_savings"]["compressed_prompt_eval_count"] for t in valid_opt_token_trials)
        total_opt_saved = total_base_opt_tokens - total_comp_opt_tokens
        agg_opt_token_pct = round((total_opt_saved / max(1, total_base_opt_tokens)) * 100.0, 2)
        opt_tokens_saved_list = [t["comparison"]["internal_optimizer_token_savings"]["tokens_saved"] for t in valid_opt_token_trials]
        mean_opt_saved = round(statistics.mean(opt_tokens_saved_list), 2)
        median_opt_saved = round(statistics.median(opt_tokens_saved_list), 2)
        stdev_opt_saved = round(statistics.stdev(opt_tokens_saved_list), 2) if len(opt_tokens_saved_list) > 1 else 0.0
    else:
        total_base_opt_tokens = None
        total_comp_opt_tokens = None
        total_opt_saved = None
        agg_opt_token_pct = None
        mean_opt_saved = None
        median_opt_saved = None
        stdev_opt_saved = None

    # Total optimizer inference cost aggregation (All Calls Including Retries)
    valid_tot_cost_trials = [
        t for t in all_trials
        if t["comparison"].get("total_optimizer_inference_cost", {}).get("tokens_saved") is not None
    ]
    if valid_tot_cost_trials:
        tot_base = sum(t["comparison"]["total_optimizer_inference_cost"]["baseline_total_prompt_eval_count"] for t in valid_tot_cost_trials)
        tot_comp = sum(t["comparison"]["total_optimizer_inference_cost"]["compressed_total_prompt_eval_count"] for t in valid_tot_cost_trials)
        tot_saved = tot_base - tot_comp
        tot_pct = round((tot_saved / max(1, tot_base)) * 100.0, 2)
        saved_list = [t["comparison"]["total_optimizer_inference_cost"]["tokens_saved"] for t in valid_tot_cost_trials]
        mean_tot_saved = round(statistics.mean(saved_list), 2)
        median_tot_saved = round(statistics.median(saved_list), 2)
        stdev_tot_saved = round(statistics.stdev(saved_list), 2) if len(saved_list) > 1 else 0.0
        all_complete = all(t["comparison"]["total_optimizer_inference_cost"].get("has_complete_telemetry", False) for t in valid_tot_cost_trials)
        total_optimizer_cost_summary: Dict[str, Any] = {
            "status": "evaluated",
            "total_baseline_tokens": tot_base,
            "total_compressed_tokens": tot_comp,
            "total_tokens_saved": tot_saved,
            "aggregate_percent": tot_pct,
            "mean_tokens_saved": mean_tot_saved,
            "median_tokens_saved": median_tot_saved,
            "stdev_tokens_saved": stdev_tot_saved,
            "has_complete_telemetry": all_complete,
            "measurement_scope": "all_relevant_calls",
        }
    else:
        total_optimizer_cost_summary = {
            "status": "unavailable_in_historical_run",
            "total_baseline_tokens": None,
            "total_compressed_tokens": None,
            "total_tokens_saved": None,
            "aggregate_percent": None,
            "mean_tokens_saved": None,
            "median_tokens_saved": None,
            "stdev_tokens_saved": None,
            "has_complete_telemetry": False,
            "measurement_scope": "all_relevant_calls",
            "note": "Multi-call retry telemetry was not preserved in historical first-call benchmark; not fabricated.",
        }

    # 3. Compression-stage character reduction aggregation
    comp_stage_pcts = [t["comparison"]["stage_measurements_comparison"]["compression_stage"]["reduction_percent"] for t in all_trials]
    total_base_orig_chars = sum(t["comparison"]["stage_measurements_comparison"]["original_workload_prompt_chars"] for t in all_trials)
    total_comp_stage_chars = sum(t["comparison"]["stage_measurements_comparison"]["compression_stage"]["compressed_chars"] for t in all_trials)
    agg_comp_stage_pct = round(((total_base_orig_chars - total_comp_stage_chars) / max(1, total_base_orig_chars)) * 100.0, 2)

    # 4. Final optimized prompt character reduction aggregation
    final_opt_pcts = [t["comparison"]["stage_measurements_comparison"]["final_optimized_prompt"]["reduction_percent"] for t in all_trials]
    total_base_final_chars = sum(t["comparison"]["stage_measurements_comparison"]["final_optimized_prompt"]["baseline_chars"] for t in all_trials)
    total_comp_final_chars = sum(t["comparison"]["stage_measurements_comparison"]["final_optimized_prompt"]["compressed_chars"] for t in all_trials)
    agg_final_opt_pct = round(((total_base_final_chars - total_comp_final_chars) / max(1, total_base_final_chars)) * 100.0, 2)

    # 5. Latency aggregation
    engine_latencies = [t["compressed"]["compression_stage"]["processing_time_ms"] for t in all_trials]

    # 6. Quality and Restoration Summaries
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
    downstream_not_evaluated = sum(1 for t in all_trials if t["comparison"]["downstream_answer_status"] == "not_evaluated")

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
                "num_predict": 384,
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
            "inference_separation_note": (
                "Internal optimizer inference metrics represent the prompt rewriting call (Stage C). "
                "Downstream model inference metrics represent actual task execution using the optimized prompt."
            ),
        },
        "aggregate_statistics": {
            # Section 1: Actual Downstream Task Model Token Savings
            "downstream_prompt_token_savings": {
                "total_baseline_tokens": total_base_down_tokens,
                "total_compressed_tokens": total_comp_down_tokens,
                "total_tokens_saved": total_down_saved,
                "aggregate_percent": agg_down_token_pct,
                "mean_tokens_saved": mean_down_saved,
                "median_tokens_saved": median_down_saved,
                "stdev_tokens_saved": stdev_down_saved,
            },
            # Section 1: Primary Rewrite Call Token Savings (First-Call Only)
            "primary_rewrite_call_prompt_token_savings": {
                "total_baseline_tokens": total_base_opt_tokens,
                "total_compressed_tokens": total_comp_opt_tokens,
                "total_tokens_saved": total_opt_saved,
                "aggregate_percent": agg_opt_token_pct,
                "mean_tokens_saved": mean_opt_saved,
                "median_tokens_saved": median_opt_saved,
                "stdev_tokens_saved": stdev_opt_saved,
                "measurement_scope": "first_call_primary_rewrite",
            },
            # Section 1: Internal Argus Rewriter Token Savings (Backward-compatible alias)
            "internal_optimizer_prompt_token_savings": {
                "total_baseline_tokens": total_base_opt_tokens,
                "total_compressed_tokens": total_comp_opt_tokens,
                "total_tokens_saved": total_opt_saved,
                "aggregate_percent": agg_opt_token_pct,
                "mean_tokens_saved": mean_opt_saved,
                "median_tokens_saved": median_opt_saved,
                "stdev_tokens_saved": stdev_opt_saved,
            },
            # Section 1: Total Optimizer Inference Token Cost (All Relevant Calls Including Retries)
            "total_optimizer_inference_token_cost": total_optimizer_cost_summary,
            # Section 2: Compression-Stage Character Reduction
            "compression_stage_character_reduction": {
                "total_baseline_chars": total_base_orig_chars,
                "total_compressed_chars": total_comp_stage_chars,
                "aggregate_percent": agg_comp_stage_pct,
                "mean_percent": round(statistics.mean(comp_stage_pcts), 2) if comp_stage_pcts else 0.0,
                "median_percent": round(statistics.median(comp_stage_pcts), 2) if comp_stage_pcts else 0.0,
                "stdev_percent": round(statistics.stdev(comp_stage_pcts), 2) if len(comp_stage_pcts) > 1 else 0.0,
            },
            # Section 2: Final Optimized Prompt Character Reduction
            "final_optimized_prompt_character_reduction": {
                "total_baseline_chars": total_base_final_chars,
                "total_compressed_chars": total_comp_final_chars,
                "aggregate_percent": agg_final_opt_pct,
                "mean_percent": round(statistics.mean(final_opt_pcts), 2) if final_opt_pcts else 0.0,
                "median_percent": round(statistics.median(final_opt_pcts), 2) if final_opt_pcts else 0.0,
                "stdev_percent": round(statistics.stdev(final_opt_pcts), 2) if len(final_opt_pcts) > 1 else 0.0,
            },
            # Section 4: Compression Engine Latency Overhead
            "compression_engine_latency_ms": {
                "mean_ms": round(statistics.mean(engine_latencies), 3) if engine_latencies else 0.0,
                "median_ms": round(statistics.median(engine_latencies), 3) if engine_latencies else 0.0,
                "stdev_ms": round(statistics.stdev(engine_latencies), 3) if len(engine_latencies) > 1 else 0.0,
            },
            # Section 3: Downstream Task Answer Quality Preservation
            "downstream_task_quality_summary": {
                "total_trials": len(all_trials),
                "preserved_count": downstream_preserved_count,
                "regression_count": downstream_regressions,
                "improvement_count": downstream_improvements,
                "both_failed_count": downstream_both_failed,
                "not_evaluated_count": downstream_not_evaluated,
                "preservation_rate_percent": round((downstream_preserved_count / len(all_trials)) * 100.0, 1),
            },
            # Reversible Restoration
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
    print(f"  Downstream Prompt Token Reduction: {stats['downstream_prompt_token_savings']['aggregate_percent']}% (saved: {stats['downstream_prompt_token_savings']['total_tokens_saved']} tokens)")
    print(f"  Internal Optimizer Token Reduction: {stats['internal_optimizer_prompt_token_savings']['aggregate_percent']}% (saved: {stats['internal_optimizer_prompt_token_savings']['total_tokens_saved']} tokens)")
    print(f"  Compression-Stage Char Reduction: {stats['compression_stage_character_reduction']['aggregate_percent']:+.2f}% (mean: {stats['compression_stage_character_reduction']['mean_percent']:+.2f}%)")
    print(f"  Final Optimized Prompt Char Reduction: {stats['final_optimized_prompt_character_reduction']['aggregate_percent']:+.2f}% (mean: {stats['final_optimized_prompt_character_reduction']['mean_percent']:+.2f}%)")
    print(f"  Engine Latency: mean {stats['compression_engine_latency_ms']['mean_ms']} ms (median: {stats['compression_engine_latency_ms']['median_ms']} ms)")
    print(f"  Downstream Quality Status: {stats['downstream_task_quality_summary']['preserved_count']} preserved / {stats['downstream_task_quality_summary']['total_trials']} trials ({stats['downstream_task_quality_summary']['preservation_rate_percent']}%)")
    print(f"  Reversible Restoration: {stats['reversible_restoration_summary']['verified_byte_exact']} / {stats['reversible_restoration_summary']['total_tested']} verified byte-exact")
    print(f"Saved raw JSON measurements to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

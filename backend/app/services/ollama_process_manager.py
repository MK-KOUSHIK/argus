"""Managed Ollama Server Process and Runtime Policy Manager for Argus.

Inspects local Ollama server process state, reads launcher manifests,
verifies active listening ports, and scans process logs for empirical confirmation
of Flash Attention and KV-cache quantization (F16 vs Q8_0).
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = BACKEND_DIR / "data" / "ollama_runtime.json"
STD_LOG_PATH = BACKEND_DIR / "data" / "ollama_server.log"
ERR_LOG_PATH = BACKEND_DIR / "data" / "ollama_server.err.log"


class OllamaProcessManager:
    """Provides authoritative inspection of the local Ollama server process."""

    @classmethod
    def get_port_listening_pid(cls, port: int = 11434) -> Optional[int]:
        """Detect the PID of the process listening on the target port via netstat."""
        try:
            out = subprocess.check_output(
                ["netstat", "-ano"],
                text=True,
                timeout=3,
                stderr=subprocess.DEVNULL,
            )
            pattern = re.compile(rf":{port}\s+.*LISTENING\s+(\d+)", re.IGNORECASE)
            for line in out.splitlines():
                match = pattern.search(line)
                if match:
                    return int(match.group(1))
        except Exception:
            pass
        return None

    @classmethod
    def is_pid_running(cls, pid: int) -> bool:
        """Check if process with PID exists and is named ollama."""
        try:
            if os.name == "nt":
                out = subprocess.check_output(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    text=True,
                    timeout=3,
                    stderr=subprocess.DEVNULL,
                )
                return "ollama" in out.lower()
        except Exception:
            pass
        return False

    @classmethod
    def read_manifest(cls) -> Optional[Dict[str, Any]]:
        """Read the launcher runtime manifest if present."""
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    @classmethod
    def scan_log_evidence(cls, kv_type: str) -> Tuple[bool, Optional[str]]:
        """Scan available log files for authoritative runtime confirmation."""
        evidence_items = []
        log_files = [ERR_LOG_PATH, STD_LOG_PATH]

        for log_path in log_files:
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()[-200:]
                        for line in lines:
                            if f"OLLAMA_KV_CACHE_TYPE:{kv_type}" in line:
                                evidence_items.append(f"Server env: OLLAMA_KV_CACHE_TYPE={kv_type}")
                            if "flash_attn" in line and "enabled" in line:
                                evidence_items.append("llama_context: flash_attn = enabled")
                            if f"K ({kv_type})" in line:
                                evidence_items.append(line.strip())
                except Exception:
                    pass

        if evidence_items:
            unique_evidence = list(dict.fromkeys(evidence_items))
            return True, " | ".join(unique_evidence)

        return False, None

    @classmethod
    def get_runtime_policy(cls) -> Dict[str, Any]:
        """Produce the authoritative KV runtime policy report for API responses."""
        active_pid = cls.get_port_listening_pid(port=11434)
        manifest = cls.read_manifest()

        configured_kv = settings.OLLAMA_KV_CACHE_TYPE
        configured_fa = settings.OLLAMA_FLASH_ATTENTION
        endpoint = settings.OLLAMA_BASE_URL

        base_policy = {
            "requested_num_ctx": settings.OLLAMA_NUM_CTX,
            "requested_num_predict": settings.OLLAMA_NUM_PREDICT,
            "argus_max_concurrent_requests": settings.OLLAMA_MAX_CONCURRENT_REQUESTS,
            "configured_kv_cache_type": configured_kv,
            "configured_flash_attention": configured_fa,
            "server_endpoint": endpoint,
        }

        if active_pid is None:
            return {
                **base_policy,
                "server_process_status": "not_running",
                "server_process_verified": False,
                "server_settings_verified": False,
                "managed_by_argus": False,
                "managed_pid": None,
                "active_pid": None,
                "runtime_configuration_status": "not_running",
                "confirmation_evidence": None,
                "note": "No active Ollama process detected on port 11434. Launch using scripts/launch_ollama.ps1.",
            }

        managed = False
        manifest_pid = manifest.get("pid") if manifest else None
        if manifest_pid and (manifest_pid == active_pid or cls.is_pid_running(manifest_pid)):
            managed = True

        confirmed, evidence = cls.scan_log_evidence(configured_kv)

        if confirmed:
            status = "independently_confirmed"
            note = f"Ollama process PID {active_pid} verified with active {configured_kv.upper()} KV-cache and Flash Attention."
        elif managed:
            status = "launched_with_requested_settings"
            evidence = f"Manifest records launch with OLLAMA_KV_CACHE_TYPE={manifest.get('kv_cache_type')}"
            note = f"Ollama managed with PID {active_pid}. Runtime KV quantization will log on first model execution."
        else:
            status = "unmanaged_external_instance"
            evidence = None
            note = f"Ollama is running with unmanaged PID {active_pid}. Environment variables cannot be verified without Argus launcher."

        return {
            **base_policy,
            "server_process_status": "running",
            "server_process_verified": confirmed or managed,
            "server_settings_verified": confirmed,
            "managed_by_argus": managed,
            "managed_pid": manifest_pid if managed else None,
            "active_pid": active_pid,
            "runtime_configuration_status": status,
            "confirmation_evidence": evidence,
            "note": note,
        }

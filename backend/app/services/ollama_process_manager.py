"""Managed Ollama Server Process and Runtime Policy Manager for Argus.

Inspects local Ollama server process state, reads launcher manifests,
verifies active listening ports, and scans process logs for empirical confirmation
of Flash Attention and KV-cache quantization (F16 vs Q8_0) tied to the active session.
"""

import ctypes
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    def get_process_start_time(cls, pid: int) -> Optional[datetime]:
        """Query process creation time via Win32 GetProcessTimes or procfs."""
        if os.name == "nt":
            try:
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if handle:
                    try:
                        creation_time = ctypes.c_int64()
                        exit_time = ctypes.c_int64()
                        kernel_time = ctypes.c_int64()
                        user_time = ctypes.c_int64()
                        if ctypes.windll.kernel32.GetProcessTimes(
                            handle,
                            ctypes.byref(creation_time),
                            ctypes.byref(exit_time),
                            ctypes.byref(kernel_time),
                            ctypes.byref(user_time),
                        ):
                            # FILETIME to unix timestamp
                            epoch_diff = 116444736000000000
                            ts = (creation_time.value - epoch_diff) / 10_000_000.0
                            return datetime.fromtimestamp(ts, tz=timezone.utc)
                    finally:
                        ctypes.windll.kernel32.CloseHandle(handle)
            except Exception:
                pass
        else:
            try:
                stat = os.stat(f"/proc/{pid}")
                return datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            except Exception:
                pass
        return None

    @classmethod
    def is_pid_running_ollama(cls, pid: int) -> bool:
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
            else:
                out = subprocess.check_output(["ps", "-p", str(pid), "-o", "comm="], text=True, timeout=3)
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
    def scan_log_evidence(
        cls,
        target_kv: str,
        target_fa: bool = True,
        active_process_start_time: Optional[datetime] = None,
        std_log_path: Optional[Path] = None,
        err_log_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Scan available log files for independent runtime confirmation of KV-cache and Flash Attention.

        Ensures log evidence is tied to the active session and rejects stale logs.
        """
        log_files = [
            err_log_path or ERR_LOG_PATH,
            std_log_path or STD_LOG_PATH,
        ]

        kv_evidence_list: List[str] = []
        fa_evidence_list: List[str] = []
        stale_detected = False

        for log_path in log_files:
            if not log_path.exists():
                continue

            try:
                # 1. Stale log detection: If log file mtime is older than process start time, skip it
                if active_process_start_time is not None:
                    mtime = log_path.stat().st_mtime
                    if mtime < (active_process_start_time.timestamp() - 5.0):
                        stale_detected = True
                        continue

                raw_bytes = log_path.read_bytes()
                if raw_bytes.startswith(b"\xff\xfe") or (len(raw_bytes) > 2 and raw_bytes[1:2] == b"\x00"):
                    text = raw_bytes.decode("utf-16le", errors="ignore")
                elif raw_bytes.startswith(b"\xfe\xff"):
                    text = raw_bytes.decode("utf-16be", errors="ignore")
                else:
                    text = raw_bytes.decode("utf-8", errors="ignore")

                lines = text.splitlines()

                for line in lines:
                    # Optional timestamp filter if line has standard time=ISO
                    time_match = re.search(r"time=(\d{4}-\d{2}-\d{2}T[^\s]+)", line)
                    if time_match and active_process_start_time is not None:
                        try:
                            # Ollama format: 2026-11-03T21:14:45.073+05:30
                            iso_str = time_match.group(1)
                            # Python 3.11 fromisoformat handles +05:30 offsets
                            line_time = datetime.fromisoformat(iso_str)
                            if line_time < active_process_start_time:
                                continue  # Ignore line from prior session
                        except Exception:
                            pass

                    # Independent KV cache quantization detection
                    if f"ollama_kv_cache_type:{target_kv.lower()}" in line.lower() or f"ollama_kv_cache_type={target_kv.lower()}" in line.lower():
                        kv_evidence_list.append(f"Server config: OLLAMA_KV_CACHE_TYPE={target_kv}")
                    if f"k ({target_kv.lower()})" in line.lower() or f"k({target_kv.lower()})" in line.lower():
                        kv_evidence_list.append(line.strip())

                    # Independent Flash Attention detection
                    if "flash_attn" in line.lower() and "enabled" in line.lower():
                        fa_evidence_list.append("llama_context: flash_attn = enabled")
                    elif target_fa and ("ollama_flash_attention:true" in line.lower() or "ollama_flash_attention:1" in line.lower() or "ollama_flash_attention=1" in line.lower()):
                        fa_evidence_list.append("Server config: OLLAMA_FLASH_ATTENTION=1")
                    elif not target_fa and ("flash_attn" in line.lower() and "disabled" in line.lower()):
                        fa_evidence_list.append("llama_context: flash_attn = disabled")

            except Exception:
                pass

        kv_confirmed = len(kv_evidence_list) > 0
        fa_confirmed = len(fa_evidence_list) > 0

        unique_kv = list(dict.fromkeys(kv_evidence_list))
        unique_fa = list(dict.fromkeys(fa_evidence_list))

        combined_items = unique_kv + unique_fa
        combined_evidence = " | ".join(combined_items) if combined_items else None

        return {
            "kv_confirmed": kv_confirmed,
            "kv_evidence": " | ".join(unique_kv) if unique_kv else None,
            "flash_attention_confirmed": fa_confirmed,
            "flash_attention_evidence": " | ".join(unique_fa) if unique_fa else None,
            "all_settings_confirmed": kv_confirmed and fa_confirmed,
            "is_stale_log": stale_detected and not (kv_confirmed or fa_confirmed),
            "combined_evidence": combined_evidence,
        }

    @classmethod
    def get_runtime_policy(cls) -> Dict[str, Any]:
        """Produce the authoritative KV runtime policy report for API responses.

        Strictly distinguishes:
          - configured: in Argus settings (.env)
          - launched: manifest indicates Argus launched it with requested env vars
          - observed: process/socket verified running on port 11434
          - independently_confirmed: actual active-session server log evidence proves settings
        """
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

        # 1. State: Not Running
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
                "kv_cache_confirmed": False,
                "flash_attention_confirmed": False,
                "note": "No active Ollama process detected on port 11434. Launch using scripts/launch_ollama.ps1.",
            }

        # Verify active process is indeed ollama
        is_ollama = cls.is_pid_running_ollama(active_pid)
        proc_start_time = cls.get_process_start_time(active_pid)

        # 2. State: Managed Verification
        # Check if manifest belongs to this active process and session
        managed = False
        manifest_stale = False
        manifest_pid = manifest.get("pid") if manifest else None

        if manifest_pid:
            if manifest_pid == active_pid and is_ollama:
                # Verify manifest start_time against process creation time
                manifest_time_str = manifest.get("start_time")
                if manifest_time_str and proc_start_time:
                    try:
                        manifest_time = datetime.fromisoformat(manifest_time_str)
                        # Process start time should be within 10 seconds of manifest start time
                        time_delta_sec = abs((proc_start_time - manifest_time).total_seconds())
                        if time_delta_sec <= 10.0:
                            managed = True
                        else:
                            manifest_stale = True
                    except Exception:
                        managed = True
                else:
                    managed = True
            else:
                manifest_stale = True

        # 3. State: Independent Active Session Log Verification
        evidence_dict = cls.scan_log_evidence(
            target_kv=configured_kv,
            target_fa=configured_fa,
            active_process_start_time=proc_start_time,
        )

        kv_confirmed = evidence_dict["kv_confirmed"]
        fa_confirmed = evidence_dict["flash_attention_confirmed"]
        all_confirmed = evidence_dict["all_settings_confirmed"]
        evidence = evidence_dict["combined_evidence"]

        if all_confirmed:
            status = "independently_confirmed"
            note = f"Ollama process PID {active_pid} verified with active {configured_kv.upper()} KV-cache and Flash Attention."
        elif kv_confirmed and not fa_confirmed:
            status = "partially_confirmed_kv_only"
            note = f"Ollama process PID {active_pid}: {configured_kv.upper()} KV-cache confirmed in logs; Flash Attention not yet observed."
        elif fa_confirmed and not kv_confirmed:
            status = "partially_confirmed_fa_only"
            note = f"Ollama process PID {active_pid}: Flash Attention confirmed in logs; {configured_kv.upper()} KV-cache not yet observed."
        elif managed:
            status = "launched_with_requested_settings"
            evidence = f"Manifest records launch with OLLAMA_KV_CACHE_TYPE={manifest.get('kv_cache_type')}"
            note = f"Ollama managed with PID {active_pid}. Runtime KV quantization will log on first model execution."
        elif manifest_stale:
            status = "unmanaged_external_instance_stale_manifest"
            evidence = None
            note = f"Ollama is running with unmanaged PID {active_pid} (manifest has stale PID {manifest_pid})."
        else:
            status = "unmanaged_external_instance"
            evidence = None
            note = f"Ollama is running with unmanaged PID {active_pid}. Environment variables cannot be verified without Argus launcher."

        return {
            **base_policy,
            "server_process_status": "running" if is_ollama else "port_occupied_by_other_process",
            "server_process_verified": is_ollama and (all_confirmed or managed),
            "server_settings_verified": all_confirmed,
            "managed_by_argus": managed,
            "manifest_stale": manifest_stale,
            "managed_pid": manifest_pid if managed else None,
            "active_pid": active_pid,
            "runtime_configuration_status": status,
            "confirmation_evidence": evidence,
            "kv_cache_confirmed": kv_confirmed,
            "flash_attention_confirmed": fa_confirmed,
            "note": note,
        }

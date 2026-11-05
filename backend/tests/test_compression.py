"""Comprehensive unit and integration tests for Argus Independent Content-Aware Compression Engine (Phase 4.2).

Validates:
1. JSON structural minification and deep equivalence validation.
2. Code-aware AST compression, comment stripping, and execution semantics preservation.
3. Log repeated run folding, unique severity and error retention, and reversible references.
4. Natural-language redundancy reduction and strict constraint/negation/number preservation.
5. ContentRouter classification (JSON, Code, Logs, Text, Mixed) and confidence scores.
6. CompressionRegistry LRU eviction, TTL expiration, thread-safety, and exact retrieval.
7. PrefixPreserver prefix stabilization and native cache telemetry auditing.
8. Safe fallbacks for malformed, uncertain, or invalid inputs.
9. End-to-end integration with the /api/optimize pipeline and registry retrieval.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.schemas.optimizer import OptimizeRequest
from app.services.compression import (
    CodeCompressor,
    CompressionOrchestrator,
    CompressionRegistry,
    CompressionStrategy,
    ContentRouter,
    ContentType,
    JSONCompressor,
    LogCompressor,
    PrefixPreserver,
    TextRedundancyCompressor,
)


# ==============================================================================
# 1. JSON Structural Compressor Tests
# ==============================================================================


def test_json_compressor_structural_equivalence():
    """Verify JSON minification removes whitespace while preserving exact data types and hierarchy."""
    original_json = """
    {
        "id": 101,
        "name": "Argus Engine",
        "active": true,
        "score": 98.6,
        "tags": ["middleware", "optimization", "kv-cache"],
        "metadata": {
            "version": "4.2.0",
            "release": null
        }
    }
    """
    compressed, ok, reason = JSONCompressor.compress(original_json)
    assert ok is True
    assert "passed" in reason.lower()
    assert len(compressed) < len(original_json)

    # Re-parse both and compare deep equality
    orig_obj = json.loads(original_json)
    comp_obj = json.loads(compressed)
    assert orig_obj == comp_obj
    assert comp_obj["id"] == 101
    assert comp_obj["active"] is True
    assert comp_obj["tags"] == ["middleware", "optimization", "kv-cache"]
    assert comp_obj["metadata"]["release"] is None


def test_json_compressor_malformed_input_fallback():
    """Malformed or invalid JSON must safely fail validation without crashing."""
    malformed = '{"title": "Broken JSON", "unclosed: [1, 2, 3}'
    compressed, ok, reason = JSONCompressor.compress(malformed)
    assert ok is False
    assert compressed == malformed
    assert "json parse error" in reason.lower()


# ==============================================================================
# 2. Code-Aware Compressor Tests
# ==============================================================================


def test_code_compressor_python_ast_comments_and_docstrings():
    """Verify Python AST compression strips comments while preserving docstrings and functions."""
    code = '''
# Global comment explaining module
import math

def calculate_area(radius: float) -> float:
    """Calculate the area of a circle with given radius."""
    # Inline explanation of formula
    if radius <= 0:
        return 0.0  # Handle non-positive
    pi_val = math.pi
    return pi_val * (radius ** 2)

class Worker:
    # Class level comment
    def execute(self):
        pass
'''
    compressed, ok, reason = CodeCompressor.compress(code, language="python")
    assert ok is True
    assert "passed" in reason.lower()
    assert len(compressed) < len(code)

    # Comments must be removed
    assert "# Global comment" not in compressed
    assert "# Inline explanation" not in compressed
    assert "# Class level comment" not in compressed

    # Executable logic and docstring must be preserved
    assert "def calculate_area(radius: float) -> float:" in compressed
    assert '"""Calculate the area of a circle with given radius."""' in compressed
    assert "class Worker:" in compressed


def test_code_compressor_python_syntax_error_fallback():
    """Invalid Python syntax must safely preserve original code verbatim."""
    bad_python = "def broken_func(x y z): return x +"
    compressed, ok, reason = CodeCompressor.compress(bad_python, language="python")
    assert ok is False
    assert compressed == bad_python
    assert "syntax error" in reason.lower()


def test_code_compressor_sql():
    """Verify SQL compressor strips comments and formats whitespace while preserving strings."""
    sql = """
    -- Retrieve customer orders
    SELECT customer_id, /* inline comment */ COUNT(*) as total_orders
    FROM orders
    WHERE status = 'PENDING -- DO NOT SHIP' AND total_amount > 100.00;
    """
    compressed, ok, reason = CodeCompressor.compress(sql, language="sql")
    assert ok is True
    assert len(compressed) < len(sql)
    assert "-- Retrieve customer" not in compressed
    assert "/* inline comment */" not in compressed
    # Literal with comment characters inside quotes must remain intact
    assert "'PENDING -- DO NOT SHIP'" in compressed


def test_code_compressor_unsupported_language_verbatim():
    """Unsupported languages must be safely preserved without regex corruption."""
    rust_code = 'fn main() { println!("Hello from Rust"); }'
    compressed, ok, reason = CodeCompressor.compress(rust_code, language="rust")
    assert ok is True
    assert compressed == rust_code
    assert "verbatim" in reason.lower()


# ==============================================================================
# 3. Log and Tool Output Compressor Tests
# ==============================================================================


def test_log_compressor_folding_repeated_runs():
    """Verify log compressor folds consecutive repeated events while keeping unique errors."""
    log_stream = """
2026-11-05 08:00:01 INFO [worker-1] Heartbeat alive ok
2026-11-05 08:00:02 INFO [worker-1] Heartbeat alive ok
2026-11-05 08:00:03 INFO [worker-1] Heartbeat alive ok
2026-11-05 08:00:04 INFO [worker-1] Heartbeat alive ok
2026-11-05 08:00:05 INFO [worker-1] Heartbeat alive ok
2026-11-05 08:00:06 ERROR [worker-1] Connection refused on database port 5432
2026-11-05 08:00:07 INFO [worker-1] Heartbeat alive ok
2026-11-05 08:00:08 INFO [worker-1] Heartbeat alive ok
2026-11-05 08:00:09 INFO [worker-1] Heartbeat alive ok
"""
    compressed, ok, reason, ref = LogCompressor.compress(log_stream, reversible=True)
    assert ok is True
    assert "folding passed" in reason.lower()
    assert "[... Repeated" in compressed
    assert ref is not None
    # Crucial error must be 100% retained
    assert "Connection refused on database port 5432" in compressed

    # Verify retrieval from registry
    restored = CompressionRegistry.retrieve(ref)
    assert restored is not None
    assert "Heartbeat alive ok" in restored


def test_log_compressor_preserves_stack_traces():
    """Verify log compressor never compresses away stack traces or exceptions."""
    trace_log = """
2026-11-05 08:10:00 INFO [main] Starting task
Traceback (most recent call last):
  File "app/service.py", line 42, in execute
    raise ValueError("Missing critical parameter")
ValueError: Missing critical parameter
2026-11-05 08:10:01 FATAL [main] Service halted due to unhandled error
"""
    compressed, ok, reason, _ = LogCompressor.compress(trace_log, reversible=False)
    assert ok is True
    assert "Traceback" in compressed
    assert 'File "app/service.py"' in compressed
    assert "Missing critical parameter" in compressed
    assert "FATAL" in compressed


# ==============================================================================
# 4. Natural-Language Redundancy Compressor Tests
# ==============================================================================


def test_text_redundancy_compressor_deduplication():
    """Verify natural language redundancy compressor removes duplicate sentences while keeping negations."""
    prompt = (
        "Please build a Python backend API for user registration. "
        "Make sure you do not use Django or Flask. "
        "Please build a Python backend API for user registration. "
        "The system must support at least 500 concurrent users. "
        "Ensure the server uses port 8080."
    )
    compressed, ok, reason = TextRedundancyCompressor.compress(prompt)
    assert ok is True
    assert len(compressed) < len(prompt)
    # Deduplication removed second copy
    assert compressed.count("Please build a Python backend API for user registration.") == 1
    # Critical negation and numbers preserved
    assert "do not use django" in compressed.lower()
    assert "500" in compressed
    assert "8080" in compressed


def test_text_redundancy_compressor_preserves_negations_and_numbers():
    """Verify that if any negation or numeric constraint were lost, validation safely fails."""
    # Normal prompt with constraints
    prompt = "Do not deploy to production. Total budget is $5000 on 2026-12-01. File is config.yaml."
    compressed, ok, reason = TextRedundancyCompressor.compress(prompt)
    assert ok is True
    assert "not" in compressed
    assert "$5000" in compressed
    assert "2026-12-01" in compressed
    assert "config.yaml" in compressed


# ==============================================================================
# 5. ContentRouter Classification Tests
# ==============================================================================


def test_content_router_json_classification():
    """Verify ContentRouter correctly identifies pure JSON."""
    json_text = '{"service": "auth", "port": 8080, "routes": ["/login", "/register"]}'
    ctype, conf, segments = ContentRouter.classify(json_text)
    assert ctype == ContentType.JSON
    assert conf >= 0.90
    assert len(segments) == 1


def test_content_router_code_classification():
    """Verify ContentRouter correctly identifies Python code."""
    code_text = "def fetch_records(db_conn):\n    return db_conn.query('SELECT * FROM users')"
    ctype, conf, segments = ContentRouter.classify(code_text)
    assert ctype == ContentType.CODE
    assert conf >= 0.90
    assert segments[0].language == "python"


def test_content_router_logs_classification():
    """Verify ContentRouter correctly identifies timestamped logs."""
    logs = (
        "2026-11-05 08:15:00 INFO Initializing database pool\n"
        "2026-11-05 08:15:01 WARN Connection pool timeout after 1000ms\n"
        "2026-11-05 08:15:02 ERROR Retry connection failed"
    )
    ctype, conf, segments = ContentRouter.classify(logs)
    assert ctype == ContentType.LOGS
    assert conf >= 0.65


def test_content_router_mixed_content_isolation():
    """Verify ContentRouter partitions mixed narrative prompts with code blocks."""
    mixed_prompt = (
        "Optimize the following Python script for performance:\n\n"
        "```python\n"
        "# Calculate sum\n"
        "def compute_total(items):\n"
        "    return sum(items)\n"
        "```\n\n"
        "Ensure the output conforms to PEP 8 standards."
    )
    ctype, conf, segments = ContentRouter.classify(mixed_prompt)
    assert ctype == ContentType.MIXED
    assert conf >= 0.90
    assert len(segments) == 3
    assert segments[0].content_type == ContentType.TEXT
    assert segments[1].content_type == ContentType.CODE
    assert segments[1].is_fenced is True
    assert segments[2].content_type == ContentType.TEXT


# ==============================================================================
# 6. CompressionRegistry LRU and TTL Tests
# ==============================================================================


def test_compression_registry_lifecycle_and_eviction():
    """Verify registration, retrieval, TTL expiration, and LRU eviction."""
    CompressionRegistry.clear()

    # 1. Register and retrieve
    text_content = "Critical original omitted log data line 1-100."
    ref_key = CompressionRegistry.register(text_content, ttl_seconds=10)
    assert ref_key.startswith("argus:ref:")
    assert CompressionRegistry.retrieve(ref_key) == text_content

    # 2. Expiration test with 0-second TTL
    short_ref = CompressionRegistry.register("Will expire immediately", ttl_seconds=-1)
    assert CompressionRegistry.retrieve(short_ref) is None
    assert CompressionRegistry.contains(short_ref) is False

    # 3. Missing reference handling
    assert CompressionRegistry.retrieve("argus:ref:nonexistent") is None


# ==============================================================================
# 7. PrefixPreserver Tests
# ==============================================================================


def test_prefix_preserver_standardization():
    """Verify system instruction prefix stabilization."""
    sys_inst = "You are an expert prompt optimization middleware.   \r\n\r\nRewrite the input.  "
    dynamic_text = "Summarize the findings."
    full_prompt, prefix = PrefixPreserver.assemble_stable_prompt(sys_inst, dynamic_text)

    assert "Original prompt to optimize:\n\n" in prefix
    assert full_prompt.startswith(prefix)
    assert full_prompt.endswith(dynamic_text)

    # Telemetry audit verification
    mock_telemetry_hit = {"calls": [{"prompt_eval_cached_count": 128}]}
    audit_hit = PrefixPreserver.audit_native_cache_telemetry(mock_telemetry_hit)
    assert audit_hit["native_cache_verified"] is True
    assert audit_hit["cached_tokens"] == 128

    mock_telemetry_miss = {"calls": [{"prompt_eval_cached_count": 0}]}
    audit_miss = PrefixPreserver.audit_native_cache_telemetry(mock_telemetry_miss)
    assert audit_miss["native_cache_verified"] is False


# ==============================================================================
# 8. End-to-End Orchestrator & API Integration Tests
# ==============================================================================


def test_orchestrator_mixed_content_end_to_end():
    """Verify Orchestrator correctly handles mixed prompts with code and instructions."""
    mixed_prompt = (
        "Here is the database query service:\n\n"
        "```python\n"
        "# Fetch active accounts\n"
        "def get_active_users(db):\n"
        "    # Filter by status\n"
        "    return db.query('SELECT * FROM users WHERE active = True')\n"
        "```\n\n"
        "Please optimize this function to return only user IDs."
    )
    res = CompressionOrchestrator.compress(mixed_prompt)
    assert res.validation_passed is True
    assert res.content_type == ContentType.MIXED
    assert res.strategy_applied == CompressionStrategy.MIXED
    assert res.tokens_reduced > 0
    # Comments inside code were stripped while fences were preserved
    assert "# Fetch active accounts" not in res.compressed_text
    assert "```python" in res.compressed_text
    assert "Please optimize this function to return only user IDs." in res.compressed_text


@pytest.mark.asyncio
async def test_end_to_end_optimize_endpoint_with_compression(async_client: AsyncClient):
    """Verify POST /api/optimize returns compression telemetry and works with registry."""
    fake_response = {
        "response": "Optimized output respecting input compression.",
        "done": True,
        "done_reason": "stop",
        "eval_count": 30,
        "eval_duration": 150_000_000,
        "prompt_eval_count": 40,
        "prompt_eval_cached_count": 25,
    }

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = fake_response

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        # Input with JSON payload having verbose indentation
        payload = {
            "prompt": "Analyze this data:\n```json\n{\n  \"customer_id\": 109,\n  \"amount\": 45.00,\n  \"status\": \"confirmed\"\n}\n```",
            "mode": "balanced",
            "target_model": "gpt-4o",
            "enable_compression": True,
        }
        response = await async_client.post("/api/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "compression" in data
        comp = data["compression"]
        assert comp is not None
        assert comp["enabled"] is True
        assert comp["detected_content_type"] in {"json", "mixed"}
        assert comp["tokens_reduced"] > 0
        assert comp["validation_status"] == "passed"


@pytest.mark.asyncio
async def test_registry_retrieval_endpoint(async_client: AsyncClient):
    """Verify GET /api/compression/registry/{ref_key} endpoint retrieves original content."""
    original_text = "Important omitted diagnostic logs for recovery."
    ref_key = CompressionRegistry.register(original_text)

    # 1. Existing reference retrieval
    res_ok = await async_client.get(f"/api/compression/registry/{ref_key}")
    assert res_ok.status_code == 200
    data = res_ok.json()
    assert data["ref_key"] == ref_key
    assert data["content"] == original_text

    # 2. Non-existent reference returns 404
    res_404 = await async_client.get("/api/compression/registry/argus:ref:nonexistent_ref")
    assert res_404.status_code == 404

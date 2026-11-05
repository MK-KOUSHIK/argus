"""Comprehensive Unit and Integration Tests for Argus Local-First Model Routing (Phase 5).

Covers:
- Dynamic local model discovery (/api/tags, /api/ps, /api/show)
- Empty registry & Ollama connection failure handling
- Missing metadata handling (zero fabrication)
- Weak-model vs strong-model deterministic selection
- Complexity boundary cases
- Resource-constrained fallback (VRAM & RAM pressure)
- Insufficient hardware telemetry handling
- Manual model override & uninstalled model rejection (HTTP 400)
- Automatic routing enabled vs disabled
- Actual routed model passed to Ollama /api/generate
- Retry model consistency
- Routing telemetry validation in OptimizeResponse
- Routing APIs (/api/models, /api/route)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.schemas.optimizer import OptimizeRequest, OptimizeResponse
from app.schemas.routing import (
    ModelCatalogResponse,
    ModelTier,
    RouteRequest,
    RoutingDecisionTelemetry,
    RoutingStatus,
)
from app.services.routing.models import DiscoveredModel
from app.services.routing.registry import LocalModelRegistry
from app.services.routing.router import (
    ModelNotInstalledError,
    ModelRouter,
    ModelRouterError,
)
from app.services.ollama_service import OllamaService
from app.services.task_analyzer import TaskAnalyzer


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure registry state is reset between tests."""
    LocalModelRegistry.reset()
    yield
    LocalModelRegistry.reset()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# --- 1. Dynamic Model Registry Discovery Tests ---


@pytest.mark.asyncio
async def test_dynamic_discovery_success():
    """Verify registry parses /api/tags and /api/ps, populating models with correct tiers."""
    tags_payload = {
        "models": [
            {
                "name": "qwen2.5:1.5b",
                "model": "qwen2.5:1.5b",
                "size": 986061892,
                "digest": "sha256-1.5b",
                "modified_at": "2026-10-21T11:13:39Z",
                "details": {
                    "parameter_size": "1.5B",
                    "quantization_level": "Q4_K_M",
                    "family": "qwen2",
                    "context_length": 32768,
                },
                "capabilities": ["completion"],
            },
            {
                "name": "qwen2.5:7b",
                "model": "qwen2.5:7b",
                "size": 4683087332,
                "digest": "sha256-7b",
                "modified_at": "2026-10-19T22:27:18Z",
                "details": {
                    "parameter_size": "7.6B",
                    "quantization_level": "Q4_K_M",
                    "family": "qwen2",
                    "context_length": 32768,
                },
                "capabilities": ["completion"],
            },
        ]
    }
    ps_payload = {
        "models": [
            {
                "name": "qwen2.5:1.5b",
                "size_vram": 1258291200,
                "expires_at": "2026-11-05T22:00:00Z",
            }
        ]
    }

    async def mock_get(url, *args, **kwargs):
        if url.endswith("/api/tags"):
            return MagicMock(status_code=200, json=lambda: tags_payload)
        if url.endswith("/api/ps"):
            return MagicMock(status_code=200, json=lambda: ps_payload)
        return MagicMock(status_code=404)

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        connected, err = await LocalModelRegistry.refresh(force=True)
        assert connected is True
        assert err is None

        models = await LocalModelRegistry.get_all_models()
        assert len(models) == 2

        m15 = await LocalModelRegistry.get_model("qwen2.5:1.5b")
        assert m15 is not None
        assert m15.tier == ModelTier.SMALL
        assert m15.parameter_size == "1.5B"
        assert m15.is_loaded is True
        assert m15.size_vram == 1258291200

        m7 = await LocalModelRegistry.get_model("qwen2.5:7b")
        assert m7 is not None
        assert m7.tier == ModelTier.MEDIUM
        assert m7.parameter_size == "7.6B"
        assert m7.is_loaded is False


@pytest.mark.asyncio
async def test_discovery_empty_registry():
    """Verify registry gracefully handles empty model response."""
    tags_payload = {"models": []}
    with patch("httpx.AsyncClient.get", return_value=MagicMock(status_code=200, json=lambda: tags_payload)):
        connected, err = await LocalModelRegistry.refresh(force=True)
        assert connected is True
        assert err is None
        models = await LocalModelRegistry.get_all_models()
        assert len(models) == 0


@pytest.mark.asyncio
async def test_discovery_ollama_unavailable():
    """Verify registry handles connection error without crashing."""
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
        connected, err = await LocalModelRegistry.refresh(force=True)
        assert connected is False
        assert "Cannot connect to Ollama" in str(err)
        models = await LocalModelRegistry.get_all_models()
        assert len(models) == 0


@pytest.mark.asyncio
async def test_discovery_missing_metadata_no_fabrication():
    """Verify missing metadata fields are recorded as None without fabrication."""
    tags_payload = {
        "models": [
            {
                "name": "custom-unknown:latest",
                "size": 2000000000,
                "digest": "sha256-unknown",
                "modified_at": "2026-10-20T00:00:00Z",
            }
        ]
    }
    with patch("httpx.AsyncClient.get", return_value=MagicMock(status_code=200, json=lambda: tags_payload)), \
         patch("httpx.AsyncClient.post", return_value=MagicMock(status_code=404)):
        connected, _ = await LocalModelRegistry.refresh(force=True)
        assert connected is True
        model = await LocalModelRegistry.get_model("custom-unknown:latest")
        assert model is not None
        assert model.parameter_size is None
        assert model.context_length is None
        assert model.tier == ModelTier.UNKNOWN


# --- 2. Deterministic Routing Decision Tests ---


@pytest.fixture
def mock_two_models():
    """Seed registry with small and medium model."""
    m_small = DiscoveredModel(
        name="qwen2.5:1.5b",
        model="qwen2.5:1.5b",
        digest="sha-1.5",
        size_bytes=986061892,
        modified_at="2026-10-21",
        parameter_size="1.5B",
        tier=ModelTier.SMALL,
    )
    m_medium = DiscoveredModel(
        name="qwen2.5:7b",
        model="qwen2.5:7b",
        digest="sha-7b",
        size_bytes=4683087332,
        modified_at="2026-10-21",
        parameter_size="7.6B",
        tier=ModelTier.MEDIUM,
    )
    LocalModelRegistry._models = {
        "qwen2.5:1.5b": m_small,
        "qwen2.5:7b": m_medium,
    }
    LocalModelRegistry._is_connected = True
    LocalModelRegistry._last_refresh_time = 1e9

    with patch.object(LocalModelRegistry, "refresh", return_value=(True, None)):
        yield m_small, m_medium


@pytest.mark.asyncio
async def test_router_weak_model_selection(mock_two_models):
    """Simple, low-constraint query routes to small model."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 5000, "used_mb": 1141, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        decision = await ModelRouter.route(
            prompt="Summarize the meeting notes into three brief bullet points."
        )
        assert decision.selected_model == "qwen2.5:1.5b"
        assert decision.target_model_tier == ModelTier.SMALL
        assert decision.routing_status == RoutingStatus.ROUTED
        assert decision.complexity_assessment.heuristic_score < 0.55
        assert decision.is_fallback is False


@pytest.mark.asyncio
async def test_router_strong_model_selection(mock_two_models):
    """Multi-constraint, high-reasoning coding task routes to medium model when resources permit."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 5500, "used_mb": 641, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 5000, "used_mb": 11000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        complex_prompt = (
            "Write an asynchronous Python data ingestion service using httpx and asyncio. "
            "Do not use requests or aiohttp. Parse the incoming JSON payloads, extract entities "
            "matching schema keys ['user_id', 'transaction_value', 'timestamp'], validate that "
            "transaction_value is non-negative, handle transient network retries with exponential backoff, "
            "and output the final aggregated metric dictionary to metrics.json."
        )
        decision = await ModelRouter.route(prompt=complex_prompt)
        assert decision.selected_model == "qwen2.5:7b"
        assert decision.target_model_tier == ModelTier.MEDIUM
        assert decision.routing_status == RoutingStatus.ROUTED
        assert decision.complexity_assessment.heuristic_score >= 0.55
        assert decision.is_fallback is False


@pytest.mark.asyncio
async def test_router_resource_constrained_fallback(mock_two_models):
    """Complex prompt warrants 7B model, but constrained VRAM forces fallback to 1.5B."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 1200, "used_mb": 4941, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        complex_prompt = (
            "Develop a full multi-threaded compiler pass in C++ to analyze AST nodes, "
            "do not use LLVM headers, detect memory leak locations, parse keys ['line', 'symbol'], and write a summary table."
        )
        decision = await ModelRouter.route(prompt=complex_prompt)
        assert decision.selected_model == "qwen2.5:1.5b"
        assert decision.target_model_tier == ModelTier.SMALL
        assert decision.routing_status == RoutingStatus.FALLBACK
        assert decision.is_fallback is True
        assert "resource constraints intervened" in decision.reason.lower()
        assert decision.resource_assessment.is_resource_sufficient is False


@pytest.mark.asyncio
async def test_router_ram_constrained_fallback(mock_two_models):
    """Host RAM free is below safety margin (<1024 MiB), forcing fallback."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 5500, "used_mb": 641, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 500, "used_mb": 15500, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        complex_prompt = (
            "Write an asynchronous Python data ingestion service using httpx and asyncio. "
            "Do not use requests or aiohttp. Parse incoming JSON, extract entities matching keys "
            "['user_id', 'value'], handle retries with backoff, and save to metrics.json."
        )
        decision = await ModelRouter.route(prompt=complex_prompt)
        assert decision.selected_model == "qwen2.5:1.5b"
        assert decision.is_fallback is True
        assert "host ram is constrained" in str(decision.resource_assessment.resource_pressure_reason).lower()


@pytest.mark.asyncio
async def test_router_insufficient_memory_info(mock_two_models):
    """When hardware telemetry returns unavailable, router executes without crashing."""
    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=None), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value={"source": "unavailable", "total_mb": 0}):
        decision = await ModelRouter.route(prompt="Simple note.")
        assert decision.selected_model == "qwen2.5:1.5b"
        assert decision.resource_assessment.gpu_available is False


@pytest.mark.asyncio
async def test_router_manual_override_installed(mock_two_models):
    """Manual override selects requested installed model regardless of complexity."""
    decision = await ModelRouter.route(
        prompt="Simple note.",
        requested_model="qwen2.5:7b",
    )
    assert decision.selected_model == "qwen2.5:7b"
    assert decision.routing_status == RoutingStatus.MANUAL_OVERRIDE
    assert decision.policy_applied == "manual"
    assert "manual user override" in decision.reason.lower()


@pytest.mark.asyncio
async def test_router_manual_override_not_installed(mock_two_models):
    """Manual override with uninstalled model raises ModelNotInstalledError."""
    with pytest.raises(ModelNotInstalledError) as exc_info:
        await ModelRouter.route(
            prompt="Simple note.",
            requested_model="mistral:7b-instruct",
        )
    assert "mistral:7b-instruct" in str(exc_info.value)


@pytest.mark.asyncio
async def test_router_disabled_by_config(mock_two_models):
    """When ROUTING_ENABLED is False, router returns default model with status disabled."""
    with patch.object(settings, "ROUTING_ENABLED", False):
        decision = await ModelRouter.route(prompt="Complex coding prompt...")
        assert decision.selected_model == settings.OLLAMA_MODEL
        assert decision.routing_status == RoutingStatus.DISABLED
        assert "disabled" in decision.reason.lower()


# --- 3. End-to-End Optimizer Integration Tests ---


@pytest.mark.asyncio
async def test_optimizer_passes_routed_model_to_ollama(mock_two_models):
    """Verify that OllamaService.optimize_prompt genuinely passes the routed model to /api/generate."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 5500, "used_mb": 641, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 5000, "used_mb": 11000, "source": "win32"}

    captured_payloads = []

    async def mock_post(url, *args, **kwargs):
        json_data = kwargs.get("json", {})
        captured_payloads.append(json_data)
        return MagicMock(
            status_code=200,
            json=lambda: {
                "response": "Optimized output here",
                "done_reason": "stop",
                "prompt_eval_count": 50,
                "eval_count": 30,
                "eval_duration": 500000000,
                "prompt_eval_duration": 100000000,
            },
        )

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot), \
         patch("httpx.AsyncClient.post", side_effect=mock_post):

        complex_prompt = (
            "Write an asynchronous Python data ingestion service using httpx. "
            "Do not use requests. Parse JSON into ['id', 'value'], and write output to out.json."
        )
        req = OptimizeRequest(prompt=complex_prompt)
        resp: OptimizeResponse = await OllamaService.optimize_prompt(req)

        assert len(captured_payloads) >= 1
        assert captured_payloads[0]["model"] == "qwen2.5:7b"
        assert resp.model == "qwen2.5:7b"
        assert resp.routing is not None
        assert resp.routing.selected_model == "qwen2.5:7b"
        assert resp.routing.routing_status == "routed"


@pytest.mark.asyncio
async def test_optimizer_retry_model_consistency(mock_two_models):
    """Verify that corrective validation retries use the identical routed model."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 5500, "used_mb": 641, "source": "nvidia_smi"}
    captured_payloads = []

    async def mock_post(url, *args, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        if len(captured_payloads) == 1:
            resp_text = "Here is code using requests library."
        else:
            resp_text = "Here is clean code without requests."

        return MagicMock(
            status_code=200,
            json=lambda: {
                "response": resp_text,
                "done_reason": "stop",
                "prompt_eval_count": 60,
                "eval_count": 40,
                "eval_duration": 600000000,
                "prompt_eval_duration": 120000000,
            },
        )

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("httpx.AsyncClient.post", side_effect=mock_post):

        complex_prompt = "Build a script. Do not use requests."
        req = OptimizeRequest(prompt=complex_prompt, requested_model="qwen2.5:7b")
        resp = await OllamaService.optimize_prompt(req)

        assert len(captured_payloads) == 2
        assert captured_payloads[0]["model"] == "qwen2.5:7b"
        assert captured_payloads[1]["model"] == "qwen2.5:7b"
        assert resp.model == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_optimizer_respects_enable_routing_false(mock_two_models):
    """When enable_routing=False in request, uses configured default model."""
    req = OptimizeRequest(
        prompt="Complex prompt with tons of constraints...",
        enable_routing=False,
    )
    with patch("app.services.ollama_service.OllamaService._execute_model_call") as mock_call:
        mock_call.return_value = ("Optimized", 100, {"eval_count": 10, "native_metrics_available": True})
        resp = await OllamaService.optimize_prompt(req)
        assert resp.model == settings.OLLAMA_MODEL
        assert resp.routing is not None
        assert resp.routing.selected_model == settings.OLLAMA_MODEL


# --- 4. API Endpoints Tests ---


def test_api_get_models(client, mock_two_models):
    """Test GET /api/models returns 200 with catalog of installed models."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert data["connected"] is True
    assert data["total_models"] == 2
    model_names = [m["name"] for m in data["models"]]
    assert "qwen2.5:1.5b" in model_names
    assert "qwen2.5:7b" in model_names


def test_api_post_route_preview(client, mock_two_models):
    """Test POST /api/route returns routing decision without executing inference."""
    payload = {
        "prompt": "Summarize this brief article.",
        "mode": "balanced",
    }
    response = client.post("/api/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["selected_model"] == "qwen2.5:1.5b"
    assert data["target_model_tier"] == "small"
    assert "complexity" in data
    assert "resources" in data
    assert data["complexity"]["heuristic_score"] is not None


def test_api_post_route_invalid_model_returns_400(client, mock_two_models):
    """Test POST /api/route with uninstalled model override returns 400."""
    payload = {
        "prompt": "Summarize this brief article.",
        "requested_model": "nonexistent-model:999b",
    }
    response = client.post("/api/route", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "nonexistent-model:999b" in data["error"]["message"]

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
from app.core.config import Settings, settings
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
    ModelResourceConstraintError,
    ModelRouter,
    ModelRouterError,
    NoSafeFallbackError,
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
    mock_get = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: tags_payload))
    mock_post = AsyncMock(return_value=MagicMock(status_code=404))
    with patch("httpx.AsyncClient.get", mock_get), \
         patch("httpx.AsyncClient.post", mock_post):
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
    ram_snapshot = {"total_mb": 16000, "free_mb": 700, "used_mb": 15300, "source": "win32"}

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
    """When hardware telemetry returns unavailable, router rejects admission with structured error."""
    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=None), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value={"source": "unavailable", "total_mb": 0}):
        with pytest.raises(ModelResourceConstraintError) as exc_info:
            await ModelRouter.route(prompt="Simple note.")
        assert "hardware telemetry unavailable" in str(exc_info.value).lower() or "telemetry is unavailable" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_router_manual_override_installed(mock_two_models):
    """Manual override safely selects requested installed model when resources are sufficient."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 5500, "used_mb": 641, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 5000, "used_mb": 11000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        decision = await ModelRouter.route(
            prompt="Simple note.",
            requested_model="qwen2.5:7b",
        )
        assert decision.selected_model == "qwen2.5:7b"
        assert decision.routing_status == RoutingStatus.MANUAL_OVERRIDE
        assert decision.policy_applied == "manual"
        assert decision.is_fallback is False
        assert "safely admitted" in decision.reason.lower()


@pytest.mark.asyncio
async def test_router_manual_override_insufficient_vram_rejected(mock_two_models):
    """Manual override rejected with ModelResourceConstraintError when VRAM is insufficient and fallback not permitted."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 2000, "used_mb": 4141, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        with pytest.raises(ModelResourceConstraintError) as exc_info:
            await ModelRouter.route(
                prompt="Simple note.",
                requested_model="qwen2.5:7b",
                allow_override_fallback=False,
            )
        assert "cannot be safely admitted" in str(exc_info.value)
        assert "qwen2.5:7b" in str(exc_info.value)


@pytest.mark.asyncio
async def test_router_manual_override_insufficient_ram_rejected(mock_two_models):
    """Manual override rejected with ModelResourceConstraintError when host RAM is below safety margin."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 5500, "used_mb": 641, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 500, "used_mb": 15500, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        with pytest.raises(ModelResourceConstraintError) as exc_info:
            await ModelRouter.route(
                prompt="Simple note.",
                requested_model="qwen2.5:7b",
                allow_override_fallback=False,
            )
        assert "host ram is constrained" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_router_manual_override_loaded_insufficient_headroom_rejected(mock_two_models):
    """Loaded resident model rejected if active free VRAM is below runtime headroom margin (<250 MiB)."""
    m_small, m_medium = mock_two_models
    m_medium.is_loaded = True
    m_medium.size_vram = 4500 * 1024 * 1024

    gpu_snapshot = {"total_mb": 6141, "free_mb": 150, "used_mb": 5991, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        with pytest.raises(ModelResourceConstraintError) as exc_info:
            await ModelRouter.route(
                prompt="Simple note.",
                requested_model="qwen2.5:7b",
                allow_override_fallback=False,
            )
        assert "insufficient runtime vram headroom" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_router_manual_override_fallback_permitted(mock_two_models):
    """When manual override model is constrained but fallback is explicitly permitted, safe fallback is executed."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 1200, "used_mb": 4941, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        decision = await ModelRouter.route(
            prompt="Simple note.",
            requested_model="qwen2.5:7b",
            allow_override_fallback=True,
        )
        assert decision.selected_model == "qwen2.5:1.5b"
        assert decision.routing_status == RoutingStatus.FALLBACK
        assert decision.is_fallback is True
        assert decision.warning is not None
        assert "not admitted" in decision.warning.lower()


@pytest.mark.asyncio
async def test_router_manual_override_fallback_refused_by_default(mock_two_models):
    """By default, manual override fallback is refused (raises ModelResourceConstraintError)."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 1200, "used_mb": 4941, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        with pytest.raises(ModelResourceConstraintError):
            await ModelRouter.route(
                prompt="Simple note.",
                requested_model="qwen2.5:7b",
            )


@pytest.mark.asyncio
async def test_router_no_safe_fallback_raises_error():
    """When a model is constrained and no safe fallback exists, router raises NoSafeFallbackError."""
    # Only a 7B model is installed; no 1.5B model exists
    m_only_7b = DiscoveredModel(
        name="qwen2.5:7b",
        model="qwen2.5:7b",
        digest="sha-7b",
        size_bytes=4683087332,
        modified_at="2026-10-21",
        parameter_size="7.6B",
        tier=ModelTier.MEDIUM,
    )
    LocalModelRegistry._models = {"qwen2.5:7b": m_only_7b}
    LocalModelRegistry._is_connected = True
    LocalModelRegistry._last_refresh_time = 1e9

    gpu_snapshot = {"total_mb": 6141, "free_mb": 1000, "used_mb": 5141, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch.object(LocalModelRegistry, "refresh", return_value=(True, None)), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        with pytest.raises(NoSafeFallbackError) as exc_info:
            await ModelRouter.route(
                prompt="Write an asynchronous Python data ingestion service with multiple constraints.",
            )
        assert "no installed safe fallback" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_router_configured_fallback_not_installed():
    """Configured default fallback is not assumed installed; router searches installed small models safely."""
    # Installed model is 'custom-small:latest' (not 'qwen2.5:1.5b')
    m_custom_small = DiscoveredModel(
        name="custom-small:latest",
        model="custom-small:latest",
        digest="sha-custom",
        size_bytes=1000000000,
        modified_at="2026-10-21",
        parameter_size="1.5B",
        tier=ModelTier.SMALL,
    )
    LocalModelRegistry._models = {"custom-small:latest": m_custom_small}
    LocalModelRegistry._is_connected = True
    LocalModelRegistry._last_refresh_time = 1e9

    gpu_snapshot = {"total_mb": 6141, "free_mb": 4000, "used_mb": 2141, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch.object(LocalModelRegistry, "refresh", return_value=(True, None)), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        # Settings points to nonexistent 'qwen2.5:1.5b'
        with patch.object(settings, "ROUTING_DEFAULT_FALLBACK_MODEL", "qwen2.5:1.5b"):
            decision = await ModelRouter.route(prompt="Simple note.")
            assert decision.selected_model == "custom-small:latest"
            assert decision.target_model_tier == ModelTier.SMALL


def test_registry_parse_tier_zero_guessing():
    """Verify _parse_tier never guesses model tier from model name substring when metadata is missing."""
    assert LocalModelRegistry._parse_tier("1.5B", None, "custom-model") == ModelTier.SMALL
    assert LocalModelRegistry._parse_tier("7.6B", None, "custom-model") == ModelTier.MEDIUM
    assert LocalModelRegistry._parse_tier("70B", None, "custom-model") == ModelTier.LARGE
    assert LocalModelRegistry._parse_tier(None, 1_500_000_000, "custom-model") == ModelTier.SMALL
    assert LocalModelRegistry._parse_tier(None, 7_000_000_000, "custom-model") == ModelTier.MEDIUM
    # Substring in name MUST NOT be used when metadata is absent!
    assert LocalModelRegistry._parse_tier(None, None, "qwen2.5:1.5b") == ModelTier.UNKNOWN
    assert LocalModelRegistry._parse_tier(None, None, "llama3:70b") == ModelTier.UNKNOWN
    assert LocalModelRegistry._parse_tier(None, None, "random-unknown") == ModelTier.UNKNOWN


@pytest.mark.asyncio
async def test_router_unknown_tier_automatic_safety():
    """An unknown-tier model is not silently assumed small or safe during automatic routing."""
    m_unknown = DiscoveredModel(
        name="custom-unlabeled:latest",
        model="custom-unlabeled:latest",
        digest="sha-unlabeled",
        size_bytes=2000000000,
        modified_at="2026-10-21",
        parameter_size=None,
        tier=ModelTier.UNKNOWN,
    )
    LocalModelRegistry._models = {"custom-unlabeled:latest": m_unknown}
    LocalModelRegistry._is_connected = True
    LocalModelRegistry._last_refresh_time = 1e9

    with patch.object(LocalModelRegistry, "refresh", return_value=(True, None)), \
         patch.object(settings, "ROUTING_ALLOW_UNKNOWN_TIER", False):
        with pytest.raises(NoSafeFallbackError):
            await ModelRouter.route(prompt="Simple query.")


@pytest.mark.asyncio
async def test_dynamic_discovery_different_model_names():
    """Verify router dynamically handles diverse model names (e.g. llama3:8b, phi3:mini)."""
    m_phi = DiscoveredModel(
        name="phi3:mini",
        model="phi3:mini",
        digest="sha-phi",
        size_bytes=2200000000,
        modified_at="2026-10-21",
        parameter_size="3.8B",
        tier=ModelTier.MEDIUM,
    )
    m_tiny = DiscoveredModel(
        name="smollm:135m",
        model="smollm:135m",
        digest="sha-smol",
        size_bytes=300000000,
        modified_at="2026-10-21",
        parameter_size="0.135B",
        tier=ModelTier.SMALL,
    )
    LocalModelRegistry._models = {
        "phi3:mini": m_phi,
        "smollm:135m": m_tiny,
    }
    LocalModelRegistry._is_connected = True
    LocalModelRegistry._last_refresh_time = 1e9

    gpu_snapshot = {"total_mb": 6141, "free_mb": 4000, "used_mb": 2141, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch.object(LocalModelRegistry, "refresh", return_value=(True, None)), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        # Simple task routes to small model
        dec_simple = await ModelRouter.route(prompt="Quick hello summary.")
        assert dec_simple.selected_model == "smollm:135m"
        assert dec_simple.target_model_tier == ModelTier.SMALL

        # Complex task routes to medium model
        dec_complex = await ModelRouter.route(
            prompt="Write an asynchronous Python ETL script. Do not use requests. Parse JSON ['id', 'metric'], handle retries.",
        )
        assert dec_complex.selected_model == "phi3:mini"
        assert dec_complex.target_model_tier == ModelTier.MEDIUM


@pytest.mark.asyncio
async def test_heuristic_margin_calculation(mock_two_models):
    """Verify decision_confidence is an uncalibrated heuristic margin (|score - threshold|), not a probability."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 5000, "used_mb": 1141, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        decision = await ModelRouter.route(prompt="Summarize notes.")
        threshold = settings.ROUTING_COMPLEXITY_THRESHOLD
        score = decision.complexity_assessment.heuristic_score
        expected_margin = round(abs(score - threshold), 3)

        assert decision.heuristic_margin == expected_margin
        assert 0.0 <= decision.heuristic_margin <= 1.0


@pytest.mark.asyncio
async def test_router_simulated_policy_vs_natural_pressure(mock_two_models):
    """Verify simulated policy constraint (high margin setting) triggers fallback without exhausting real memory."""
    # Natural hardware is completely healthy (5500 MB free VRAM, 5000 MB free RAM)
    gpu_snapshot = {"total_mb": 6141, "free_mb": 5500, "used_mb": 641, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 5000, "used_mb": 11000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        complex_prompt = (
            "Write an asynchronous Python data ingestion service using httpx and asyncio. "
            "Do not use requests. Parse JSON ['user_id', 'value'], handle retries, save to metrics.json."
        )

        # 1. Under natural policy margins, it safely routes to 7B
        dec_natural = await ModelRouter.route(prompt=complex_prompt)
        assert dec_natural.selected_model == "qwen2.5:7b"
        assert dec_natural.is_fallback is False

        # 2. Under a simulated restrictive policy margin, it falls back to 1.5B
        with patch.object(settings, "ROUTING_RAM_SAFETY_MARGIN_MB", 50000):
            dec_simulated = await ModelRouter.route(prompt=complex_prompt)
            assert dec_simulated.selected_model == "qwen2.5:1.5b"
            assert dec_simulated.is_fallback is True
            assert dec_simulated.routing_status == RoutingStatus.FALLBACK


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


def test_api_post_route_override_constrained_returns_400(client, mock_two_models):
    """Test POST /api/route with resource-constrained manual override returns 400 by default."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 2000, "used_mb": 4141, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        payload = {
            "prompt": "Simple note.",
            "requested_model": "qwen2.5:7b",
            "allow_override_fallback": False,
        }
        response = client.post("/api/route", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "cannot be safely admitted" in data["error"]["message"]


def test_api_post_route_override_fallback_permitted(client, mock_two_models):
    """Test POST /api/route with resource-constrained manual override and allow_override_fallback=True returns 200 with fallback."""
    gpu_snapshot = {"total_mb": 6141, "free_mb": 2000, "used_mb": 4141, "source": "nvidia_smi"}
    ram_snapshot = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu_snapshot), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram_snapshot):
        payload = {
            "prompt": "Simple note.",
            "requested_model": "qwen2.5:7b",
            "allow_override_fallback": True,
        }
        response = client.post("/api/route", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["routing_status"] == "fallback"
        assert data["selected_model"] == "qwen2.5:1.5b"
        assert data["is_fallback"] is True


# ==============================================================================
# Phase 5 Corrections: Neutral Ranking, VRAM Estimation & Fallback Tests
# ==============================================================================


def test_neutral_ranking_vendor_independent_non_qwen():
    """Verify that routing decisions are strictly vendor-independent and treat non-Qwen models neutrally."""
    m_llama = DiscoveredModel(
        name="llama3.2:1b", model="llama3.2:1b", digest="d1", size_bytes=1_200_000_000,
        modified_at="2026-10-01", parameter_size="1.2B", quantization_level="Q4_K_M",
        tier=ModelTier.SMALL, is_loaded=False,
    )
    m_phi = DiscoveredModel(
        name="phi3:mini", model="phi3:mini", digest="d2", size_bytes=2_400_000_000,
        modified_at="2026-10-01", parameter_size="3.8B", quantization_level="Q4_K_M",
        tier=ModelTier.MEDIUM, is_loaded=False,
    )
    m_mistral = DiscoveredModel(
        name="mistral:7b", model="mistral:7b", digest="d3", size_bytes=4_500_000_000,
        modified_at="2026-10-01", parameter_size="7.2B", quantization_level="Q4_K_M",
        tier=ModelTier.MEDIUM, is_loaded=False,
    )

    # In medium tier, higher parameter count (mistral 7.2B > phi 3.8B) is ranked first
    ranked_medium = ModelRouter.rank_candidates([m_phi, m_mistral], ModelTier.MEDIUM)
    assert ranked_medium[0].name == "mistral:7b"
    assert ranked_medium[1].name == "phi3:mini"

    # In small tier, llama3.2:1b is cleanly ranked
    ranked_small = ModelRouter.rank_candidates([m_llama], ModelTier.SMALL)
    assert ranked_small[0].name == "llama3.2:1b"


def test_ranking_ties_deterministic_alphabetical():
    """Verify that when parameters and quantization are identical, tie-breaking is deterministically alphabetical."""
    m_alpha = DiscoveredModel(
        name="alpha-llm:7b", model="alpha-llm:7b", digest="da", size_bytes=4_000_000_000,
        modified_at="2026-10-01", parameter_size="7B", quantization_level="Q4_K_M",
        tier=ModelTier.MEDIUM, is_loaded=False,
    )
    m_beta = DiscoveredModel(
        name="beta-llm:7b", model="beta-llm:7b", digest="db", size_bytes=4_000_000_000,
        modified_at="2026-10-01", parameter_size="7B", quantization_level="Q4_K_M",
        tier=ModelTier.MEDIUM, is_loaded=False,
    )
    m_gamma = DiscoveredModel(
        name="gamma-llm:7b", model="gamma-llm:7b", digest="dg", size_bytes=4_000_000_000,
        modified_at="2026-10-01", parameter_size="7B", quantization_level="Q4_K_M",
        tier=ModelTier.MEDIUM, is_loaded=False,
    )

    ranked_1 = ModelRouter.rank_candidates([m_gamma, m_alpha, m_beta], ModelTier.MEDIUM)
    ranked_2 = ModelRouter.rank_candidates([m_beta, m_gamma, m_alpha], ModelTier.MEDIUM)

    expected = ["alpha-llm:7b", "beta-llm:7b", "gamma-llm:7b"]
    assert [m.name for m in ranked_1] == expected
    assert [m.name for m in ranked_2] == expected


def test_ranking_resident_priority_over_unloaded():
    """Verify currently loaded/resident model is prioritized over unloaded candidate in same tier."""
    m_unloaded = DiscoveredModel(
        name="unloaded-model:7b", model="unloaded-model:7b", digest="d1", size_bytes=4_500_000_000,
        modified_at="2026-10-01", parameter_size="7.5B", quantization_level="Q4_K_M",
        tier=ModelTier.MEDIUM, is_loaded=False,
    )
    m_loaded = DiscoveredModel(
        name="resident-model:7b", model="resident-model:7b", digest="d2", size_bytes=4_200_000_000,
        modified_at="2026-10-01", parameter_size="7.0B", quantization_level="Q4_K_M",
        tier=ModelTier.MEDIUM, is_loaded=True, size_vram=4_300_000_000,
    )

    ranked = ModelRouter.rank_candidates([m_unloaded, m_loaded], ModelTier.MEDIUM)
    assert ranked[0].name == "resident-model:7b"
    assert ranked[0].is_loaded is True


def test_ranking_quantization_precision_priority():
    """Verify higher precision quantization is prioritized when parameter counts match."""
    m_q4 = DiscoveredModel(
        name="model:7b-q4", model="model:7b-q4", digest="d1", size_bytes=4_000_000_000,
        modified_at="2026-10-01", parameter_size="7B", quantization_level="Q4_K_M",
        tier=ModelTier.MEDIUM, is_loaded=False,
    )
    m_q8 = DiscoveredModel(
        name="model:7b-q8", model="model:7b-q8", digest="d2", size_bytes=7_500_000_000,
        modified_at="2026-10-01", parameter_size="7B", quantization_level="Q8_0",
        tier=ModelTier.MEDIUM, is_loaded=False,
    )

    ranked = ModelRouter.rank_candidates([m_q4, m_q8], ModelTier.MEDIUM)
    assert ranked[0].name == "model:7b-q8"


def test_ranking_missing_metadata_handled_gracefully():
    """Verify models with verified metadata rank ahead of models with missing parameter information."""
    m_verified = DiscoveredModel(
        name="verified:1b", model="verified:1b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1B", quantization_level="Q4_0",
        tier=ModelTier.SMALL, is_loaded=False,
    )
    m_unverified = DiscoveredModel(
        name="unverified:1b", model="unverified:1b", digest="d2", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size=None, quantization_level=None,
        tier=ModelTier.SMALL, is_loaded=False,
    )

    ranked = ModelRouter.rank_candidates([m_unverified, m_verified], ModelTier.SMALL)
    assert ranked[0].name == "verified:1b"
    assert ranked[1].name == "unverified:1b"


def test_evaluate_resources_measured_resident_vram():
    """Verify loaded model uses exact size_vram from /api/ps and identifies as 'measured_resident'."""
    m_loaded = DiscoveredModel(
        name="resident:7b", model="resident:7b", digest="d1", size_bytes=4_500_000_000,
        modified_at="2026-10-01", parameter_size="7B", tier=ModelTier.MEDIUM,
        is_loaded=True, size_vram=4_199_263_764,  # ~4005 MiB
    )
    gpu = {"total_mb": 6141, "free_mb": 4000, "used_mb": 2141, "source": "nvidia_smi"}
    ram = {"total_mb": 16000, "free_mb": 8000, "used_mb": 8000, "source": "win32"}

    res = ModelRouter.evaluate_resources(m_loaded, gpu_telemetry=gpu, ram_telemetry=ram)
    assert res.vram_measurement_type == "measured_resident"
    assert res.measured_vram_mb == round(4_199_263_764 / (1024 * 1024))
    assert res.weight_footprint_mb == round(4_500_000_000 / (1024 * 1024))
    assert res.is_resource_sufficient is True  # 4000 MiB free > 250 MiB headroom required for loaded


def test_evaluate_resources_unloaded_model_proxy_distinction():
    """Verify unloaded model reports 'estimated_proxy', separates weight size from runtime proxy."""
    m_unloaded = DiscoveredModel(
        name="unloaded:7b", model="unloaded:7b", digest="d1", size_bytes=4_683_087_332,  # ~4466 MiB
        modified_at="2026-10-01", parameter_size="7.6B", tier=ModelTier.MEDIUM,
        is_loaded=False, size_vram=None,
    )
    gpu = {"total_mb": 6141, "free_mb": 5500, "used_mb": 641, "source": "nvidia_smi"}
    ram = {"total_mb": 16000, "free_mb": 8000, "used_mb": 8000, "source": "win32"}

    res = ModelRouter.evaluate_resources(m_unloaded, gpu_telemetry=gpu, ram_telemetry=ram, context_tier=4096)
    assert res.vram_measurement_type in ("estimated_proxy", "rough_proxy")
    assert res.measured_vram_mb is None
    assert res.weight_footprint_mb == 4466
    # Runtime proxy includes weight (4466) + KV cache buffer (350) = 4816 MiB
    assert res.estimated_runtime_vram_mb == 4816
    # Headroom check: 5500 MiB free >= 4816 + 512 margin (5328) -> True
    assert res.is_resource_sufficient is True


def test_evaluate_resources_missing_metadata_conservative_rejection():
    """Verify unloaded model with 0 or missing size_bytes triggers conservative admission rejection."""
    m_bad = DiscoveredModel(
        name="bad:model", model="bad:model", digest="d1", size_bytes=0,
        modified_at="2026-10-01", parameter_size="1B", tier=ModelTier.SMALL,
        is_loaded=False,
    )
    gpu = {"total_mb": 6141, "free_mb": 5500, "used_mb": 641, "source": "nvidia_smi"}
    ram = {"total_mb": 16000, "free_mb": 8000, "used_mb": 8000, "source": "win32"}

    res = ModelRouter.evaluate_resources(m_bad, gpu_telemetry=gpu, ram_telemetry=ram)
    assert res.is_resource_sufficient is False
    assert res.vram_measurement_type == "unverified"
    assert "unverified resource metadata" in str(res.resource_pressure_reason).lower()


def test_evaluate_resources_vram_headroom_limits():
    """Verify resource evaluation rejects models when available VRAM is insufficient for runtime requirements."""
    m_7b = DiscoveredModel(
        name="heavy:7b", model="heavy:7b", digest="d1", size_bytes=4_683_087_332,  # 4466 MiB
        modified_at="2026-10-01", parameter_size="7.6B", tier=ModelTier.MEDIUM,
        is_loaded=False,
    )
    # Only 3000 MiB VRAM free -> cannot fit 4950 MiB runtime requirement + 512 MiB margin
    gpu = {"total_mb": 6141, "free_mb": 3000, "used_mb": 3141, "source": "nvidia_smi"}
    ram = {"total_mb": 16000, "free_mb": 8000, "used_mb": 8000, "source": "win32"}

    res = ModelRouter.evaluate_resources(m_7b, gpu_telemetry=gpu, ram_telemetry=ram)
    assert res.is_resource_sufficient is False
    assert "insufficient available vram" in str(res.resource_pressure_reason).lower()


def test_evaluate_resources_ram_headroom_limits():
    """Verify resource evaluation rejects model when host RAM free is below safety margin."""
    m_1b = DiscoveredModel(
        name="light:1b", model="light:1b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1B", tier=ModelTier.SMALL,
        is_loaded=False,
    )
    gpu = {"total_mb": 6141, "free_mb": 5000, "used_mb": 1141, "source": "nvidia_smi"}
    # Only 200 MiB free host RAM (< 512 MiB small-tier RAM margin)
    ram = {"total_mb": 16000, "free_mb": 200, "used_mb": 15800, "source": "win32"}

    res = ModelRouter.evaluate_resources(m_1b, gpu_telemetry=gpu, ram_telemetry=ram)
    assert res.is_resource_sufficient is False
    assert "host ram is constrained" in str(res.resource_pressure_reason).lower()


def test_find_safe_fallback_selects_best_safe_installed_model():
    """Verify find_safe_fallback uses neutral ranking to select the best verified safe candidate."""
    m_heavy = DiscoveredModel(
        name="heavy:7b", model="heavy:7b", digest="d1", size_bytes=5_000_000_000,
        modified_at="2026-10-01", parameter_size="7B", tier=ModelTier.MEDIUM,
        is_loaded=False,
    )
    m_small_1 = DiscoveredModel(
        name="model-b:1.5b", model="model-b:1.5b", digest="d2", size_bytes=1_200_000_000,
        modified_at="2026-10-01", parameter_size="1.5B", tier=ModelTier.SMALL,
        is_loaded=False,
    )
    m_small_2 = DiscoveredModel(
        name="model-a:1.0b", model="model-a:1.0b", digest="d3", size_bytes=900_000_000,
        modified_at="2026-10-01", parameter_size="1.0B", tier=ModelTier.SMALL,
        is_loaded=False,
    )

    gpu = {"total_mb": 6141, "free_mb": 2000, "used_mb": 4141, "source": "nvidia_smi"}
    ram = {"total_mb": 16000, "free_mb": 4000, "used_mb": 12000, "source": "win32"}

    with patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_gpu_memory", return_value=gpu), \
         patch("app.services.hardware_telemetry.HardwareTelemetryProvider.get_system_ram", return_value=ram):
        # In small tier, model-a (1.0B) is ranked ahead of model-b (1.5B) for throughput
        fallback = ModelRouter.find_safe_fallback([m_heavy, m_small_1, m_small_2])
        assert fallback is not None
        assert fallback.name == "model-a:1.0b"


def test_evaluate_resources_missing_ram_telemetry():
    """Verify missing RAM telemetry does not pass safety check and reports unverified status."""
    m_1b = DiscoveredModel(
        name="test:1b", model="test:1b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1B", tier=ModelTier.SMALL,
        is_loaded=False,
    )
    gpu = {"total_mb": 6141, "free_mb": 5000, "used_mb": 1141, "source": "nvidia_smi"}
    ram = {"total_mb": 0.0, "free_mb": 0.0, "used_mb": 0.0, "source": "unavailable"}

    res = ModelRouter.evaluate_resources(m_1b, gpu_telemetry=gpu, ram_telemetry=ram)
    assert res.is_resource_sufficient is False
    assert res.vram_measurement_type == "unverified"
    assert "ram telemetry is unavailable" in str(res.resource_pressure_reason).lower()


def test_evaluate_resources_unavailable_gpu_telemetry_cpu_disabled():
    """Verify unavailable GPU telemetry rejects execution when ROUTING_ALLOW_CPU_EXECUTION is False."""
    m_1b = DiscoveredModel(
        name="test:1b", model="test:1b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1B", tier=ModelTier.SMALL,
        is_loaded=False,
    )
    ram = {"total_mb": 16000, "free_mb": 8000, "used_mb": 8000, "source": "win32"}

    with patch.object(settings, "ROUTING_ALLOW_CPU_EXECUTION", False):
        res = ModelRouter.evaluate_resources(m_1b, gpu_telemetry=None, ram_telemetry=ram)
        assert res.is_resource_sufficient is False
        assert "cpu-only execution is disabled" in str(res.resource_pressure_reason).lower()


def test_evaluate_resources_cpu_only_sufficient_ram():
    """Verify CPU-only execution is safely admitted when host RAM is sufficient and allowed by policy."""
    m_1b = DiscoveredModel(
        name="test:1b", model="test:1b", digest="d1", size_bytes=1_000_000_000,  # ~954 MiB
        modified_at="2026-10-01", parameter_size="1B", tier=ModelTier.SMALL,
        is_loaded=False,
    )
    ram = {"total_mb": 16000, "free_mb": 8000, "used_mb": 8000, "source": "win32"}

    with patch.object(settings, "ROUTING_ALLOW_CPU_EXECUTION", True):
        res = ModelRouter.evaluate_resources(m_1b, gpu_telemetry=None, ram_telemetry=ram)
        assert res.is_resource_sufficient is True
        assert res.gpu_available is False
        assert res.vram_measurement_type == "cpu_only"
        assert res.resource_pressure_reason is None


def test_evaluate_resources_cpu_only_insufficient_ram():
    """Verify CPU-only execution is rejected when host RAM cannot accommodate model weights and margin."""
    m_7b = DiscoveredModel(
        name="test:7b", model="test:7b", digest="d1", size_bytes=4_683_087_332,  # ~4466 MiB
        modified_at="2026-10-01", parameter_size="7.6B", tier=ModelTier.MEDIUM,
        is_loaded=False,
    )
    # 2000 MiB free RAM < 4466 weights + 1024 margin (5490 MiB)
    ram = {"total_mb": 16000, "free_mb": 2000, "used_mb": 14000, "source": "win32"}

    with patch.object(settings, "ROUTING_ALLOW_CPU_EXECUTION", True):
        res = ModelRouter.evaluate_resources(m_7b, gpu_telemetry=None, ram_telemetry=ram)
        assert res.is_resource_sufficient is False
        assert res.vram_measurement_type == "cpu_only"
        assert "insufficient host ram for cpu-only execution" in str(res.resource_pressure_reason).lower()


def test_evaluate_resources_both_telemetry_sources_unavailable():
    """Verify resource safety cannot be established when both GPU and RAM telemetry are missing."""
    m_1b = DiscoveredModel(
        name="test:1b", model="test:1b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1B", tier=ModelTier.SMALL,
        is_loaded=False,
    )
    res = ModelRouter.evaluate_resources(m_1b, gpu_telemetry=None, ram_telemetry=None)
    assert res.is_resource_sufficient is False
    assert res.vram_measurement_type == "unverified"
    assert "both gpu and host ram telemetry are missing" in str(res.resource_pressure_reason).lower()


@pytest.mark.asyncio
async def test_route_manual_override_unavailable_telemetry(mock_two_models):
    """Verify manual override enforces the safety policy and rejects admission when telemetry is unavailable."""
    # When override fallback is disabled, raises ModelResourceConstraintError with diagnostic reason
    with pytest.raises(ModelResourceConstraintError) as exc_info:
        await ModelRouter.route(
            prompt="Simple summary request",
            requested_model="qwen2.5:1.5b",
            allow_override_fallback=False,
            ram_telemetry=None,
        )
    assert "ram telemetry is unavailable" in str(exc_info.value).lower()

    # When override fallback is allowed, but telemetry is unavailable for all models, raises NoSafeFallbackError
    with pytest.raises(NoSafeFallbackError) as exc_info:
        await ModelRouter.route(
            prompt="Simple summary request",
            requested_model="qwen2.5:1.5b",
            allow_override_fallback=True,
            ram_telemetry=None,
        )
    assert "no alternative safe fallback" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_route_fallback_when_resource_measurements_unavailable(mock_two_models):
    """Verify automatic routing fails safe with NoSafeFallbackError when measurements are unavailable."""
    complex_prompt = (
        "Write an asynchronous Python data ingestion service using httpx and asyncio. "
        "Do not use requests or aiohttp. Parse incoming JSON, extract entities matching keys "
        "['user_id', 'transaction_value', 'timestamp'], validate non-negative values, "
        "handle transient network retries with exponential backoff, and save aggregated dictionary to metrics.json."
    )
    with pytest.raises(NoSafeFallbackError) as exc_info:
        await ModelRouter.route(
            prompt=complex_prompt,
            ram_telemetry=None,
        )
    assert "no verified safe fallback" in str(exc_info.value).lower() or "no safe fallback" in str(exc_info.value).lower()


def test_discovered_model_architecture_derived_kv_cache():
    """Verify model with complete architecture metadata computes architecture-derived KV-cache."""
    m_arch = DiscoveredModel(
        name="arch:1.5b", model="arch:1.5b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1.5B", tier=ModelTier.SMALL,
        is_loaded=False,
        num_layers=28,
        num_kv_heads=2,
        head_dim=128,
        context_length=32768,
    )
    assert m_arch.has_architecture_metadata is True
    assert m_arch.get_vram_measurement_type() == "architecture_proxy"

    kv_mb, method = m_arch.estimate_kv_cache_mb(context_tier=4096, kv_dtype="q8_0")
    assert method == "architecture_derived"
    assert 50 <= kv_mb <= 70


def test_discovered_model_rough_proxy_when_architecture_missing():
    """Verify model without architecture metadata clearly identifies rough proxy estimation."""
    m_no_arch = DiscoveredModel(
        name="noarch:1.5b", model="noarch:1.5b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1.5B", tier=ModelTier.SMALL,
        is_loaded=False,
        num_layers=None,
    )
    assert m_no_arch.has_architecture_metadata is False
    assert m_no_arch.get_vram_measurement_type() == "rough_proxy"

    kv_mb, method = m_no_arch.estimate_kv_cache_mb(context_tier=4096)
    assert method == "rough_proxy"
    assert kv_mb == 100


def test_evaluate_resources_requires_architecture_metadata_policy():
    """Verify ROUTING_REQUIRE_ARCHITECTURE_METADATA policy rejects models relying on rough proxies."""
    m_rough = DiscoveredModel(
        name="rough:1.5b", model="rough:1.5b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1.5B", tier=ModelTier.SMALL,
        is_loaded=False,
        num_layers=None,
    )
    gpu = {"total_mb": 6141, "free_mb": 5000, "used_mb": 1141, "source": "nvidia_smi"}
    ram = {"total_mb": 16000, "free_mb": 8000, "used_mb": 8000, "source": "win32"}

    with patch.object(settings, "ROUTING_REQUIRE_ARCHITECTURE_METADATA", True):
        res = ModelRouter.evaluate_resources(m_rough, gpu_telemetry=gpu, ram_telemetry=ram)
        assert res.is_resource_sufficient is False
        assert "lacks verified architecture metadata" in str(res.resource_pressure_reason).lower()


def test_evaluate_resources_cpu_only_rejected_when_ram_sufficient_for_weights_alone_but_insufficient_with_kv_and_margin():
    """Verify CPU admission rejects model when RAM covers weights alone but not weights + KV cache + safety allowances."""
    # 1000 MiB weights. Small tier: kv_mb = 100, cpu_allowance = 64, ram_margin = 512.
    # Total required headroom: 1000 + 100 + 64 + 512 = 1676 MiB.
    m_1b = DiscoveredModel(
        name="test:1b", model="test:1b", digest="d1", size_bytes=1000 * 1024 * 1024,
        modified_at="2026-10-01", parameter_size="1B", tier=ModelTier.SMALL,
        is_loaded=False,
    )
    # 1200 MiB free host RAM > 1000 MiB weights, BUT < 1676 MiB total required!
    ram = {"total_mb": 16000, "free_mb": 1200, "used_mb": 14800, "source": "win32"}

    with patch.object(settings, "ROUTING_ALLOW_CPU_EXECUTION", True):
        res = ModelRouter.evaluate_resources(m_1b, gpu_telemetry=None, ram_telemetry=ram)
        assert res.is_resource_sufficient is False
        assert res.vram_measurement_type == "cpu_only"
        assert "insufficient host ram for cpu-only execution" in str(res.resource_pressure_reason).lower()
        assert "1000 mib weights" in str(res.resource_pressure_reason).lower()
        assert "100 mib kv cache" in str(res.resource_pressure_reason).lower()


def test_evaluate_resources_cpu_only_requires_architecture_metadata_policy():
    """Verify CPU-only execution rejects models relying on rough proxies when policy demands verified estimates."""
    m_rough = DiscoveredModel(
        name="rough:1.5b", model="rough:1.5b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1.5B", tier=ModelTier.SMALL,
        is_loaded=False,
        num_layers=None,
    )
    ram = {"total_mb": 16000, "free_mb": 8000, "used_mb": 8000, "source": "win32"}

    with patch.object(settings, "ROUTING_ALLOW_CPU_EXECUTION", True), \
         patch.object(settings, "ROUTING_REQUIRE_ARCHITECTURE_METADATA", True):
        res = ModelRouter.evaluate_resources(m_rough, gpu_telemetry=None, ram_telemetry=ram)
        assert res.is_resource_sufficient is False
        assert "lacks verified architecture metadata for exact cpu ram kv-cache estimation" in str(res.resource_pressure_reason).lower()


def test_estimate_kv_cache_standard_multi_head_attention():
    """Verify architecture-derived KV-cache calculation for standard Multi-Head Attention (MHA)."""
    m_mha = DiscoveredModel(
        name="mha:7b", model="mha:7b", digest="d1", size_bytes=4_000_000_000,
        modified_at="2026-10-01", parameter_size="7B", tier=ModelTier.MEDIUM,
        is_loaded=False,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,  # Standard MHA: num_kv_heads == num_heads
        head_dim=128,
        embedding_length=4096,
        context_length=4096,
    )
    assert m_mha.effective_kv_dim == 4096
    assert m_mha.has_architecture_metadata is True
    assert m_mha.get_vram_measurement_type() == "architecture_proxy"

    # For 4096 context with FP16 (2.0 bytes/elem):
    # 2 (K+V) * 32 layers * 4096 dim * 4096 tokens * 2.0 = 2,147,483,648 bytes = 2048 MiB
    kv_mb, method = m_mha.estimate_kv_cache_mb(context_tier=4096, kv_dtype="f16")
    assert method == "architecture_derived"
    assert kv_mb == 2048


def test_estimate_kv_cache_grouped_query_attention():
    """Verify GQA does not substitute embedding_length and calculates verified num_kv_heads * head_dim."""
    m_gqa = DiscoveredModel(
        name="gqa:1.5b", model="gqa:1.5b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1.5B", tier=ModelTier.SMALL,
        is_loaded=False,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,  # GQA: 2 KV heads vs 12 query heads
        head_dim=128,
        embedding_length=1536,  # 12 * 128 = 1536
        context_length=32768,
    )
    # KV dimension per layer MUST be 2 * 128 = 256, NOT embedding_length (1536)!
    assert m_gqa.effective_kv_dim == 256
    assert m_gqa.has_architecture_metadata is True
    assert m_gqa.get_vram_measurement_type() == "architecture_proxy"

    # For 4096 context with Q8_0 (1.0625 bytes/elem):
    # 2 * 28 layers * 256 dim * 4096 tokens * 1.0625 = 62,390,272 bytes ≈ 59.5 MiB
    kv_mb, method = m_gqa.estimate_kv_cache_mb(context_tier=4096, kv_dtype="q8_0")
    assert method == "architecture_derived"
    assert 55 <= kv_mb <= 65


def test_estimate_kv_cache_missing_kv_head_metadata_falls_back_to_rough_proxy():
    """Verify when num_kv_heads is missing, router does NOT guess or substitute embedding_length."""
    m_unknown_heads = DiscoveredModel(
        name="unknown-kv:1.5b", model="unknown-kv:1.5b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1.5B", tier=ModelTier.SMALL,
        is_loaded=False,
        num_layers=28,
        num_heads=12,
        num_kv_heads=None,  # Missing! Cannot know if MHA, GQA, or MQA
        head_dim=128,
        embedding_length=1536,
    )
    assert m_unknown_heads.effective_kv_dim is None
    assert m_unknown_heads.has_architecture_metadata is False
    assert m_unknown_heads.get_vram_measurement_type() == "rough_proxy"

    kv_mb, method = m_unknown_heads.estimate_kv_cache_mb(context_tier=4096)
    assert method == "rough_proxy"
    assert kv_mb == 100


def test_estimate_kv_cache_missing_head_dimension_falls_back_to_rough_proxy():
    """Verify when head dimension cannot be established, router falls back to rough proxy."""
    m_no_dim = DiscoveredModel(
        name="no-dim:1.5b", model="no-dim:1.5b", digest="d1", size_bytes=1_000_000_000,
        modified_at="2026-10-01", parameter_size="1.5B", tier=ModelTier.SMALL,
        is_loaded=False,
        num_layers=28,
        num_kv_heads=2,
        head_dim=None,
        num_heads=None,
        embedding_length=None,
    )
    assert m_no_dim.effective_head_dim is None
    assert m_no_dim.effective_kv_dim is None
    assert m_no_dim.has_architecture_metadata is False
    assert m_no_dim.get_vram_measurement_type() == "rough_proxy"

    kv_mb, method = m_no_dim.estimate_kv_cache_mb(context_tier=4096)
    assert method == "rough_proxy"


def test_estimate_kv_cache_unknown_architecture_metadata_falls_back_to_rough_proxy():
    """Verify when model layers are unknown, router falls back to rough proxy."""
    m_no_arch = DiscoveredModel(
        name="no-arch:7b", model="no-arch:7b", digest="d1", size_bytes=4_000_000_000,
        modified_at="2026-10-01", parameter_size="7B", tier=ModelTier.MEDIUM,
        is_loaded=False,
        num_layers=None,
    )
    assert m_no_arch.has_architecture_metadata is False
    assert m_no_arch.get_vram_measurement_type() == "rough_proxy"

    kv_mb, method = m_no_arch.estimate_kv_cache_mb(context_tier=4096)
    assert method == "rough_proxy"
    assert kv_mb == 350


def test_evaluate_resources_cpu_allowance_changes_admission_decision():
    """Verify changing configured CPU runtime allowance directly toggles CPU admission decisions."""
    # 1000 MiB model. Small tier has kv_mb=100 (rough proxy), ram_margin=min(1024, 512)=512.
    m_1b = DiscoveredModel(
        name="test:1b", model="test:1b", digest="d1", size_bytes=1000 * 1024 * 1024,
        modified_at="2026-10-01", parameter_size="1B", tier=ModelTier.SMALL,
        is_loaded=False,
    )
    # Available RAM = 1700 MiB
    ram = {"total_mb": 16000, "free_mb": 1700, "used_mb": 14300, "source": "win32"}

    # Case A: Default allowance = 64 MiB -> total required = 1000 + 100 + 64 + 512 = 1676 MiB <= 1700 MiB -> SUFFICIENT
    with patch.object(settings, "ROUTING_ALLOW_CPU_EXECUTION", True), \
         patch.object(settings, "ROUTING_CPU_ALLOWANCE_SMALL_MB", 64):
        res_pass = ModelRouter.evaluate_resources(m_1b, gpu_telemetry=None, ram_telemetry=ram)
        assert res_pass.is_resource_sufficient is True
        assert res_pass.resource_pressure_reason is None

    # Case B: Increased allowance = 200 MiB -> total required = 1000 + 100 + 200 + 512 = 1812 MiB > 1700 MiB -> INSUFFICIENT
    with patch.object(settings, "ROUTING_ALLOW_CPU_EXECUTION", True), \
         patch.object(settings, "ROUTING_CPU_ALLOWANCE_SMALL_MB", 200):
        res_fail = ModelRouter.evaluate_resources(m_1b, gpu_telemetry=None, ram_telemetry=ram)
        assert res_fail.is_resource_sufficient is False
        assert "insufficient host ram for cpu-only execution" in str(res_fail.resource_pressure_reason).lower()
        assert "200 mib runtime allowance" in str(res_fail.resource_pressure_reason).lower()


def test_evaluate_resources_cpu_allowance_tier_independence():
    """Verify CPU allowances can be configured independently per capacity tier."""
    m_small = DiscoveredModel(
        name="test:small", model="test:small", digest="d1", size_bytes=1000 * 1024 * 1024,
        modified_at="2026-10-01", parameter_size="1B", tier=ModelTier.SMALL,
    )
    m_medium = DiscoveredModel(
        name="test:medium", model="test:medium", digest="d2", size_bytes=4000 * 1024 * 1024,
        modified_at="2026-10-01", parameter_size="7B", tier=ModelTier.MEDIUM,
    )
    m_large = DiscoveredModel(
        name="test:large", model="test:large", digest="d3", size_bytes=8000 * 1024 * 1024,
        modified_at="2026-10-01", parameter_size="14B", tier=ModelTier.LARGE,
    )
    m_unknown = DiscoveredModel(
        name="test:unknown", model="test:unknown", digest="d4", size_bytes=2000 * 1024 * 1024,
        modified_at="2026-10-01", tier=ModelTier.UNKNOWN,
    )

    with patch.object(settings, "ROUTING_CPU_ALLOWANCE_SMALL_MB", 50), \
         patch.object(settings, "ROUTING_CPU_ALLOWANCE_MEDIUM_MB", 150), \
         patch.object(settings, "ROUTING_CPU_ALLOWANCE_LARGE_MB", 350), \
         patch.object(settings, "ROUTING_CPU_ALLOWANCE_DEFAULT_MB", 75):
        assert ModelRouter.get_cpu_runtime_allowance_mb(m_small) == 50
        assert ModelRouter.get_cpu_runtime_allowance_mb(m_medium) == 150
        assert ModelRouter.get_cpu_runtime_allowance_mb(m_large) == 350
        assert ModelRouter.get_cpu_runtime_allowance_mb(m_unknown) == 75


def test_evaluate_resources_cpu_loaded_model_distinguishes_memory_from_unloaded():
    """Verify diagnostics distinguish loaded resident memory from unloaded weights in CPU admission."""
    # Model size = 2000 MiB, small tier (kv_mb = 100, cpu_allowance = 64, ram_margin = 512).
    # Unloaded requirement = 2000 + 100 + 64 + 512 = 2676 MiB.
    # Loaded requirement = 100 + 64 + 512 = 676 MiB (weights already resident).
    m_loaded = DiscoveredModel(
        name="resident:2b", model="resident:2b", digest="d1", size_bytes=2000 * 1024 * 1024,
        modified_at="2026-10-01", parameter_size="2B", tier=ModelTier.SMALL,
        is_loaded=True,
    )
    # Available RAM = 600 MiB (< 676 MiB required for loaded headroom)
    ram_tight = {"total_mb": 16000, "free_mb": 600, "used_mb": 15400, "source": "win32"}

    with patch.object(settings, "ROUTING_ALLOW_CPU_EXECUTION", True):
        res = ModelRouter.evaluate_resources(m_loaded, gpu_telemetry=None, ram_telemetry=ram_tight)
        assert res.is_resource_sufficient is False
        reason = str(res.resource_pressure_reason).lower()
        assert "insufficient host ram for cpu-only execution of loaded model" in reason
        assert "model weights (2000 mib) are already loaded in ram" in reason
        assert "runtime headroom requirement" in reason
        assert "64 mib runtime allowance" in reason


def test_settings_cpu_allowance_validation():
    """Verify CPU runtime memory allowance settings enforce non-negative integers."""
    from pydantic import ValidationError

    # Valid configurations
    s_valid = Settings(
        ROUTING_CPU_ALLOWANCE_DEFAULT_MB=0,
        ROUTING_CPU_ALLOWANCE_SMALL_MB=32,
        ROUTING_CPU_ALLOWANCE_MEDIUM_MB=96,
        ROUTING_CPU_ALLOWANCE_LARGE_MB=192,
    )
    assert s_valid.ROUTING_CPU_ALLOWANCE_DEFAULT_MB == 0
    assert s_valid.ROUTING_CPU_ALLOWANCE_SMALL_MB == 32
    assert s_valid.ROUTING_CPU_ALLOWANCE_MEDIUM_MB == 96
    assert s_valid.ROUTING_CPU_ALLOWANCE_LARGE_MB == 192

    # Negative values are rejected
    with pytest.raises(ValidationError):
        Settings(ROUTING_CPU_ALLOWANCE_SMALL_MB=-1)

    with pytest.raises(ValidationError):
        Settings(ROUTING_CPU_ALLOWANCE_MEDIUM_MB=-50)

    with pytest.raises(ValidationError):
        Settings(ROUTING_CPU_ALLOWANCE_LARGE_MB=-100)

    with pytest.raises(ValidationError):
        Settings(ROUTING_CPU_ALLOWANCE_DEFAULT_MB=-1)



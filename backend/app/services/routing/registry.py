"""Dynamic Local Model Registry for Argus (Phase 5).

Discovers installed Ollama models via live HTTP endpoints:
- `GET /api/tags`: Enumerates installed models, digests, sizes, and formats.
- `GET /api/ps`: Identifies currently resident/loaded models in VRAM/RAM.
- `POST /api/show`: Retrieves deep parameter count and architecture metadata when useful.

Enforces zero-fabrication standards:
- Missing metadata is recorded as None, never synthesized or hallucinated.
- Uninstalled models are never assumed or phantom-registered.
- Dynamic refresh is supported without backend restarts.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
import httpx

from app.core.config import settings
from app.services.routing.models import DiscoveredModel, ModelCatalogItem, ModelCatalogResponse, ModelTier


class LocalModelRegistry:
    """Thread-safe dynamic registry of installed and active local Ollama models."""

    _models: Dict[str, DiscoveredModel] = {}
    _last_refresh_time: float = 0.0
    _is_connected: bool = False
    _last_error: Optional[str] = None
    _lock = asyncio.Lock()

    @classmethod
    def _parse_tier(cls, param_size_str: Optional[str], param_count: Optional[int], name: str = "") -> ModelTier:
        """Derive model capacity tier from verified parameter metadata without guessing.

        Strictly enforces zero-guessing:
        - Evaluates parameter_count if provided by /api/show (model_info.*.parameter_count).
        - Evaluates parameter_size if provided by /api/tags or /api/show (e.g. '1.5B', '7B').
        - Never infers model capacity from a model-name substring when metadata is unavailable.
        - If metadata is absent or unparseable, returns ModelTier.UNKNOWN.
        """
        if param_count is not None and param_count > 0:
            if param_count <= 3_500_000_000:
                return ModelTier.SMALL
            if param_count <= 14_500_000_000:
                return ModelTier.MEDIUM
            return ModelTier.LARGE

        if param_size_str:
            clean = param_size_str.strip().upper()
            if clean.endswith("B"):
                try:
                    val = float(clean[:-1])
                    if val <= 3.5:
                        return ModelTier.SMALL
                    if val <= 14.5:
                        return ModelTier.MEDIUM
                    return ModelTier.LARGE
                except ValueError:
                    pass

        return ModelTier.UNKNOWN

    @classmethod
    async def fetch_running_processes(cls, client: httpx.AsyncClient) -> Dict[str, Dict]:
        """Query /api/ps to detect models currently resident in memory."""
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/ps"
        try:
            resp = await client.get(url, timeout=settings.OLLAMA_STATUS_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                models_list = data.get("models", [])
                running: Dict[str, Dict] = {}
                for m in models_list:
                    if isinstance(m, dict) and "name" in m:
                        running[m["name"].lower()] = m
                        # Also map without tag if present (e.g. "qwen2.5:1.5b" -> "qwen2.5:1.5b")
                        model_attr = m.get("model", "")
                        if model_attr:
                            running[model_attr.lower()] = m
                return running
        except Exception:
            pass
        return {}

    @classmethod
    async def fetch_model_details(cls, client: httpx.AsyncClient, model_name: str) -> Dict:
        """Query /api/show for rich parameter counts and context length."""
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/show"
        try:
            resp = await client.post(url, json={"model": model_name}, timeout=settings.OLLAMA_STATUS_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}

    @classmethod
    async def unload_model(cls, model_name: str) -> bool:
        """Explicitly request Ollama to unload a resident model from VRAM/RAM."""
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_STATUS_TIMEOUT_SECONDS) as client:
                resp = await client.post(url, json={"model": model_name, "keep_alive": 0})
                return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    async def unload_other_resident_models(cls, keep_model: str) -> List[str]:
        """Evict idle resident models from VRAM when switching models."""
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/ps"
        evicted: List[str] = []
        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_STATUS_TIMEOUT_SECONDS) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("models", [])
                    for m in models:
                        name = m.get("name") or m.get("model", "")
                        if name and name.lower() != keep_model.lower():
                            if await cls.unload_model(name):
                                evicted.append(name)
        except Exception:
            pass
        return evicted

    @classmethod
    async def refresh(cls, force: bool = False) -> Tuple[bool, Optional[str]]:
        """Query Ollama and refresh the installed model registry.

        Returns:
            Tuple of (is_connected, error_message).
        """
        now = time.monotonic()
        ttl = getattr(settings, "ROUTING_REGISTRY_TTL_SECONDS", 5.0)

        async with cls._lock:
            if not force and cls._last_refresh_time > 0 and (now - cls._last_refresh_time) < ttl:
                return cls._is_connected, cls._last_error

            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
            try:
                async with httpx.AsyncClient(timeout=settings.OLLAMA_STATUS_TIMEOUT_SECONDS) as client:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        cls._is_connected = False
                        cls._last_error = f"Ollama returned HTTP {resp.status_code}"
                        return False, cls._last_error

                    data = resp.json()
                    raw_models = data.get("models", [])
                    if not isinstance(raw_models, list):
                        cls._is_connected = False
                        cls._last_error = "Malformed response from /api/tags"
                        return False, cls._last_error

                    # Also query /api/ps for currently loaded models
                    running_models = await cls.fetch_running_processes(client)

                    updated_models: Dict[str, DiscoveredModel] = {}
                    for item in raw_models:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("name", "")
                        if not name:
                            continue

                        details = item.get("details", {}) or {}
                        param_size = details.get("parameter_size")
                        quant = details.get("quantization_level")
                        family = details.get("family")
                        families = details.get("families", [])
                        context_len = details.get("context_length")
                        embedding_len = details.get("embedding_length")
                        caps = item.get("capabilities", [])

                        # Check if loaded via /api/ps
                        ps_info = running_models.get(name.lower()) or running_models.get(item.get("model", "").lower())
                        is_loaded = ps_info is not None
                        size_vram = ps_info.get("size_vram") if ps_info else None
                        expires_at = ps_info.get("expires_at") if ps_info else None

                        # If context_len or parameter_size is missing from tags, query /api/show
                        param_count = None
                        if context_len is None or param_size is None:
                            show_data = await cls.fetch_model_details(client, name)
                            if show_data:
                                model_info = show_data.get("model_info", {})
                                for k, v in model_info.items():
                                    if k.endswith(".context_length") and isinstance(v, int) and context_len is None:
                                        context_len = v
                                    if k.endswith("general.parameter_count") and isinstance(v, int):
                                        param_count = v
                                if not param_size and "details" in show_data:
                                    param_size = show_data["details"].get("parameter_size")

                        tier = cls._parse_tier(param_size, param_count, name)

                        disc = DiscoveredModel(
                            name=name,
                            model=item.get("model", name),
                            digest=item.get("digest", ""),
                            size_bytes=item.get("size", 0),
                            modified_at=item.get("modified_at", ""),
                            parameter_size=param_size,
                            parameter_count=param_count,
                            quantization_level=quant,
                            family=family,
                            families=families if isinstance(families, list) else [],
                            context_length=context_len,
                            embedding_length=embedding_len,
                            capabilities=caps if isinstance(caps, list) else [],
                            is_loaded=is_loaded,
                            size_vram=size_vram,
                            expires_at=expires_at,
                            tier=tier,
                        )
                        updated_models[name.lower()] = disc

                    cls._models = updated_models
                    cls._is_connected = True
                    cls._last_error = None
                    cls._last_refresh_time = now
                    return True, None

            except httpx.ConnectError:
                cls._is_connected = False
                cls._last_error = f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}."
                return False, cls._last_error
            except httpx.TimeoutException:
                cls._is_connected = False
                cls._last_error = f"Connection timed out checking Ollama at {settings.OLLAMA_BASE_URL}."
                return False, cls._last_error
            except Exception as exc:
                cls._is_connected = False
                cls._last_error = f"Unexpected error querying Ollama: {str(exc)}"
                return False, cls._last_error

    @classmethod
    async def get_model(cls, model_name: str, force_refresh: bool = False) -> Optional[DiscoveredModel]:
        """Find a model by exact name or partial tag match."""
        await cls.refresh(force=force_refresh)
        target = model_name.strip().lower()
        if target in cls._models:
            return cls._models[target]

        # Check partial / untagged match (e.g. "qwen2.5:1.5b" vs "qwen2.5:1.5b-instruct")
        for key, m in cls._models.items():
            if target in key or key in target:
                return m
        return None

    @classmethod
    async def get_all_models(cls, force_refresh: bool = False) -> List[DiscoveredModel]:
        """Return list of all discovered models."""
        await cls.refresh(force=force_refresh)
        return list(cls._models.values())

    @classmethod
    async def is_model_installed(cls, model_name: str, force_refresh: bool = False) -> bool:
        """Check if a specific model tag is installed in local Ollama."""
        model = await cls.get_model(model_name, force_refresh=force_refresh)
        return model is not None

    @classmethod
    async def get_catalog(cls, force_refresh: bool = False) -> ModelCatalogResponse:
        """Produce an API catalog response for GET /api/models."""
        is_conn, err = await cls.refresh(force=force_refresh)
        items: List[ModelCatalogItem] = []
        active_loaded: List[str] = []

        for m in cls._models.values():
            if m.is_loaded:
                active_loaded.append(m.name)
            vram_mb = round(m.size_vram / (1024 * 1024), 1) if m.size_vram else None
            items.append(
                ModelCatalogItem(
                    name=m.name,
                    parameter_size=m.parameter_size,
                    parameter_count=m.parameter_count,
                    quantization_level=m.quantization_level,
                    family=m.family,
                    context_length=m.context_length,
                    size_bytes=m.size_bytes,
                    size_mb=round(m.size_bytes / (1024 * 1024), 2),
                    tier=m.tier.value,
                    is_loaded=m.is_loaded,
                    size_vram_mb=vram_mb,
                    modified_at=m.modified_at,
                )
            )

        # Sort: small models first, then medium, then large
        tier_order = {ModelTier.SMALL.value: 1, ModelTier.MEDIUM.value: 2, ModelTier.LARGE.value: 3, ModelTier.UNKNOWN.value: 4}
        items.sort(key=lambda x: (tier_order.get(x.tier, 99), x.name))

        return ModelCatalogResponse(
            connected=is_conn,
            base_url=settings.OLLAMA_BASE_URL,
            total_models=len(items),
            models=items,
            active_loaded_models=active_loaded,
            error=err,
        )

    @classmethod
    def reset(cls) -> None:
        """Reset internal registry cache (useful in tests)."""
        cls._models = {}
        cls._last_refresh_time = 0.0
        cls._is_connected = False
        cls._last_error = None


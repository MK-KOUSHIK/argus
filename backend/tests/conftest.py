"""Pytest fixtures for Argus backend testing."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.services.tokenizer_service import TokenizerService


@pytest.fixture
async def async_client():
    """Async HTTP client fixture configured for FastAPI testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture(autouse=True)
def mock_tokenizer_service_isolation(request, monkeypatch):
    """Ensure all unit tests are fully isolated and never depend on a local Ollama daemon."""
    if (
        "ollama_integration" in request.keywords
        or "test_tokenizer_service" in request.node.nodeid
        or "test_prompt_optimization_acceptance" in request.node.nodeid
    ):
        return  # Do not mock for integration tests or acceptance tests verifying fallback provenance

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    async def _mock_measure_pair(orig_text: str, cand_text: str, model=None):
        t_orig = orig_text.strip()
        t_cand = cand_text.strip()
        if not t_orig and not t_cand:
            return 0, 0, "exact_zero"
        if not t_orig:
            return 0, len(enc.encode(t_cand)), "test_mock"
        if not t_cand:
            return len(enc.encode(t_orig)), 0, "test_mock"
        return len(enc.encode(t_orig)), len(enc.encode(t_cand)), "test_mock"

    async def _mock_measure(text: str, model=None):
        t = text.strip()
        if not t:
            return 0, "exact_zero"
        return len(enc.encode(t)), "test_mock"

    monkeypatch.setattr(TokenizerService, "measure_tokens_pair", _mock_measure_pair)
    monkeypatch.setattr(TokenizerService, "measure_tokens", _mock_measure)

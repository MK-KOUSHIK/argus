"""Regression tests for TokenizerService measurement logic and failure fallbacks."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.services.tokenizer_service import TokenizerService


@pytest.mark.asyncio
async def test_measure_tokens_pair_equal_native_counts_different_char_lengths():
    """Verify that equal native token counts are accepted even when character lengths differ."""
    orig_prompt = "This is a prompt with a noticeably longer character length than the candidate"
    cand_prompt = "Short candidate"
    assert len(orig_prompt) != len(cand_prompt)

    mock_resp_orig = httpx.Response(
        status_code=200,
        json={"prompt_eval_count": 18, "done": True},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    mock_resp_cand = httpx.Response(
        status_code=200,
        json={"prompt_eval_count": 18, "done": True},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    mock_client = AsyncMock()
    mock_client.post.side_effect = [mock_resp_orig, mock_resp_cand]

    with patch("app.services.tokenizer_service.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_cls.return_value.__aexit__.return_value = None

        orig_cnt, cand_cnt, method = await TokenizerService.measure_tokens_pair(
            orig_prompt, cand_prompt, model="qwen2.5:1.5b"
        )

        assert orig_cnt == 18
        assert cand_cnt == 18
        assert method == "ollama_native"
        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_measure_tokens_pair_different_native_counts():
    """Verify that different native token counts are accurately returned with ollama_native."""
    orig_prompt = "Please explain the Raft consensus algorithm with diagrams."
    cand_prompt = "Explain Raft consensus."

    mock_resp_orig = httpx.Response(
        status_code=200,
        json={"prompt_eval_count": 25, "done": True},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    mock_resp_cand = httpx.Response(
        status_code=200,
        json={"prompt_eval_count": 10, "done": True},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    mock_client = AsyncMock()
    mock_client.post.side_effect = [mock_resp_orig, mock_resp_cand]

    with patch("app.services.tokenizer_service.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_cls.return_value.__aexit__.return_value = None

        orig_cnt, cand_cnt, method = await TokenizerService.measure_tokens_pair(
            orig_prompt, cand_prompt, model="qwen2.5:1.5b"
        )

        assert orig_cnt == 25
        assert cand_cnt == 10
        assert method == "ollama_native"


@pytest.mark.asyncio
async def test_measure_tokens_pair_identical_prompts():
    """Verify measurement for identical prompts returns exact counts."""
    prompt = "Identical prompt text for evaluation."

    mock_resp = httpx.Response(
        status_code=200,
        json={"prompt_eval_count": 8, "done": True},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    mock_client = AsyncMock()
    mock_client.post.side_effect = [mock_resp, mock_resp]

    with patch("app.services.tokenizer_service.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_cls.return_value.__aexit__.return_value = None

        orig_cnt, cand_cnt, method = await TokenizerService.measure_tokens_pair(
            prompt, prompt, model="qwen2.5:1.5b"
        )

        assert orig_cnt == 8
        assert cand_cnt == 8
        assert method == "ollama_native"


@pytest.mark.asyncio
async def test_measure_tokens_pair_empty_prompts():
    """Verify measurement for empty prompts returns exact_zero without network calls."""
    orig_cnt, cand_cnt, method = await TokenizerService.measure_tokens_pair("   ", "")
    assert orig_cnt == 0
    assert cand_cnt == 0
    assert method == "exact_zero"


@pytest.mark.asyncio
async def test_measure_tokens_pair_invalid_or_missing_native_measurements():
    """Verify fallback to tiktoken when Ollama returns missing or non-integer eval counts."""
    orig_prompt = "Original test prompt for fallback verification."
    cand_prompt = "Candidate prompt."

    mock_resp_invalid = httpx.Response(
        status_code=200,
        json={"prompt_eval_count": None, "done": True},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp_invalid

    with patch("app.services.tokenizer_service.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_cls.return_value.__aexit__.return_value = None

        orig_cnt, cand_cnt, method = await TokenizerService.measure_tokens_pair(
            orig_prompt, cand_prompt, model="qwen2.5:1.5b"
        )

        # Must fall back to tiktoken_cl100k
        assert method == "tiktoken_cl100k"
        assert orig_cnt > 0
        assert cand_cnt > 0

@pytest.mark.asyncio
async def test_measure_tokens_pair_connection_failure():
    """Verify fallback to tiktoken when Ollama connection fails with ConnectError."""
    orig_prompt = "Original prompt test."
    cand_prompt = "Cand prompt."

    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ConnectError("Connection to Ollama refused on 127.0.0.1:11434")

    with patch("app.services.tokenizer_service.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_cls.return_value.__aexit__.return_value = None

        orig_cnt, cand_cnt, method = await TokenizerService.measure_tokens_pair(
            orig_prompt, cand_prompt, model="qwen2.5:1.5b"
        )

        assert method == "tiktoken_cl100k"
        assert orig_cnt > 0
        assert cand_cnt > 0

@pytest.mark.asyncio
async def test_measure_tokens_pair_server_error_500():
    """Verify fallback to tiktoken when Ollama returns HTTP 500 error."""
    orig_prompt = "Original prompt test."
    cand_prompt = "Cand prompt."

    mock_resp_500 = httpx.Response(
        status_code=500,
        json={"error": "model error"},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp_500

    with patch("app.services.tokenizer_service.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_cls.return_value.__aexit__.return_value = None

        orig_cnt, cand_cnt, method = await TokenizerService.measure_tokens_pair(
            orig_prompt, cand_prompt, model="qwen2.5:1.5b"
        )

        assert method == "tiktoken_cl100k"
        assert orig_cnt > 0
        assert cand_cnt > 0

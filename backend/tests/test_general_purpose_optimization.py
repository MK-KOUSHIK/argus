"""Unit, integration, and regression tests for Argus General-Purpose Semantic-Preserving Optimization Engine."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

from app.schemas.optimizer import OptimizeRequest
from app.services.task_analyzer import TaskAnalyzer, TaskAnalysisResult
from app.services.candidate_validator import CandidateValidator, ValidationResult
from app.services.prompt_rewriter import PromptRewriter
from app.services.tokenizer_service import TokenizerService


def test_general_purpose_entity_extraction():
    """Verify data-driven extraction of domain entities and technical identifiers."""
    prompt = (
        "Configure Prometheus to scrape metrics from Kubernetes cluster using ServiceMonitors. "
        "Also setup Grafana dashboard for Docker containers running PostgreSQL and Redis."
    )
    entities = TaskAnalyzer._extract_entities(prompt)
    assert "Prometheus" in entities
    assert "Kubernetes" in entities
    assert "ServiceMonitors" in entities
    assert "Grafana" in entities
    assert "Docker" in entities
    assert "PostgreSQL" in entities
    assert "Redis" in entities


def test_general_purpose_facts_and_constants_extraction():
    """Verify extraction of numbers, units, versions, dates, times, currencies, and emails."""
    prompt = (
        "Generate an RSA 4096-bit private key on Ubuntu 22.04 LTS. "
        "Maintenance window on November 15, 2026 from 02:00 UTC to 04:00 UTC. "
        "Downtime must not exceed 120 minutes. Cost is $50,000. Contact ops@argus.internal."
    )
    facts = TaskAnalyzer._extract_facts_and_constants(prompt)
    assert any("4096-bit" in f for f in facts)
    assert any("22.04 LTS" in f for f in facts)
    assert any("November 15, 2026" in f for f in facts)
    assert any("02:00 UTC" in f for f in facts)
    assert any("04:00 UTC" in f for f in facts)
    assert any("120 minutes" in f for f in facts)
    assert any("$50,000" in f for f in facts)
    assert any("ops@argus.internal" in f for f in facts)


def test_general_purpose_negation_extraction():
    """Verify extraction of real negative constraints while filtering conversational filler."""
    prompt_real_neg = (
        "Write a SQL query for PostgreSQL 14+. Filter where count > 1 and do not use subqueries. "
        "Do not drop corrupted records silently."
    )
    negations = TaskAnalyzer._extract_generalized_negations(prompt_real_neg)
    assert any("subqueries" in n.lower() for n in negations)
    assert any("corrupted records silently" in n.lower() for n in negations)

    prompt_filler_neg = "Build a Python program. Make it good and don't miss anything important."
    filler_negations = TaskAnalyzer._extract_generalized_negations(prompt_filler_neg)
    assert len(filler_negations) == 0  # 'don't miss anything' is conversational filler


def test_general_purpose_conditions_extraction():
    """Verify extraction of comparison operations, SQL clauses, and boundary constraints."""
    prompt = (
        "Query PostgreSQL for Enterprise accounts created after 2026-01-15 with balance > $50,000. "
        "Order by count descending and ensure email under 100 words."
    )
    conditions = TaskAnalyzer._extract_conditions(prompt)
    assert any("created after" in c for c in conditions)
    assert any("balance > $50,000" in c or "50,000" in c for c in conditions)
    assert any("order by" in c.lower() for c in conditions)
    assert any("under 100 words" in c.lower() for c in conditions)


def test_general_purpose_grammar_issue_detection():
    """Verify detection of informal slang, broken syntax, and typos without discarding information."""
    prompt = "can u pls write me a functon in python for calculate fibonaci number fastly? thx!"
    issues = TaskAnalyzer._detect_grammatical_issues(prompt)
    assert len(issues) >= 3
    assert any("pls" in i.lower() for i in issues)
    assert any("function" in i.lower() for i in issues)
    assert any("fibonacci" in i.lower() for i in issues)

    # Verify that analyzer still extracts entities and actions despite broken English
    analysis = TaskAnalyzer.analyze(prompt)
    assert any(e.lower() == "python" for e in analysis.entities)
    assert analysis.primary_action in {"write", "calculate"}


def test_semantic_validator_catches_missing_entity():
    """CandidateValidator rejects candidates that drop required technical entities."""
    orig = "Show me how to configure Prometheus to scrape metrics using ServiceMonitors."
    cand_missing_tool = "Show me how to configure Prometheus to scrape metrics."
    analysis = TaskAnalyzer.analyze(orig)

    val = CandidateValidator.validate(orig, cand_missing_tool, analysis)
    assert val.is_valid is False
    assert "ServiceMonitors" in (val.reason or "")


def test_semantic_validator_catches_missing_number_or_fact():
    """CandidateValidator rejects candidates that drop numerical values, times, or dates."""
    orig = "Generate an RSA 4096-bit key on Ubuntu 22.04 LTS."
    cand_dropped_bits = "Generate an RSA key on Ubuntu 22.04 LTS."
    analysis = TaskAnalyzer.analyze(orig)

    val = CandidateValidator.validate(orig, cand_dropped_bits, analysis)
    assert val.is_valid is False
    assert "4096-bit" in (val.reason or "")


def test_semantic_validator_catches_inverted_negation():
    """CandidateValidator rejects candidates that invert negative prohibitions."""
    orig = "Write a SQL query to find duplicate users. Do not use subqueries."
    cand_inverted = "Write a SQL query to find duplicate users using subqueries."
    analysis = TaskAnalyzer.analyze(orig)

    val = CandidateValidator.validate(orig, cand_inverted, analysis)
    assert val.is_valid is False
    assert "subqueries" in (val.reason or "")


def test_semantic_validator_catches_missing_condition():
    """CandidateValidator rejects candidates that drop filtering conditions."""
    orig = "Query database for accounts where balance > $50,000 and created after 2026-01-15."
    cand_missing_cond = "Query database for accounts created after 2026-01-15."
    analysis = TaskAnalyzer.analyze(orig)

    val = CandidateValidator.validate(orig, cand_missing_cond, analysis)
    assert val.is_valid is False
    assert any(term in (val.reason or "") for term in ["50,000", "balance"])


@pytest.mark.asyncio
async def test_grammar_corrected_shorter_candidate_accepted(async_client: AsyncClient):
    """End-to-end API: Grammatically corrected candidate with strictly fewer tokens is accepted."""
    from tests.test_optimizer import create_mock_client
    orig = "can u pls make pythn script for get all image url from web page thx"
    cand = "Write a Python script to extract all image URLs from a web page."

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": cand},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)), \
         patch.object(TokenizerService, "measure_tokens_pair", return_value=(18, 14, "test_mock")):

        res = await async_client.post("/api/optimize", json={"prompt": orig, "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is True
        assert data["optimized_prompt"] == cand
        assert data["actual_tokens_saved"] == 4
        assert data["outcome"] == "validated_rewrite"


@pytest.mark.asyncio
async def test_grammar_corrected_longer_candidate_rejected(async_client: AsyncClient):
    """End-to-end API: Grammatically corrected candidate that inflates tokens is strictly rejected."""
    from tests.test_optimizer import create_mock_client
    orig = "can u pls make pythn script for get all image url from web page thx"
    longer_cand = (
        "I would like to kindly ask if you could please develop a comprehensive Python script "
        "designed specifically to extract every single image URL from a given web page."
    )

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": longer_cand},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)), \
         patch.object(TokenizerService, "measure_tokens_pair", return_value=(18, 29, "test_mock")):

        res = await async_client.post("/api/optimize", json={"prompt": orig, "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is False
        assert data["optimized_prompt"] == orig
        assert data["actual_tokens_saved"] == 0
        assert data["outcome"] == "unchanged_original"

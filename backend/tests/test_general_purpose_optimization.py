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


def test_multi_word_directional_relation_extraction():
    """Verify TaskAnalyzer extracts multi-word phrases for source and destination in directional relations."""
    p1 = "Transfer $500 from savings account to checking account immediately."
    a1 = TaskAnalyzer.analyze(p1)
    assert ("savings account", "to", "checking account") in a1.entity_relations

    p2 = "Migrate user data from primary database to backup replica safely."
    a2 = TaskAnalyzer.analyze(p2)
    assert ("primary database", "to", "backup replica") in a2.entity_relations


def test_directional_reversal_rejected_by_validator():
    """Verify CandidateValidator rejects candidates that invert directional source/destination."""
    prompt = "Transfer $500 from savings account to checking account immediately."
    analysis = TaskAnalyzer.analyze(prompt)

    # Inversion pattern 1: from Destination to Source
    cand_inv1 = "Transfer $500 from checking account to savings account immediately."
    res1 = CandidateValidator.validate(prompt, cand_inv1, analysis)
    assert res1.is_valid is False
    assert "inverted directional relationship" in res1.reason

    # Inversion pattern 2: to Source from Destination
    cand_inv2 = "Transfer $500 to savings account from checking account immediately."
    res2 = CandidateValidator.validate(prompt, cand_inv2, analysis)
    assert res2.is_valid is False
    assert "inverted directional relationship" in res2.reason

    # Preservation pattern: unchanged direction
    cand_ok = "Transfer $500 from savings account to checking account now."
    res_ok = CandidateValidator.validate(prompt, cand_ok, analysis)
    assert res_ok.is_valid is True


def test_non_directional_from_to_ranges_ignored():
    """Verify non-directional uses of 'from ... to ...' (numeric, temporal, day ranges) are not extracted as relations."""
    assert TaskAnalyzer.analyze("Count from 1 to 10.").entity_relations == []
    assert TaskAnalyzer.analyze("The store is open from 9am to 5pm.").entity_relations == []
    assert TaskAnalyzer.analyze("Office hours are from Monday to Friday.").entity_relations == []
    assert TaskAnalyzer.analyze("Read the document from start to finish.").entity_relations == []


def test_generic_named_entity_extraction():
    """Verify extraction of multi-word and proper noun entities beyond hardcoded tech names."""
    prompt = (
        "Alliance University is located in Bengaluru, Karnataka. "
        "Wayne Enterprises partnered with Globex Industries in Gotham City."
    )
    entities = TaskAnalyzer._extract_entities(prompt)
    assert "Alliance University" in entities
    assert "Bengaluru" in entities
    assert "Karnataka" in entities
    assert "Wayne Enterprises" in entities
    assert "Globex Industries" in entities
    assert "Gotham City" in entities

    # Imperative verbs and common prompt instructions at sentence start must not be treated as proper nouns
    inst_prompt = "Select all records. Create a report. Explain the workflow. Optimize the query."
    inst_entities = TaskAnalyzer._extract_entities(inst_prompt)
    assert "Select" not in inst_entities
    assert "Create" not in inst_entities
    assert "Explain" not in inst_entities
    assert "Optimize" not in inst_entities


def test_candidate_validator_rejects_missing_generic_entity():
    """Verify CandidateValidator rejects candidates that drop generic proper nouns / domain entities."""
    prompt = "Provide historical background and campus facilities for Alliance University in Bengaluru, Karnataka."
    analysis = TaskAnalyzer.analyze(prompt)
    assert "Alliance University" in analysis.entities
    assert "Bengaluru" in analysis.entities
    assert "Karnataka" in analysis.entities

    # Dropping multi-word entity
    cand_missing_org = "Provide historical background and campus facilities in Bengaluru, Karnataka."
    res1 = CandidateValidator.validate(prompt, cand_missing_org, analysis)
    assert res1.is_valid is False
    assert "Alliance University" in res1.reason

    # Dropping city entity
    cand_missing_city = "Provide historical background and campus facilities for Alliance University in Karnataka."
    res2 = CandidateValidator.validate(prompt, cand_missing_city, analysis)
    assert res2.is_valid is False
    assert "Bengaluru" in res2.reason

    # Preserving all entities
    cand_valid = "Summarize the history and facilities of Alliance University in Bengaluru, Karnataka."
    res3 = CandidateValidator.validate(prompt, cand_valid, analysis)
    assert res3.is_valid is True


def test_directional_preposition_boundary_and_article_handling():
    """Verify directional relations exclude trailing prepositions and strip leading determiners."""
    p1 = "Migrate user data from the primary database to a backup replica for disaster recovery."
    a1 = TaskAnalyzer.analyze(p1)
    assert ("primary database", "to", "backup replica") in a1.entity_relations

    p2 = "Transfer $500 from the savings account to my checking account immediately."
    a2 = TaskAnalyzer.analyze(p2)
    assert ("savings account", "to", "checking account") in a2.entity_relations

    # Candidate drops articles when inverting direction: validator must still catch it
    cand_inv_no_articles = "Transfer $500 from checking account to savings account immediately."
    res1 = CandidateValidator.validate(p2, cand_inv_no_articles, a2)
    assert res1.is_valid is False
    assert "inverted directional relationship" in res1.reason

    # Candidate retains articles when inverting direction: validator must still catch it
    cand_inv_with_articles = "Transfer $500 from the checking account to the savings account immediately."
    res2 = CandidateValidator.validate(p2, cand_inv_with_articles, a2)
    assert res2.is_valid is False
    assert "inverted directional relationship" in res2.reason


def test_markdown_heading_and_compound_tech_entities():
    """Verify markdown headings are stripped and compound tech entities are preserved."""
    prompt = (
        "# Architectural Plan\n"
        "### Project Requirements\n"
        "The Python Software Foundation released updates for React Native and Docker Desktop on Azure."
    )
    entities = TaskAnalyzer._extract_entities(prompt)
    assert "Architectural Plan" not in entities
    assert "Project Requirements" not in entities
    assert "Python Software Foundation" in entities
    assert "React Native" in entities
    assert "Docker Desktop" in entities
    assert "Azure" in entities

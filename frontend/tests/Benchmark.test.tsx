import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BenchmarkPage } from '../src/pages/BenchmarkPage';
import { apiService } from '../src/services/api';
import { BenchmarkReport } from '../src/types';

const MOCK_REPORT: BenchmarkReport = {
  metadata: {
    version: 'Argus Phase 4 Benchmark v1.0',
    dataset_count: 5,
    model: 'gpt-4o',
    is_live: false,
  },
  summary: {
    total_cases: 5,
    passed_cases: 5,
    pass_rate_percent: 100.0,
    baseline_prompt_tokens: 254,
    optimized_prompt_tokens: 169,
    prompt_tokens_saved: 85,
    prompt_tokens_reduction_percent: 33.46,
    baseline_total_tokens: 504,
    optimized_total_tokens: 419,
    total_tokens_saved: 85,
    total_tokens_reduction_percent: 16.87,
    baseline_cost_usd: 0.003135,
    optimized_cost_usd: 0.002923,
    cost_saved_usd: 0.000212,
    cost_savings_percent: 6.78,
    avg_baseline_latency_ms: 45.0,
    avg_optimizer_latency_ms: 373.8,
    avg_optimized_total_latency_ms: 418.8,
    avg_latency_overhead_ms: 373.8,
    constraint_retention_rate_percent: 100.0,
    format_compliance_rate_percent: 100.0,
    provider_name: 'mock',
    model_name: 'gpt-4o',
    is_live: false,
    timestamp: '2026-10-22T10:00:00Z',
  },
  case_results: [
    {
      case_id: 'case-1-simple-informational',
      case_name: 'TCP vs UDP Summary',
      category: 'simple_informational',
      description: 'Simple informational prompt with clear deliverable.',
      comparison: {
        original_prompt: 'What is the key architectural difference between TCP and UDP?',
        optimized_prompt: 'Summarize key architectural differences between TCP and UDP in 3 bullet points.',
        mode: 'concise',
        model: 'gpt-4o',
        optimizer_details: {
          strategy: 'compress',
          applied_strategy: 'compress',
          validation_status: 'passed',
          outcome: 'validated_rewrite',
        },
        baseline: {
          path_name: 'baseline',
          prompt_used: 'What is the key architectural difference between TCP and UDP?',
          tokenizer_tokens: { token_count: 14, char_count: 61, tokenizer_name: 'tiktoken', is_exact: true },
          provider_usage: { prompt_tokens: 14, completion_tokens: 40, total_tokens: 54 },
          cost: { input_cost_usd: 0.000035, output_cost_usd: 0.00040, total_cost_usd: 0.000435, is_estimated: false, currency: 'USD' },
          llm_response: 'TCP is connection-oriented, UDP is connectionless.',
          latency_ms: 45,
          total_latency_ms: 45,
          is_mock: true,
        },
        optimized: {
          path_name: 'optimized',
          prompt_used: 'Summarize key architectural differences between TCP and UDP in 3 bullet points.',
          tokenizer_tokens: { token_count: 12, char_count: 78, tokenizer_name: 'tiktoken', is_exact: true },
          provider_usage: { prompt_tokens: 12, completion_tokens: 40, total_tokens: 52 },
          cost: { input_cost_usd: 0.000030, output_cost_usd: 0.00040, total_cost_usd: 0.000430, is_estimated: false, currency: 'USD' },
          llm_response: 'TCP is connection-oriented, UDP is connectionless.',
          latency_ms: 45,
          total_latency_ms: 220,
          is_mock: true,
        },
        prompt_tokens_saved: 2,
        prompt_tokens_saved_percent: 14.29,
        total_tokens_saved: 2,
        total_tokens_saved_percent: 3.7,
        cost_comparison: {
          baseline_cost_usd: 0.000435,
          optimized_cost_usd: 0.000430,
          cost_saved_usd: 0.000005,
          savings_percent: 1.15,
          is_estimated: false,
        },
        latency_overhead_ms: 175,
        constraints_preserved: true,
        is_mock: true,
      },
      baseline_quality: {
        prompt_constraints_preserved: true,
        output_prohibitions_respected: true,
        output_required_terms_present: true,
        output_format_valid: true,
        quality_score: 1.0,
        is_passing: true,
        details: ['All verified'],
      },
      optimized_quality: {
        prompt_constraints_preserved: true,
        output_prohibitions_respected: true,
        output_required_terms_present: true,
        output_format_valid: true,
        quality_score: 1.0,
        is_passing: true,
        details: ['All verified'],
      },
      is_passing: true,
      summary_notes: 'Prompt optimized with constraint retention.',
    },
  ],
};

describe('BenchmarkPage Component', () => {
  beforeEach(() => {
    vi.spyOn(apiService, 'runBenchmark').mockResolvedValue(MOCK_REPORT);
  });

  it('renders benchmark header and summary KPI cards', async () => {
    render(<BenchmarkPage mainLLMStatus={{ provider: 'openai', model: 'gpt-4o', is_configured: false, base_url: '', message: '' }} />);

    expect(screen.getByText('Phase 4 Reproducible Benchmark Suite')).toBeInTheDocument();
    expect(screen.getByText('Empirical Cost & Quality Benchmark')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('33.46%')).toBeInTheDocument();
      expect(screen.getByText('6.78%')).toBeInTheDocument();
      expect(screen.getAllByText('100%').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('expands case details to view prompt diff and constraint checklist', async () => {
    render(<BenchmarkPage mainLLMStatus={{ provider: 'openai', model: 'gpt-4o', is_configured: false, base_url: '', message: '' }} />);

    await waitFor(() => {
      expect(screen.getByText('TCP vs UDP Summary')).toBeInTheDocument();
    });

    const caseCard = screen.getByText('TCP vs UDP Summary');
    fireEvent.click(caseCard);

    await waitFor(() => {
      expect(screen.getByText(/Original Input Prompt \(Baseline\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Argus Optimized Prompt/i)).toBeInTheDocument();
      expect(screen.getByText('Automated Constraint & Safety Verification')).toBeInTheDocument();
      expect(screen.getByText('Prompt Constraints Preserved')).toBeInTheDocument();
    });
  });
});

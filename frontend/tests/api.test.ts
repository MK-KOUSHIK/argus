import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { apiService, transformOptimizeResponse } from '@/services/api';

describe('ApiService compareOutputs Telemetry Mapping (DEFECT-01)', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('preserves all top-level telemetry fields from backend ComparisonResponse', async () => {
    const backendPayload = {
      original_execution: {
        prompt: 'Original prompt',
        response: 'Original response',
        model: 'qwen2.5:1.5b',
        provider: 'ollama',
        prompt_tokens: 30,
        output_tokens: 20,
        total_tokens: 50,
        duration_ms: 200,
        prompt_eval_duration_ms: 50,
        status: 'completed',
        done_reason: 'stop',
      },
      optimized_execution: {
        prompt: 'Optimized prompt',
        response: 'Optimized response',
        model: 'qwen2.5:1.5b',
        provider: 'ollama',
        prompt_tokens: 20,
        output_tokens: 18,
        total_tokens: 38,
        duration_ms: 150,
        prompt_eval_duration_ms: 30,
        status: 'completed',
        done_reason: 'stop',
      },
      metrics: {
        prompt_tokens_saved: 10,
        prompt_token_reduction_percent: 33.3,
        output_tokens_diff: -2,
        output_token_change_percent: -10,
        total_tokens_saved: 12,
        total_token_reduction_percent: 24,
        latency_delta_ms: -50,
        latency_reduction_percent: 25,
        semantic_quality: 'not_evaluated',
        semantic_quality_reason: 'Verification needed',
      },
      downstream_model: 'qwen2.5:1.5b',
      effective_settings: {
        model: 'qwen2.5:1.5b',
        num_ctx: 4096,
        num_predict: 128,
        temperature: 0.2,
        top_p: 0.9,
        context_window_shared: true,
      },
      trials: 2,
      warmup: true,
      warmup_attempted: true,
      warmup_succeeded: false,
      warmup_error: 'Ollama daemon connection timed out during warm-up phase.',
      trial_measurements: [
        {
          trial_index: 0,
          execution_order: 1,
          prompt_tokens: 30,
          output_tokens: 20,
          total_tokens: 50,
          latency: { inference_duration_ms: 190 },
          done_reason: 'stop',
          response: 'Trial 0 response',
        },
      ],
      aggregate_latency: {
        mean_inference_duration_ms: 175,
      },
      timestamp: '2026-11-22T19:30:00Z',
      optimizer_details: {
        original_prompt: 'Original prompt',
        optimized_prompt: 'Optimized prompt',
        mode: 'balanced',
        target_model: 'gemini-1-5-flash',
        task_category: 'general',
        strategy: 'compress',
        applied_strategy: 'compress',
        outcome: 'validated_rewrite',
        length_result: 'compressed',
        validation_status: 'passed',
        is_preserved: false,
        explicit_constraints: [],
        preserved_constraints: [],
        original_metrics: { char_count: 15, estimated_tokens: 4 },
        optimized_metrics: { char_count: 16, estimated_tokens: 4 },
        token_reduction_percent: 20,
        token_change_percent: -20,
        token_savings: 1,
        char_delta: 1,
        candidate_prompt: 'Optimized candidate',
        original_token_count: 30,
        candidate_token_count: 20,
        dispatched_token_count: 20,
        actual_tokens_saved: 10,
        actual_reduction_percent: 33.3,
        candidate_accepted: true,
        rejection_reason: null,
        token_measurement_method: 'ollama_native',
        execution_time_ms: 85,
        model: 'qwen2.5:1.5b',
        timestamp: '2026-11-22T19:30:00Z',
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => backendPayload,
    } as Response);

    const result = await apiService.compareOutputs({
      prompt: 'Test prompt',
      downstream_model: 'qwen2.5:1.5b',
    });

    // Verify all top-level telemetry survives mapping
    expect(result.warmup_attempted).toBe(true);
    expect(result.warmup_succeeded).toBe(false);
    expect(result.warmup_error).toBe('Ollama daemon connection timed out during warm-up phase.');
    expect(result.effective_settings?.num_ctx).toBe(4096);
    expect(result.trials).toBe(2);
    expect(result.warmup).toBe(true);
    expect(result.trial_measurements?.length).toBe(1);
    expect(result.aggregate_latency?.mean_inference_duration_ms).toBe(175);
    expect(result.optimizerDetails.actualTokensSaved).toBe(10);
    expect(result.optimizerDetails.candidateAccepted).toBe(true);
    expect(result.optimizerDetails.tokenMeasurementMethod).toBe('ollama_native');
  });

  it('transformOptimizeResponse guards against null and undefined inputs', () => {
    expect(() => transformOptimizeResponse(null)).toThrow(
      'transformOptimizeResponse received null or undefined data'
    );
    expect(() => transformOptimizeResponse(undefined)).toThrow(
      'transformOptimizeResponse received null or undefined data'
    );
  });
});

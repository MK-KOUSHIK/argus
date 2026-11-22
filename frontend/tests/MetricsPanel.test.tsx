import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MetricsPanel, formatMetricNumber } from '../src/components/MetricsPanel';
import { OptimizationResult } from '../src/types';

const BASE_RESULT: OptimizationResult = {
  originalPrompt: 'Explain how neural networks learn.',
  optimizedPrompt: 'Explain neural network training mechanics.',
  mode: 'balanced',
  targetModel: 'gpt-4o',
  taskCategory: 'general',
  strategy: 'compress',
  appliedStrategy: 'compress',
  outcome: 'validated_rewrite',
  lengthResult: 'compressed',
  validationStatus: 'passed',
  isPreserved: false,
  explicitConstraints: ['Preserve concept explanations'],
  preservedConstraints: ['Key terminology retained'],
  originalMetrics: { charCount: 35, estimatedTokens: 9 },
  optimizedMetrics: { charCount: 41, estimatedTokens: 10 },
  tokenReductionPercent: 0,
  tokenChangePercent: 11,
  tokenSavings: 0,
  charDelta: 6,
  executionTimeMs: 450,
  model: 'qwen2.5:1.5b',
  timestamp: new Date().toISOString(),
  isSimulated: false,
};

describe('MetricsPanel Defensive Rendering & Crash Prevention', () => {
  it('formatMetricNumber correctly formats finite numbers and handles null, undefined, NaN, Infinity', () => {
    expect(formatMetricNumber(12.3456, 2)).toBe('12.35');
    expect(formatMetricNumber(0, 2)).toBe('0.00');
    expect(formatMetricNumber(-5.2, 1)).toBe('-5.2');
    expect(formatMetricNumber(null)).toBe('N/A');
    expect(formatMetricNumber(undefined)).toBe('N/A');
    expect(formatMetricNumber(NaN)).toBe('N/A');
    expect(formatMetricNumber(Infinity)).toBe('N/A');
    expect(formatMetricNumber(-Infinity)).toBe('N/A');
    expect(formatMetricNumber(null, 2, 'n/a')).toBe('n/a');
  });

  it('renders gracefully without crashing when result is null or undefined', () => {
    const { container: c1 } = render(<MetricsPanel result={null} />);
    expect(c1).toBeEmptyDOMElement();

    const { container: c2 } = render(<MetricsPanel result={undefined} />);
    expect(c2).toBeEmptyDOMElement();
  });

  it('renders without runtime error when compression object is present but compression_time_ms is undefined', () => {
    // This exact scenario caused the confirmed runtime crash:
    // TypeError: Cannot read properties of undefined (reading 'toFixed')
    const resultWithBackendCompression: OptimizationResult = {
      ...BASE_RESULT,
      compression: {
        enabled: true,
        detected_content_type: 'code',
        strategy_applied: 'code_ast',
        original_tokens: 50,
        compressed_tokens: 35,
        reduction_percent: 30.0,
        validation_status: 'passed',
        // Neither compression_time_ms nor processing_time_ms provided
      } as any,
    };

    render(<MetricsPanel result={resultWithBackendCompression} />);
    expect(screen.getByText('Content-Aware Compression Engine')).toBeInTheDocument();
    // Engine Latency should render N/A instead of crashing
    expect(screen.getAllByText('N/A').length).toBeGreaterThan(0);
  });

  it('renders correctly when compression has raw backend processing_time_ms', () => {
    const resultWithProcessingTime: OptimizationResult = {
      ...BASE_RESULT,
      compression: {
        enabled: true,
        detected_content_type: 'json',
        strategy_applied: 'json_structural',
        original_tokens: 120,
        compressed_tokens: 80,
        reduction_percent: 33.3,
        processing_time_ms: 1.456,
        validation_status: 'passed',
      } as any,
    };

    render(<MetricsPanel result={resultWithProcessingTime} />);
    expect(screen.getByText('1.46 ms')).toBeInTheDocument();
    expect(screen.getByText('Applied: json_structural')).toBeInTheDocument();
  });

  it('renders valid finite numbers with toFixed when compression_time_ms is present', () => {
    const resultWithCompressionTime: OptimizationResult = {
      ...BASE_RESULT,
      compression: {
        applied: true,
        content_type: 'code',
        strategy: 'ast_minify',
        original_chars: 500,
        compressed_chars: 300,
        char_reduction_percent: 40.0,
        compression_time_ms: 2.75,
        reversible: false,
        validation_passed: true,
        reason: 'AST tree minified cleanly',
      },
    };

    render(<MetricsPanel result={resultWithCompressionTime} />);
    expect(screen.getByText('2.75 ms')).toBeInTheDocument();
    expect(screen.getByText('-40.0% reduction')).toBeInTheDocument();
    expect(screen.getByText('500 → 300')).toBeInTheDocument();
  });

  it('renders N/A and does not crash when metric values are null or NaN', () => {
    const resultWithNullsAndNaNs: OptimizationResult = {
      ...BASE_RESULT,
      executionTimeMs: NaN,
      tokenSavings: null as any,
      tokenReductionPercent: null as any,
      charDelta: null as any,
      compression: {
        applied: true,
        content_type: 'text',
        strategy: 'dedup',
        compression_time_ms: null as any,
        char_reduction_percent: NaN,
        original_chars: null as any,
        compressed_chars: null as any,
      } as any,
      routing: {
        routing_status: 'routed',
        selected_model: 'qwen2.5:1.5b',
        target_model_tier: 'small',
        heuristic_margin: NaN,
        routing_duration_ms: null as any,
        reason: 'Optimal fit',
        complexity: {
          task_category: 'general',
          heuristic_score: null as any, // Should render N/A and not be coerced to 0%
          complexity_factors: [],
        },
        resources: {
          vram_free_mb: null as any,
          vram_measurement_type: 'none',
        },
      } as any,
    };

    render(<MetricsPanel result={resultWithNullsAndNaNs} />);
    // Verify that N/A is rendered for unavailable metrics without crash
    const naElements = screen.getAllByText('N/A');
    expect(naElements.length).toBeGreaterThan(0);
    // Confirm heuristic score is not falsely coersed to "0% (0.00)"
    expect(screen.queryByText('0% (0.00)')).not.toBeInTheDocument();
  });

  it('renders partial responses safely when optional blocks (routing, outputBudget, telemetry) are omitted', () => {
    const minimalResult: OptimizationResult = {
      originalPrompt: 'Minimal prompt',
      optimizedPrompt: 'Optimized',
      mode: 'balanced',
      targetModel: 'gpt-4o',
      taskCategory: 'general',
      explicitConstraints: [],
      originalMetrics: { charCount: 14, estimatedTokens: 4 },
      optimizedMetrics: { charCount: 9, estimatedTokens: 3 },
      tokenReductionPercent: 25,
      tokenChangePercent: -25,
      tokenSavings: 1,
      charDelta: -5,
      strategy: 'compress',
      validationStatus: 'passed',
      isPreserved: false,
      preservedConstraints: [],
      executionTimeMs: 120,
      model: 'qwen2.5:1.5b',
      timestamp: new Date().toISOString(),
    };

    render(<MetricsPanel result={minimalResult} />);
    expect(screen.getByText('Optimization & Efficiency Metrics')).toBeInTheDocument();
    expect(screen.getByText('-25%')).toBeInTheDocument();
    expect(screen.getByText('120 ms')).toBeInTheDocument();
    // Sections for missing objects should simply not render
    expect(screen.queryByText('Content-Aware Compression Engine')).not.toBeInTheDocument();
    expect(screen.queryByText('Local Model Routing Decision (Phase 5)')).not.toBeInTheDocument();
  });

  it('catches unexpected child crashes via Error Boundary without breaking the app', () => {
    // Spy on console.error to keep test output clean
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    // Create a result that triggers an unexpected error during property access
    const poisonousResult: any = {
      get originalMetrics() {
        throw new Error('Unexpected crash simulation in child');
      },
    };

    render(<MetricsPanel result={poisonousResult} />);

    expect(
      screen.getByText(/Unable to display complete metrics panel/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Optimization results remain available/i)
    ).toBeInTheDocument();

    consoleErrorSpy.mockRestore();
  });

  it('renders Candidate Rejected badge, alert banner, and 0% savings when candidate rewrite is rejected', () => {
    const rejectedResult: OptimizationResult = {
      ...BASE_RESULT,
      candidateAccepted: false,
      rejectionReason: 'Candidate token count (49) is greater than original (46). No token reduction achieved.',
      candidatePrompt: 'A longer rewritten prompt that expanded tokens',
      originalTokenCount: 46,
      candidateTokenCount: 49,
      dispatchedTokenCount: 46,
      actualTokensSaved: 0,
      actualReductionPercent: 0,
      tokenMeasurementMethod: 'ollama_native',
      optimizedPrompt: BASE_RESULT.originalPrompt,
      isPreserved: true,
      outcome: 'unchanged_original',
    };

    render(<MetricsPanel result={rejectedResult} />);

    // Verify rejection badge
    expect(screen.getByText('Candidate Rejected — Original Preserved')).toBeInTheDocument();

    // Verify rejection banner with reason and token breakdown
    expect(
      screen.getByText('Candidate Rewrite Rejected — Original Preserved')
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Candidate token count \(49\) is greater than original \(46\)/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Original: 46 tokens • Candidate: 49 tokens • Dispatched: 46 tokens/i)
    ).toBeInTheDocument();

    // Verify token change displays 0% and explains rejection
    expect(screen.getByText('0%')).toBeInTheDocument();
    expect(screen.getByText('Candidate rejected: original retained')).toBeInTheDocument();

    // Verify native Ollama tokenizer indicator
    expect(screen.getByText('Native Ollama tokenizer')).toBeInTheDocument();
  });
});

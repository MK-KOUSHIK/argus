import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { OutputComparisonView } from '@/components/OutputComparisonView';
import { WorkspacePage } from '@/pages/WorkspacePage';
import { apiService } from '@/services/api';
import { ComparisonResult, OllamaStatus } from '@/types';

vi.mock('@/services/api', () => ({
  apiService: {
    getHealth: vi.fn(),
    getOllamaStatus: vi.fn(),
    getModels: vi.fn(),
    routePrompt: vi.fn(),
    optimizePrompt: vi.fn(),
    compareOutputs: vi.fn(),
  },
}));

const mockComparisonResult: ComparisonResult = {
  downstreamModel: 'qwen2.5:1.5b',
  timestamp: '2026-11-06T00:00:00Z',
  warmup_attempted: true,
  warmup_succeeded: true,
  warmup_error: null,
  originalExecution: {
    prompt: 'Summarize the history of Python in 2 short bullet points.',
    response: '• Created by Guido van Rossum in 1991.\n• Emphasizes code readability and simplicity.',
    model: 'qwen2.5:1.5b',
    provider: 'ollama',
    prompt_tokens: 28,
    output_tokens: 22,
    total_tokens: 50,
    duration_ms: 180.5,
    prompt_eval_duration_ms: 45.2,
    status: 'completed',
    done_reason: 'stop',
    native_telemetry: {
      prompt_eval_count: 28,
      eval_count: 22,
    },
  },
  optimizedExecution: {
    prompt: 'Summarize Python history in 2 concise bullets:',
    response: '• Created in 1991 by Guido van Rossum.\n• Designed for clean readability.',
    model: 'qwen2.5:1.5b',
    provider: 'ollama',
    prompt_tokens: 14,
    output_tokens: 18,
    total_tokens: 32,
    duration_ms: 125.0,
    prompt_eval_duration_ms: 22.0,
    status: 'completed',
    done_reason: 'stop',
    native_telemetry: {
      prompt_eval_count: 14,
      eval_count: 18,
    },
  },
  metrics: {
    prompt_tokens_saved: 14,
    prompt_token_reduction_percent: 50.0,
    output_tokens_diff: -4,
    output_token_change_percent: -18.18,
    total_tokens_saved: 18,
    total_token_reduction_percent: 36.0,
    latency_delta_ms: -55.5,
    latency_reduction_percent: 30.75,
    semantic_quality: 'not_evaluated',
    semantic_quality_reason:
      'No automated judge model is configured. Semantic equivalence and response quality must be verified by human inspection.',
  },
  optimizerDetails: {
    originalPrompt: 'Summarize the history of Python in 2 short bullet points.',
    optimizedPrompt: 'Summarize Python history in 2 concise bullets:',
    mode: 'concise',
    targetModel: 'gemini-1-5-flash',
    taskCategory: 'summarization',
    strategy: 'compress',
    appliedStrategy: 'compress',
    outcome: 'validated_rewrite',
    lengthResult: 'compressed',
    validationStatus: 'passed',
    isPreserved: false,
    explicitConstraints: [],
    preservedConstraints: [],
    originalMetrics: { charCount: 57, estimatedTokens: 15 },
    optimizedMetrics: { charCount: 46, estimatedTokens: 12 },
    tokenReductionPercent: 20,
    tokenChangePercent: -20,
    tokenSavings: 3,
    charDelta: -11,
    executionTimeMs: 90,
    model: 'qwen2.5:1.5b',
    timestamp: '2026-11-06T00:00:00Z',
    isSimulated: false,
  },
};

const mockOllamaStatus: OllamaStatus = {
  connected: true,
  base_url: 'http://127.0.0.1:11434',
  model: 'qwen2.5:1.5b',
  model_installed: true,
  available_models: ['qwen2.5:1.5b', 'qwen2.5:7b'],
};

describe('OutputComparisonView Component', () => {
  it('renders pinned downstream model and execution summary', () => {
    render(<OutputComparisonView comparison={mockComparisonResult} />);

    expect(screen.getByText(/Pinned Model: qwen2.5:1.5b/i)).toBeInTheDocument();
    expect(screen.getByText(/Completed \(Warm-Up \+ Paired Run\)/i)).toBeInTheDocument();
    expect(screen.getAllByText(/50% saved/).length).toBeGreaterThanOrEqual(1); // Prompt token reduction
    expect(screen.getByText('-4')).toBeInTheDocument(); // Output token delta
    expect(screen.getByText('18')).toBeInTheDocument(); // Total tokens saved
  });

  it('renders side-by-side prompts and generated downstream responses', () => {
    render(<OutputComparisonView comparison={mockComparisonResult} />);

    // Prompts
    expect(screen.getByText('Summarize the history of Python in 2 short bullet points.')).toBeInTheDocument();
    expect(screen.getByText('Summarize Python history in 2 concise bullets:')).toBeInTheDocument();

    // Responses
    expect(screen.getByText(/Created by Guido van Rossum in 1991/)).toBeInTheDocument();
    expect(screen.getByText(/Created in 1991 by Guido van Rossum/)).toBeInTheDocument();
  });

  it('displays semantic quality honestly as not_evaluated', () => {
    render(<OutputComparisonView comparison={mockComparisonResult} />);

    expect(screen.getByText('not_evaluated')).toBeInTheDocument();
    expect(
      screen.getByText(/No automated judge model is configured/i)
    ).toBeInTheDocument();
  });

  it('handles missing telemetry by rendering N/A rather than zero', () => {
    const missingTelemetryResult: ComparisonResult = {
      ...mockComparisonResult,
      originalExecution: {
        ...mockComparisonResult.originalExecution,
        prompt_tokens: null,
        output_tokens: null,
        duration_ms: null,
      },
      optimizedExecution: {
        ...mockComparisonResult.optimizedExecution,
        prompt_tokens: null,
        output_tokens: null,
        duration_ms: null,
      },
      metrics: {
        ...mockComparisonResult.metrics,
        prompt_tokens_saved: null,
        prompt_token_reduction_percent: null,
        output_tokens_diff: null,
        total_tokens_saved: null,
        latency_delta_ms: null,
      },
    };

    render(<OutputComparisonView comparison={missingTelemetryResult} />);

    // N/A should be displayed in the metrics grid
    const naElements = screen.getAllByText('N/A');
    expect(naElements.length).toBeGreaterThanOrEqual(4);
  });

  it('displays cold-start warning when warm-up fails', () => {
    const warmupFailedResult: ComparisonResult = {
      ...mockComparisonResult,
      warmup_attempted: true,
      warmup_succeeded: false,
      warmup_error: 'Ollama warm-up connection timeout',
    };

    render(<OutputComparisonView comparison={warmupFailedResult} />);

    expect(screen.getByText(/Cold-Start Warning \(Warm-Up Failed\)/i)).toBeInTheDocument();
  });
});

describe('WorkspacePage Output Comparison Flow', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiService.getModels).mockResolvedValue({
      connected: true,
      base_url: 'http://127.0.0.1:11434',
      total_models: 1,
      models: [
        {
          name: 'qwen2.5:1.5b',
          size_bytes: 986061892,
          size_mb: 940,
          modified_at: '2026-10-21',
          parameter_size: '1.5B',
          tier: 'small',
          is_loaded: true,
        },
      ],
      active_loaded_models: ['qwen2.5:1.5b'],
    });
  });

  it('triggers compareOutputs and renders OutputComparisonView on success', async () => {
    vi.mocked(apiService.compareOutputs).mockResolvedValueOnce(mockComparisonResult);

    render(<WorkspacePage ollamaStatus={mockOllamaStatus} />);

    const textarea = screen.getByPlaceholderText(/Enter a prompt/i);
    fireEvent.change(textarea, { target: { value: 'Explain quantum computing in simple terms.' } });

    const compareBtn = screen.getByRole('button', { name: /Compare Downstream Outputs/i });
    expect(compareBtn).toBeEnabled();

    fireEvent.click(compareBtn);

    await waitFor(() => {
      expect(apiService.compareOutputs).toHaveBeenCalledWith({
        prompt: 'Explain quantum computing in simple terms.',
        downstream_model: 'qwen2.5:1.5b',
        mode: 'balanced',
        target_model: 'gemini-1-5-flash',
        enable_optimizer_routing: true,
        requested_optimizer_model: undefined,
        trials: 1,
      });
    });

    // OutputComparisonView should now be displayed
    await waitFor(() => {
      expect(screen.getByTestId('output-comparison-view')).toBeInTheDocument();
      expect(screen.getByText(/Pinned Model: qwen2.5:1.5b/i)).toBeInTheDocument();
    });
  });

  it('renders error notice when comparison fails', async () => {
    vi.mocked(apiService.compareOutputs).mockRejectedValueOnce(
      new Error('Downstream Model Error: Selected downstream model is unavailable')
    );

    render(<WorkspacePage ollamaStatus={mockOllamaStatus} />);

    const textarea = screen.getByPlaceholderText(/Enter a prompt/i);
    fireEvent.change(textarea, { target: { value: 'Test prompt' } });

    const compareBtn = screen.getByRole('button', { name: /Compare Downstream Outputs/i });
    fireEvent.click(compareBtn);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
      expect(
        screen.getByText(/Downstream Model Error: Selected downstream model is unavailable/i)
      ).toBeInTheDocument();
    });
  });
});

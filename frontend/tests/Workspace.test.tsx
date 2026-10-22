import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { WorkspacePage } from '../src/pages/WorkspacePage';
import { apiService } from '../src/services/api';
import { OptimizationResult } from '../src/types';

const MOCK_OPTIMIZE_RESULT: OptimizationResult = {
  originalPrompt:
    'You are an expert principal software engineer and cybersecurity auditor. Please review the following Python code snippet carefully and thoroughly. I would like you to make sure to check for any memory leaks, security vulnerabilities such as SQL injection, XSS, insecure deserialization, and race conditions.',
  optimizedPrompt:
    'Role: Principal Security & Software Engineer. Review Python code for memory leaks, SQLi, XSS, deserialization, and race conditions.',
  mode: 'balanced',
  targetModel: 'gpt-4o',
  originalMetrics: { charCount: 284, estimatedTokens: 71 },
  optimizedMetrics: { charCount: 130, estimatedTokens: 33 },
  tokenReductionPercent: 54,
  tokenChangePercent: -54,
  tokenSavings: 38,
  charDelta: -154,
  taskCategory: 'coding',
  strategy: 'compress',
  appliedStrategy: 'compress',
  outcome: 'validated_rewrite',
  lengthResult: 'compressed',
  validationStatus: 'passed',
  isPreserved: false,
  explicitConstraints: ['Prohibition: No SQL injection', 'Check memory leaks'],
  preservedConstraints: ['Core query intent & technical details', 'Role & persona parameters'],
  executionTimeMs: 1680,
  model: 'qwen2.5:1.5b',
  timestamp: new Date().toISOString(),
};

describe('Argus Workspace Dashboard (Phase 3)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders workspace headline and local model status pill', () => {
    render(<WorkspacePage ollamaStatus={{ connected: true, model: 'qwen2.5:1.5b', model_installed: true, base_url: 'http://127.0.0.1:11434', available_models: ['qwen2.5:1.5b'] }} />);
    expect(
      screen.getByText('Optimize Input Tokens & Preserve Constraints')
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Local Engine: qwen2.5:1.5b/i)
    ).toBeInTheDocument();
  });

  it('renders raw prompt editor with character and estimated token counters', () => {
    render(<WorkspacePage />);
    const textarea = screen.getByLabelText(/raw prompt text input/i);
    expect(textarea).toBeInTheDocument();
    expect(screen.getByText('~4 chars/token heuristic')).toBeInTheDocument();
  });

  it('populates textarea when an example prompt is selected', () => {
    render(<WorkspacePage />);
    const exampleButton = screen.getByText(/Code Review & Security Audit/i);
    fireEvent.click(exampleButton);

    const textarea = screen.getByLabelText(/raw prompt text input/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain('expert principal software engineer');
    expect(screen.getByText('Clear')).toBeInTheDocument();
  });

  it('enables optimize button when text is present and disables it when empty', () => {
    render(<WorkspacePage />);
    const optimizeBtn = screen.getByRole('button', { name: /optimize prompt/i });
    expect(optimizeBtn).toBeDisabled();

    const textarea = screen.getByLabelText(/raw prompt text input/i);
    fireEvent.change(textarea, { target: { value: 'Analyze this database query for indexes.' } });

    expect(optimizeBtn).not.toBeDisabled();

    // Clear prompt
    const clearBtn = screen.getByText('Clear');
    fireEvent.click(clearBtn);
    expect(optimizeBtn).toBeDisabled();
  });

  it('allows changing optimization mode and target model', () => {
    render(<WorkspacePage />);
    const conciseRadio = screen.getByLabelText(/Concise/i);
    fireEvent.click(conciseRadio);
    expect(conciseRadio).toBeChecked();

    const modelSelect = screen.getByLabelText(/Target Main LLM/i) as HTMLSelectElement;
    fireEvent.change(modelSelect, { target: { value: 'claude-3-5-sonnet' } });
    expect(modelSelect.value).toBe('claude-3-5-sonnet');
  });

  it('executes API optimization via Ollama and displays real results, metrics, and latency', async () => {
    const optimizeSpy = vi
      .spyOn(apiService, 'optimizePrompt')
      .mockResolvedValue(MOCK_OPTIMIZE_RESULT);

    render(
      <WorkspacePage
        ollamaStatus={{
          connected: true,
          model: 'qwen2.5:1.5b',
          model_installed: true,
          base_url: 'http://127.0.0.1:11434',
          available_models: ['qwen2.5:1.5b'],
        }}
      />
    );

    const exampleButton = screen.getByText(/Code Review & Security Audit/i);
    fireEvent.click(exampleButton);

    const optimizeBtn = screen.getByRole('button', { name: /optimize prompt/i });
    fireEvent.click(optimizeBtn);

    // Shows loading feedback
    expect(screen.getByText(/Optimizing Prompt with/i)).toBeInTheDocument();

    // Awaits real API results
    await waitFor(
      () => {
        expect(screen.getByText(/Optimization & Efficiency Metrics/i)).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    expect(optimizeSpy).toHaveBeenCalledWith({
      prompt: expect.stringContaining('expert principal software engineer'),
      mode: 'balanced',
      target_model: 'gpt-4o',
    });

    // Check latency and model
    expect(screen.getByText('1,680 ms')).toBeInTheDocument();
    expect(screen.getByText('qwen2.5:1.5b')).toBeInTheDocument();

    // Check metrics
    expect(screen.getByText(/Estimated Token Change/i)).toBeInTheDocument();
    expect(screen.getByText(/Optimization Strategy/i)).toBeInTheDocument();
    expect(screen.getByText(/Estimated Token Footprint Comparison/i)).toBeInTheDocument();
    expect(screen.getByText(/Validated Rewrite — Shorter/i)).toBeInTheDocument();

    // Check comparison view
    expect(screen.getByText(/Original Prompt \(Pre-Optimization\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Optimized Prompt \(Rewritten\)/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /copy optimized/i })).toBeInTheDocument();
  });

  it('displays user-friendly error notice when optimization API fails', async () => {
    vi.spyOn(apiService, 'optimizePrompt').mockRejectedValue(
      new Error("Ollama service is unreachable. Ensure Ollama is running ('ollama serve').")
    );

    render(<WorkspacePage />);
    const textarea = screen.getByLabelText(/raw prompt text input/i);
    fireEvent.change(textarea, { target: { value: 'Test prompt' } });

    const optimizeBtn = screen.getByRole('button', { name: /optimize prompt/i });
    fireEvent.click(optimizeBtn);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
      expect(screen.getByText(/Ollama service is unreachable/i)).toBeInTheDocument();
    });
  });

  it('allows editing optimized prompt in comparison view', async () => {
    vi.spyOn(apiService, 'optimizePrompt').mockResolvedValue(MOCK_OPTIMIZE_RESULT);

    render(<WorkspacePage />);
    const exampleButton = screen.getByText(/JSON Extraction & Entity Parser/i);
    fireEvent.click(exampleButton);

    const optimizeBtn = screen.getByRole('button', { name: /optimize prompt/i });
    fireEvent.click(optimizeBtn);

    await waitFor(
      () => {
        expect(screen.getByText(/Optimized Prompt \(Rewritten\)/i)).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    // Click edit
    const editBtn = screen.getByTitle(/toggle manual edits/i);
    fireEvent.click(editBtn);

    const editableTextarea = screen.getByLabelText(/editable optimized prompt/i) as HTMLTextAreaElement;
    expect(editableTextarea).toBeInTheDocument();
    fireEvent.change(editableTextarea, { target: { value: '{"custom_json": true}' } });
    expect(editableTextarea.value).toBe('{"custom_json": true}');
  });
});

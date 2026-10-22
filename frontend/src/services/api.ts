/**
 * Argus Frontend API Service Layer
 * Encapsulates network communication with the backend.
 */

import {
  HealthResponse,
  OllamaStatus,
  OptimizationResult,
  OptimizeRequestPayload,
  MainLLMStatus,
  ComparisonResult,
  BenchmarkReport,
} from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

export class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  /**
   * Health check endpoint
   */
  async getHealth(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Health check failed with status: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Ollama status and local model availability endpoint (Phase 3)
   */
  async getOllamaStatus(): Promise<OllamaStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/ollama/status`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        return {
          connected: false,
          base_url: 'http://127.0.0.1:11434',
          model: 'qwen2.5:1.5b',
          model_installed: false,
          available_models: [],
          error: `Ollama status check failed (HTTP ${response.status})`,
        };
      }

      return response.json();
    } catch (err) {
      return {
        connected: false,
        base_url: 'http://127.0.0.1:11434',
        model: 'qwen2.5:1.5b',
        model_installed: false,
        available_models: [],
        error: err instanceof Error ? err.message : 'Backend unreachable',
      };
    }
  }

  /**
   * Real Prompt Optimization endpoint powered by local Ollama LLM (Phase 3 & 3.1)
   */
  async optimizePrompt(payload: OptimizeRequestPayload): Promise<OptimizationResult> {
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(payload),
      });
    } catch (networkErr) {
      throw new Error(
        `Failed to reach Argus backend at ${this.baseUrl}. Ensure the FastAPI server is running.`
      );
    }

    if (!response.ok) {
      let errorMessage = `Optimization failed (HTTP ${response.status})`;
      try {
        const errorJson = await response.json();
        if (errorJson?.error?.message) {
          errorMessage = errorJson.error.message;
        } else if (errorJson?.detail) {
          errorMessage =
            typeof errorJson.detail === 'string'
              ? errorJson.detail
              : JSON.stringify(errorJson.detail);
        }
      } catch {
        // Fall back to HTTP status message
      }

      if (response.status === 503) {
        throw new Error(
          `Ollama service is unreachable. Ensure Ollama is running ('ollama serve') and '${payload.target_model || 'qwen2.5:1.5b'}' is available: ${errorMessage}`
        );
      }
      if (response.status === 504) {
        throw new Error(`Inference timed out: ${errorMessage}`);
      }
      if (response.status === 422) {
        throw new Error(`Validation Error: ${errorMessage}`);
      }

      throw new Error(errorMessage);
    }

    const data = await response.json();

    return {
      originalPrompt: data.original_prompt,
      optimizedPrompt: data.optimized_prompt,
      mode: data.mode,
      targetModel: data.target_model,
      taskCategory: data.task_category || 'general',
      strategy: data.strategy || 'compress',
      appliedStrategy: data.applied_strategy || data.strategy || 'compress',
      outcome: data.outcome || (data.validation_status === 'fallback' ? 'fallback' : data.is_preserved ? 'unchanged_original' : 'validated_rewrite'),
      lengthResult: data.length_result || (data.char_delta < 0 ? 'compressed' : data.char_delta > 0 ? 'expanded' : 'unchanged'),
      validationStatus: data.validation_status || 'passed',
      isPreserved: Boolean(data.is_preserved),
      primaryAction: data.primary_action,
      primaryDeliverable: data.primary_deliverable,
      explicitConstraints: data.explicit_constraints || [],
      requiredOperations: data.required_operations || [],
      originalMetrics: {
        charCount: data.original_metrics.char_count,
        estimatedTokens: data.original_metrics.estimated_tokens,
      },
      optimizedMetrics: {
        charCount: data.optimized_metrics.char_count,
        estimatedTokens: data.optimized_metrics.estimated_tokens,
      },
      tokenReductionPercent: data.token_reduction_percent ?? 0,
      tokenChangePercent: data.token_change_percent ?? 0,
      tokenSavings: data.token_savings ?? 0,
      charDelta: data.char_delta ?? 0,
      preservedConstraints: data.preserved_constraints || [],
      executionTimeMs: data.execution_time_ms,
      model: data.model,
      timestamp: data.timestamp,
      isSimulated: false,
    };
  }

  /**
   * Main LLM Provider status endpoint (Phase 4)
   */
  async getMainLLMStatus(): Promise<MainLLMStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/main-llm/status`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      if (!response.ok) {
        return {
          provider: 'openai',
          model: 'gpt-4o',
          is_configured: false,
          base_url: 'https://api.openai.com/v1',
          message: 'Unable to check provider status',
        };
      }
      return response.json();
    } catch {
      return {
        provider: 'openai',
        model: 'gpt-4o',
        is_configured: false,
        base_url: 'https://api.openai.com/v1',
        message: 'Backend service offline',
      };
    }
  }

  /**
   * Compare Baseline vs. Optimized paths on Main LLM (Phase 4)
   */
  async compareMainLLM(payload: {
    prompt: string;
    mode: string;
    target_model: string;
    temperature?: number;
    max_tokens?: number;
  }): Promise<ComparisonResult> {
    const response = await fetch(`${this.baseUrl}/main-llm/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let msg = `Comparison failed (HTTP ${response.status})`;
      try {
        const errJson = await response.json();
        if (errJson?.error?.message) msg = errJson.error.message;
        else if (errJson?.detail) msg = errJson.detail;
      } catch {}
      throw new Error(msg);
    }

    return response.json();
  }

  /**
   * Get version-controlled benchmark dataset (Phase 4)
   */
  async getBenchmarkDataset(): Promise<any[]> {
    const response = await fetch(`${this.baseUrl}/main-llm/benchmark/dataset`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`Failed to load benchmark dataset (HTTP ${response.status})`);
    }
    return response.json();
  }

  /**
   * Run reproducible benchmark suite (Phase 4)
   */
  async runBenchmark(live: boolean = false, model: string = 'gpt-4o'): Promise<BenchmarkReport> {
    const response = await fetch(`${this.baseUrl}/main-llm/benchmark/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ live, model }),
    });

    if (!response.ok) {
      let msg = `Benchmark execution failed (HTTP ${response.status})`;
      try {
        const errJson = await response.json();
        if (errJson?.error?.message) msg = errJson.error.message;
        else if (errJson?.detail) msg = errJson.detail;
      } catch {}
      throw new Error(msg);
    }

    return response.json();
  }
}

export const apiService = new ApiService();

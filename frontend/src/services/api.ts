/**
 * Argus Frontend API Service Layer
 * Encapsulates network communication with the backend.
 */

import {
  ComparisonRequestPayload,
  ComparisonResult,
  HealthResponse,
  ModelCatalogResponse,
  OllamaStatus,
  OptimizationResult,
  OptimizeRequestPayload,
  RouteRequestPayload,
  RoutingDecisionTelemetry,
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
  async getHealth(signal?: AbortSignal): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      signal,
    });

    if (!response.ok) {
      throw new Error(`Health check failed with status: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Ollama status and local model availability endpoint (Phase 3)
   */
  async getOllamaStatus(signal?: AbortSignal): Promise<OllamaStatus> {
    if (signal?.aborted) {
      throw new DOMException('The user aborted a request.', 'AbortError');
    }
    try {
      const response = await fetch(`${this.baseUrl}/ollama/status`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        signal,
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
      if ((err as { name?: string })?.name === 'AbortError' || signal?.aborted) {
        throw err;
      }
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
   * Discover installed local models and hardware state (Phase 5)
   */
  async getModels(signal?: AbortSignal): Promise<ModelCatalogResponse> {
    const response = await fetch(`${this.baseUrl}/models`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      signal,
    });

    if (!response.ok) {
      throw new Error(`Failed to retrieve models catalog (HTTP ${response.status})`);
    }

    return response.json();
  }

  /**
   * Evaluate model routing decision without running inference (Phase 5)
   */
  async routePrompt(
    payload: RouteRequestPayload,
    signal?: AbortSignal
  ): Promise<RoutingDecisionTelemetry> {
    const response = await fetch(`${this.baseUrl}/route`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err?.error?.message || err?.detail || `Routing evaluation failed (HTTP ${response.status})`);
    }

    return response.json();
  }

  /**
   * Real Prompt Optimization endpoint powered by local Ollama LLM (Phase 3, 4 & 5)
   */
  async optimizePrompt(payload: OptimizeRequestPayload, signal?: AbortSignal): Promise<OptimizationResult> {
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(payload),
        signal,
      });
    } catch (networkErr) {
      if ((networkErr as { name?: string })?.name === 'AbortError' || signal?.aborted) {
        throw networkErr;
      }
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
          `Ollama service is unreachable. Ensure Ollama is running ('ollama serve') and '${payload.requested_model || 'qwen2.5:1.5b'}' is available: ${errorMessage}`
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
      inferenceTelemetry: data.inference_telemetry
        ? {
            calls: data.inference_telemetry.calls || [],
            native_metrics_available: Boolean(data.inference_telemetry.native_metrics_available),
          }
        : null,
      outputBudget: data.output_budget || null,
      compression: data.compression || null,
      routing: data.routing || null,
      kvRuntimePolicy: data.kv_runtime_policy || null,
    };
  }

  /**
   * Run end-to-end downstream output comparison (Original vs Optimized)
   */
  async compareOutputs(
    payload: ComparisonRequestPayload,
    signal?: AbortSignal
  ): Promise<ComparisonResult> {
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/compare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(payload),
        signal,
      });
    } catch (networkErr) {
      if ((networkErr as { name?: string })?.name === 'AbortError' || signal?.aborted) {
        throw networkErr;
      }
      throw new Error(
        `Failed to reach Argus backend at ${this.baseUrl}. Ensure the FastAPI server is running.`
      );
    }

    if (!response.ok) {
      let errorMessage = `Comparison failed (HTTP ${response.status})`;
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

      if (response.status === 400) {
        throw new Error(`Downstream Model Error: ${errorMessage}`);
      }
      if (response.status === 503) {
        throw new Error(`Ollama service unreachable: ${errorMessage}`);
      }
      if (response.status === 504) {
        throw new Error(`Downstream inference timed out: ${errorMessage}`);
      }
      if (response.status === 422) {
        throw new Error(`Validation Error: ${errorMessage}`);
      }

      throw new Error(errorMessage);
    }

    const data = await response.json();
    const opt = data.optimizer_details;

    return {
      originalExecution: data.original_execution,
      optimizedExecution: data.optimized_execution,
      metrics: data.metrics,
      downstreamModel: data.downstream_model,
      timestamp: data.timestamp,
      optimizerDetails: {
        originalPrompt: opt.original_prompt,
        optimizedPrompt: opt.optimized_prompt,
        mode: opt.mode,
        targetModel: opt.target_model,
        taskCategory: opt.task_category || 'general',
        strategy: opt.strategy || 'compress',
        appliedStrategy: opt.applied_strategy || opt.strategy || 'compress',
        outcome:
          opt.outcome ||
          (opt.validation_status === 'fallback'
            ? 'fallback'
            : opt.is_preserved
            ? 'unchanged_original'
            : 'validated_rewrite'),
        lengthResult:
          opt.length_result ||
          (opt.char_delta < 0 ? 'compressed' : opt.char_delta > 0 ? 'expanded' : 'unchanged'),
        validationStatus: opt.validation_status || 'passed',
        isPreserved: Boolean(opt.is_preserved),
        primaryAction: opt.primary_action,
        primaryDeliverable: opt.primary_deliverable,
        explicitConstraints: opt.explicit_constraints || [],
        requiredOperations: opt.required_operations || [],
        originalMetrics: {
          charCount: opt.original_metrics.char_count,
          estimatedTokens: opt.original_metrics.estimated_tokens,
        },
        optimizedMetrics: {
          charCount: opt.optimized_metrics.char_count,
          estimatedTokens: opt.optimized_metrics.estimated_tokens,
        },
        tokenReductionPercent: opt.token_reduction_percent ?? 0,
        tokenChangePercent: opt.token_change_percent ?? 0,
        tokenSavings: opt.token_savings ?? 0,
        charDelta: opt.char_delta ?? 0,
        preservedConstraints: opt.preserved_constraints || [],
        executionTimeMs: opt.execution_time_ms,
        model: opt.model,
        timestamp: opt.timestamp,
        isSimulated: false,
        inferenceTelemetry: opt.inference_telemetry
          ? {
              calls: opt.inference_telemetry.calls || [],
              native_metrics_available: Boolean(opt.inference_telemetry.native_metrics_available),
            }
          : null,
        outputBudget: opt.output_budget || null,
        compression: opt.compression || null,
        routing: opt.routing || null,
        kvRuntimePolicy: opt.kv_runtime_policy || null,
      },
    };
  }
}

export const apiService = new ApiService();

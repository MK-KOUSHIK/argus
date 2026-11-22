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
    return transformOptimizeResponse(data);
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
    return {
      originalExecution: data.original_execution,
      optimizedExecution: data.optimized_execution,
      metrics: data.metrics,
      downstreamModel: data.downstream_model,
      effective_settings: data.effective_settings,
      trials: data.trials,
      warmup: data.warmup,
      warmup_attempted: data.warmup_attempted,
      warmup_succeeded: data.warmup_succeeded,
      warmup_error: data.warmup_error,
      trial_measurements: data.trial_measurements,
      aggregate_latency: data.aggregate_latency,
      timestamp: data.timestamp,
      optimizerDetails: transformOptimizeResponse(data.optimizer_details),
    };
  }
}

export function transformOptimizeResponse(data: any): OptimizationResult {
  if (!data) {
    throw new Error('transformOptimizeResponse received null or undefined data');
  }
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
    entities: data.entities || [],
    conditions: data.conditions || [],
    factsAndConstants: data.facts_and_constants || [],
    semanticPreservationScore: data.semantic_preservation_score,
    originalMetrics: {
      charCount: data.original_metrics?.char_count ?? (data.original_prompt?.length || 0),
      estimatedTokens: data.original_metrics?.estimated_tokens ?? (data.original_token_count || 0),
    },
    optimizedMetrics: {
      charCount: data.optimized_metrics?.char_count ?? (data.optimized_prompt?.length || 0),
      estimatedTokens: data.optimized_metrics?.estimated_tokens ?? (data.dispatched_token_count || 0),
    },
    tokenReductionPercent: data.token_reduction_percent ?? 0,
    tokenChangePercent: data.token_change_percent ?? 0,
    tokenSavings: data.token_savings ?? 0,
    charDelta: data.char_delta ?? 0,
    preservedConstraints: data.preserved_constraints || [],
    executionTimeMs: data.execution_time_ms ?? 0,
    model: data.model || 'qwen2.5:1.5b',
    timestamp: data.timestamp || new Date().toISOString(),
    isSimulated: false,
    inferenceTelemetry: data.inference_telemetry
      ? {
        calls: data.inference_telemetry.calls || [],
        native_metrics_available: Boolean(data.inference_telemetry.native_metrics_available),
      }
      : null,
    outputBudget: data.output_budget || null,
    compression: data.compression
      ? {
        ...data.compression,
        applied: data.compression.applied ?? data.compression.enabled ?? false,
        content_type:
          data.compression.content_type ?? data.compression.detected_content_type ?? 'general',
        strategy:
          data.compression.strategy ?? data.compression.strategy_applied ?? 'none',
        original_chars:
          data.compression.original_chars ?? data.compression.original_tokens ?? 0,
        compressed_chars:
          data.compression.compressed_chars ?? data.compression.compressed_tokens ?? 0,
        char_reduction_percent:
          data.compression.char_reduction_percent ?? data.compression.reduction_percent ?? 0,
        compression_time_ms:
          data.compression.compression_time_ms ?? data.compression.processing_time_ms,
        processing_time_ms:
          data.compression.processing_time_ms ?? data.compression.compression_time_ms,
        reversible:
          data.compression.reversible ?? Boolean(data.compression.restoration_reference),
        restoration_reference: data.compression.restoration_reference || null,
        segments: data.compression.segments || [],
        validation_passed:
          typeof data.compression.validation_passed === 'boolean'
            ? data.compression.validation_passed
            : data.compression.validation_status === 'passed',
        reason:
          data.compression.reason ||
          data.compression.fallback_reason ||
          (data.compression.enabled || data.compression.applied
            ? 'Content-aware compression applied successfully'
            : 'Original content preserved verbatim'),
      }
      : null,
    routing: data.routing || null,
    kvRuntimePolicy: data.kv_runtime_policy || null,
    candidatePrompt: data.candidate_prompt ?? null,
    originalTokenCount: data.original_token_count ?? data.original_metrics?.estimated_tokens,
    candidateTokenCount: data.candidate_token_count ?? null,
    dispatchedTokenCount: data.dispatched_token_count ?? data.optimized_metrics?.estimated_tokens,
    actualTokensSaved: data.actual_tokens_saved ?? data.token_savings ?? 0,
    actualReductionPercent: data.actual_reduction_percent ?? data.token_reduction_percent ?? 0,
    candidateAccepted: typeof data.candidate_accepted === 'boolean' ? data.candidate_accepted : !data.is_preserved,
    rejectionReason: data.rejection_reason ?? null,
    tokenMeasurementMethod: data.token_measurement_method ?? 'heuristic_chars',
  };
}

export const apiService = new ApiService();

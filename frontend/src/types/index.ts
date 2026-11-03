/**
 * Argus Frontend Types
 * Domain and API contracts
 */

export type ServiceStatus = 'healthy' | 'degraded' | 'unhealthy' | 'checking';

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error';
  app_name: string;
  version: string;
  environment: string;
  timestamp: string;
  message?: string;
}

export interface ApiError {
  detail: string;
  code?: string;
  timestamp?: string;
}

export interface SystemMetrics {
  totalOptimizations: number;
  averageTokenReductionPercent: number;
  estimatedCostSavingsUsd: number;
  activeModel: string;
}

/* =========================================================================
   Phase 2, 3 & 3.1: Prompt Optimization Workspace Contracts
   ========================================================================= */

export type OptimizationMode = 'balanced' | 'concise' | 'structure';

export type OptimizationStrategy = 'compress' | 'clarify' | 'structure' | 'enrich' | 'preserve';

export type TaskCategory =
  | 'coding'
  | 'data_analysis'
  | 'writing'
  | 'summarization'
  | 'research'
  | 'extraction'
  | 'planning'
  | 'general';

export type ValidationStatus = 'passed' | 'warning' | 'fallback';

export type OptimizationOutcome = 'validated_rewrite' | 'unchanged_original' | 'fallback';

export type LengthResult = 'compressed' | 'expanded' | 'unchanged';

export type TargetModel =
  | 'gemini-1-5-flash'
  | 'gemini-1-5-pro'
  | 'gemini-2.0-flash'
  | 'gpt-4o'
  | 'claude-3-5-sonnet'
  | 'deepseek-v3';

export interface ModelOption {
  id: TargetModel;
  name: string;
  provider: string;
  contextWindow: string;
  isIllustrative: true;
}

export type OptimizationStatus = 'idle' | 'optimizing' | 'success' | 'error';

export interface TokenEstimation {
  charCount: number;
  estimatedTokens: number;
}

export interface OptimizeRequestPayload {
  prompt: string;
  mode: OptimizationMode;
  target_model: TargetModel;
}

export interface OllamaCallTelemetry {
  queue_wait_ms: number;
  wall_time_ms: number;
  estimated_input_tokens_for_budget?: number;
  selected_num_ctx?: number;
  num_predict_limit?: number;
  load_duration_ns?: number | null;
  prompt_eval_count?: number | null;
  prompt_eval_duration_ns?: number | null;
  prompt_tokens_per_second?: number | null;
  eval_count?: number | null;
  eval_duration_ns?: number | null;
  generation_tokens_per_second?: number | null;
  native_metrics_available: boolean;
}

export interface OptimizationResult {
  originalPrompt: string;
  optimizedPrompt: string;
  mode: OptimizationMode;
  targetModel: TargetModel;
  
  // Phase 3.1 & 3.2 Task-Aware Metadata
  taskCategory: string;
  strategy: OptimizationStrategy;
  appliedStrategy?: string;
  outcome?: OptimizationOutcome;
  lengthResult?: LengthResult;
  validationStatus: ValidationStatus;
  isPreserved: boolean;
  primaryAction?: string;
  primaryDeliverable?: string;
  explicitConstraints: string[];
  requiredOperations?: string[];

  // Token & Length Metrics (Heuristic)
  originalMetrics: TokenEstimation;
  optimizedMetrics: TokenEstimation;
  tokenReductionPercent: number;
  tokenChangePercent: number;
  tokenSavings: number;
  charDelta: number;
  preservedConstraints: string[];
  executionTimeMs: number;
  model: string;
  timestamp: string;
  isSimulated?: boolean;
  inferenceTelemetry?: {
    calls: OllamaCallTelemetry[];
    native_metrics_available: boolean;
  } | null;
  kvRuntimePolicy?: {
    requested_num_ctx: number;
    requested_num_predict: number;
    argus_max_concurrent_requests: number;
    configured_kv_cache_type: string;
    configured_flash_attention: boolean;
    server_settings_verified: boolean;
    note: string;
  } | null;
}

export interface ExamplePrompt {
  id: string;
  title: string;
  category: string;
  description: string;
  content: string;
}

export interface OllamaStatus {
  connected: boolean;
  base_url: string;
  model: string;
  model_installed: boolean;
  available_models: string[];
  error?: string | null;
}


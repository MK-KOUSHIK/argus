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

export interface CompressionSegmentTelemetry {
  content_type: string;
  strategy: string;
  original_chars: number;
  compressed_chars: number;
  savings_percent: number;
  restoration_reference?: string | null;
}

export interface CompressionTelemetry {
  applied: boolean;
  content_type: string;
  strategy: string;
  original_chars: number;
  compressed_chars: number;
  char_reduction_percent: number;
  compression_time_ms: number;
  reversible: boolean;
  restoration_reference?: string | null;
  segments?: CompressionSegmentTelemetry[];
  validation_passed: boolean;
  reason: string;
}

export interface OutputBudgetTelemetry {
  requested_budget?: number | null;
  allocated_budget: number;
  policy: string;
  reason: string;
  task_category: string;
  task_confidence?: number | null;
  actual_generated_tokens?: number | null;
  generation_duration_ms?: number | null;
  generation_tokens_per_second?: number | null;
  done_reason?: string | null;
  hit_budget_limit?: boolean | null;
  budget_utilization_percent?: number | null;
}

/* =========================================================================
   Phase 5: Local-First Model Routing Contracts
   ========================================================================= */

export type ModelTier = 'small' | 'medium' | 'large' | 'unknown';
export type RoutingStatus = 'routed' | 'manual_override' | 'fallback' | 'disabled';

export interface ModelCatalogItem {
  name: string;
  parameter_size?: string | null;
  parameter_count?: number | null;
  quantization_level?: string | null;
  family?: string | null;
  context_length?: number | null;
  size_bytes: number;
  size_mb: number;
  tier: ModelTier;
  is_loaded: boolean;
  size_vram_mb?: number | null;
  modified_at: string;
}

export interface ModelCatalogResponse {
  connected: boolean;
  base_url: string;
  total_models: number;
  models: ModelCatalogItem[];
  active_loaded_models: string[];
  error?: string | null;
}

export interface ComplexityTelemetry {
  heuristic_score: number;
  task_category: string;
  task_confidence?: number | null;
  category_weight: number;
  explicit_constraints_count: number;
  required_operations_count: number;
  input_tokens: number;
  context_tier: number;
  output_budget: number;
  has_prohibitions: boolean;
  has_code_or_data: boolean;
  complexity_factors: string[];
}

export interface ResourceTelemetry {
  gpu_available: boolean;
  vram_total_mb?: number | null;
  vram_free_mb?: number | null;
  vram_used_mb?: number | null;
  ram_total_mb?: number | null;
  ram_free_mb?: number | null;
  ram_used_mb?: number | null;
  estimated_model_vram_mb?: number | null;
  is_resource_sufficient: boolean;
  resource_pressure_reason?: string | null;
  is_currently_loaded: boolean;
  concurrency_in_flight: number;
  concurrency_capacity: number;
}

export interface RoutingDecisionTelemetry {
  selected_model: string;
  target_model_tier: string;
  routing_status: string;
  policy_applied: string;
  reason: string;
  decision_confidence: number;
  heuristic_margin?: number | null;
  complexity: ComplexityTelemetry;
  resources: ResourceTelemetry;
  routing_duration_ms: number;
  fallback_model?: string | null;
  is_fallback: boolean;
  warning?: string | null;
}

export interface OptimizeRequestPayload {
  prompt: string;
  mode: OptimizationMode;
  target_model: TargetModel;
  requested_output_budget?: number;
  enable_compression?: boolean;
  compression_reversible?: boolean;
  requested_model?: string;
  enable_routing?: boolean;
  allow_override_fallback?: boolean;
}

export interface RouteRequestPayload {
  prompt: string;
  mode?: OptimizationMode;
  requested_model?: string;
  requested_output_budget?: number;
  allow_override_fallback?: boolean;
}

export interface OllamaCallTelemetry {
  model?: string;
  queue_wait_ms: number;
  wall_time_ms: number;
  estimated_input_tokens_for_budget?: number;
  selected_num_ctx?: number;
  num_predict_limit?: number;
  done_reason?: string | null;
  hit_budget_limit?: boolean | null;
  budget_utilization_percent?: number | null;
  load_duration_ns?: number | null;
  prompt_eval_count?: number | null;
  prompt_eval_duration_ns?: number | null;
  prompt_tokens_per_second?: number | null;
  eval_count?: number | null;
  eval_duration_ns?: number | null;
  generation_tokens_per_second?: number | null;
  native_metrics_available: boolean;
  prompt_eval_cached_count?: number | null;
  cache_hit_ratio?: number | null;
  telemetry_source?: string;
  hardware_snapshot?: {
    vram_before_mb?: number | null;
    vram_after_mb?: number | null;
    ram_before_mb?: number | null;
    ram_after_mb?: number | null;
  };
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
  outputBudget?: OutputBudgetTelemetry | null;

  compression?: CompressionTelemetry | null;
  routing?: RoutingDecisionTelemetry | null;
  kvRuntimePolicy?: {
    requested_num_ctx: number;
    requested_num_predict: number;
    argus_max_concurrent_requests: number;
    configured_kv_cache_type: string;
    configured_flash_attention: boolean;
    server_settings_verified: boolean;
    server_process_verified?: boolean;
    runtime_configuration_status?: string;
    confirmation_evidence?: string;
    active_pid?: number | null;
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
  runtime_process?: {
    requested_num_ctx: number;
    requested_num_predict: number;
    argus_max_concurrent_requests: number;
    configured_kv_cache_type: string;
    configured_flash_attention: boolean;
    server_endpoint: string;
    server_process_status: string;
    server_process_verified: boolean;
    server_settings_verified?: boolean;
    managed_by_argus: boolean;
    managed_pid: number | null;
    active_pid: number | null;
    runtime_configuration_status: string;
    confirmation_evidence: string;
    note: string;
  } | null;
  hardware_telemetry?: {
    gpu: {
      total_mb: number;
      free_mb: number;
      used_mb: number;
      source: string;
    };
    ram: {
      total_mb: number;
      free_mb: number;
      used_mb: number;
      source: string;
    };
  } | null;
}

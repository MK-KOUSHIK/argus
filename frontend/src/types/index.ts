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
  | 'gpt-4o'
  | 'claude-3-5-sonnet'
  | 'gemini-1-5-pro'
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

/* =========================================================================
   Phase 4: Main LLM Integration & Benchmark Contracts
   ========================================================================= */

export interface MainLLMStatus {
  provider: string;
  model: string;
  is_configured: boolean;
  base_url: string;
  message: string;
}

export interface CostBreakdown {
  input_cost_usd: number;
  output_cost_usd: number;
  total_cost_usd: number;
  is_estimated: boolean;
  currency: string;
}

export interface CostSavings {
  baseline_cost_usd: number;
  optimized_cost_usd: number;
  cost_saved_usd: number;
  savings_percent: number;
  is_estimated: boolean;
}

export interface ProviderUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface TokenCountResult {
  token_count: number;
  char_count: number;
  tokenizer_name: string;
  is_exact: boolean;
}

export interface PathExecutionResult {
  path_name: 'baseline' | 'optimized';
  prompt_used: string;
  tokenizer_tokens: TokenCountResult;
  provider_usage: ProviderUsage;
  cost: CostBreakdown;
  llm_response: string;
  latency_ms: number;
  total_latency_ms: number;
  is_mock: boolean;
}

export interface ComparisonResult {
  original_prompt: string;
  optimized_prompt: string;
  mode: OptimizationMode;
  model: string;
  optimizer_details: any;
  baseline: PathExecutionResult;
  optimized: PathExecutionResult;
  prompt_tokens_saved: number;
  prompt_tokens_saved_percent: number;
  total_tokens_saved: number;
  total_tokens_saved_percent: number;
  cost_comparison: CostSavings;
  latency_overhead_ms: number;
  constraints_preserved: boolean;
  is_mock: boolean;
}

export interface QualityEvaluation {
  prompt_constraints_preserved: boolean;
  output_prohibitions_respected: boolean;
  output_required_terms_present: boolean;
  output_format_valid: boolean;
  quality_score: number;
  is_passing: boolean;
  details: string[];
}

export interface BenchmarkCaseResult {
  case_id: string;
  case_name: string;
  category: string;
  description: string;
  comparison: ComparisonResult;
  baseline_quality: QualityEvaluation;
  optimized_quality: QualityEvaluation;
  is_passing: boolean;
  summary_notes: string;
}

export interface BenchmarkSummary {
  total_cases: number;
  passed_cases: number;
  pass_rate_percent: number;
  baseline_prompt_tokens: number;
  optimized_prompt_tokens: number;
  prompt_tokens_saved: number;
  prompt_tokens_reduction_percent: number;
  baseline_total_tokens: number;
  optimized_total_tokens: number;
  total_tokens_saved: number;
  total_tokens_reduction_percent: number;
  baseline_cost_usd: number;
  optimized_cost_usd: number;
  cost_saved_usd: number;
  cost_savings_percent: number;
  avg_baseline_latency_ms: number;
  avg_optimizer_latency_ms: number;
  avg_optimized_total_latency_ms: number;
  avg_latency_overhead_ms: number;
  constraint_retention_rate_percent: number;
  format_compliance_rate_percent: number;
  provider_name: string;
  model_name: string;
  is_live: boolean;
  timestamp: string;
}

export interface BenchmarkReport {
  metadata: {
    version: string;
    dataset_count: number;
    model: string;
    is_live: boolean;
  };
  summary: BenchmarkSummary;
  case_results: BenchmarkCaseResult[];
}

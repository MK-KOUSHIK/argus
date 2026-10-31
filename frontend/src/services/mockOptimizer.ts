/**
 * Argus Mock Optimization Service
 *
 * NOTE: This is a client-side mock for Phase 2 UI/UX preview and testing.
 * All optimizations and metrics are simulated and deterministic.
 * Real local LLM (Qwen2.5-1.5B via Ollama) and main-LLM dispatch will be integrated in Phase 3 & 4.
 */

import {
  OptimizationMode,
  TargetModel,
  OptimizationResult,
  TokenEstimation,
  ExamplePrompt,
  ModelOption,
} from '@/types';

/**
 * Token Estimation Heuristic
 * Documented approximation: Average English tokens are approximately ~4 characters per token.
 * Note: Actual token counts will vary depending on the target model's BPE/SentencePiece tokenizer.
 */
export function estimateTokens(text: string): TokenEstimation {
  const trimmed = text.trim();
  const charCount = text.length;
  if (!trimmed) {
    return { charCount: 0, estimatedTokens: 0 };
  }
  // Standard approximation: ~4 characters per token
  const estimatedTokens = Math.max(1, Math.ceil(trimmed.length / 4));
  return { charCount, estimatedTokens };
}

/**
 * Illustrative Target Model Options
 */
export const ILLUSTRATIVE_MODELS: ModelOption[] = [
  {
    id: 'gemini-1-5-flash',
    name: 'Gemini 1.5 Flash',
    provider: 'Google Gemini',
    contextWindow: '1M',
    isIllustrative: true,
  },
  {
    id: 'gpt-4o',
    name: 'GPT-4o',
    provider: 'OpenAI',
    contextWindow: '128k',
    isIllustrative: true,
  },
  {
    id: 'claude-3-5-sonnet',
    name: 'Claude 3.5 Sonnet',
    provider: 'Anthropic',
    contextWindow: '200k',
    isIllustrative: true,
  },
  {
    id: 'gemini-1-5-pro',
    name: 'Gemini 1.5 Pro',
    provider: 'Google Cloud',
    contextWindow: '2M',
    isIllustrative: true,
  },
  {
    id: 'deepseek-v3',
    name: 'DeepSeek V3',
    provider: 'DeepSeek',
    contextWindow: '64k',
    isIllustrative: true,
  },
];

/**
 * Curated Example Prompts for Developer Evaluation
 */
export const EXAMPLE_PROMPTS: ExamplePrompt[] = [
  {
    id: 'code-review',
    title: 'Code Review & Security Audit',
    category: 'Engineering',
    description: 'Verbose code review prompt with repeated formatting instructions',
    content: `You are an expert principal software engineer and cybersecurity auditor. Please review the following Python code snippet carefully and thoroughly. I would like you to make sure to check for any memory leaks, security vulnerabilities such as SQL injection, XSS, insecure deserialization, and race conditions. 

In addition to that, please make sure that your response is formatted in clean Markdown. First, give me a high-level summary of what the code does. Second, list all identified vulnerabilities ranked from critical to low severity. Third, provide the refactored code that fixes every single issue you found. Please do not include unnecessary conversational filler in your response, and make sure that you write production-grade code.`,
  },
  {
    id: 'customer-support-rag',
    title: 'Customer Support Policy RAG',
    category: 'Operations',
    description: 'Enterprise support system prompt with lengthy conversational rules',
    content: `You are a helpful, polite, and professional customer support representative for CloudScale Solutions. When answering user inquiries, you must always adhere strictly to the company policy document provided in the context below.

Please make sure to never disclose internal IP addresses, server names, or pricing tiers that are not explicitly documented. If the customer is asking about a refund, tell them that refunds are only available within 14 days of subscription renewal. If they are angry or frustrated, maintain an empathetic, calm, and reassuring tone. Do not guess any facts, and if you do not know the answer based on the context, politely state that you will escalate their ticket to Level 2 engineering.`,
  },
  {
    id: 'json-extraction',
    title: 'JSON Extraction & Entity Parser',
    category: 'Data Extraction',
    description: 'Data extraction spec with redundant explanations and formatting constraints',
    content: `Please act as a data parsing engine. I am going to give you unstructured text containing invoices, receipts, and order confirmations. Your job is to extract the vendor name, the total invoice amount including tax, the billing address, line items with quantities and unit prices, and the purchase order number.

Make sure that your output is 100% valid JSON and nothing else. Do not output any markdown code blocks, do not output any introductory or concluding explanations, do not say "Here is your JSON". The JSON schema must strictly contain keys: vendor, total_amount, currency, billing_address, line_items, and po_number. If any field is missing from the input, set its value to null.`,
  },
];

/**
 * Deterministic Mock Optimization Function
 * Simulates intent extraction and token compression across Balanced, Concise, and Structure modes.
 */
export function mockOptimizePrompt(
  rawPrompt: string,
  mode: OptimizationMode,
  targetModel: TargetModel
): OptimizationResult {
  const originalMetrics = estimateTokens(rawPrompt);

  let optimizedPrompt = '';
  let preservedConstraints: string[] = [];

  // 1. Check for preset example prompts to provide high-quality deterministic rewrites
  if (rawPrompt.includes('expert principal software engineer') && rawPrompt.includes('review the following Python code')) {
    if (mode === 'concise') {
      optimizedPrompt = `Role: Principal Security & Software Engineer.
Task: Audit Python snippet for vulnerabilities (SQLi, XSS, deserialization, race conditions) and memory leaks.
Output Requirements:
1. Executive summary
2. Vulnerability table (Critical -> Low)
3. Refactored production-ready code`;
      preservedConstraints = ['Role persona', 'Vulnerability categories', '3-step output structure', 'Production code quality'];
    } else if (mode === 'structure') {
      optimizedPrompt = `### System Role
Principal Software Engineer & Cybersecurity Auditor

### Objective
Audit Python code snippet for security vulnerabilities and memory leaks.

### Checklist
- SQL Injection, XSS, Insecure Deserialization, Race Conditions
- Resource leaks & performance bottlenecks

### Output Format (Markdown)
1. High-level Summary
2. Ranked Vulnerability Report (Critical to Low)
3. Hardened, Refactored Code`;
      preservedConstraints = ['Role specification', 'Specific threat vectors', 'Markdown schema order'];
    } else {
      // balanced
      optimizedPrompt = `Role: Principal Software Engineer & Cybersecurity Auditor.
Review the following Python snippet for memory leaks and vulnerabilities (SQLi, XSS, deserialization, race conditions).

Output structure:
1. High-level summary of code functionality.
2. Ranked vulnerability breakdown (Critical to Low).
3. Production-grade refactored code resolving all issues.`;
      preservedConstraints = ['Role persona', 'Full threat list', 'Output structure', 'No filler requirement'];
    }
  } else if (rawPrompt.includes('CloudScale Solutions') && rawPrompt.includes('customer support')) {
    if (mode === 'concise') {
      optimizedPrompt = `Role: CloudScale Solutions Support. Tone: Empathetic, calm.
Rules:
- Restrict answers strictly to provided context.
- Never disclose internal IPs, server hostnames, or unlisted tiers.
- Refund policy: strictly within 14 days of renewal.
- Unknown inquiries: Escalate to Level 2 engineering.`;
      preservedConstraints = ['Brand identity', 'Empathetic tone', '14-day refund window', 'Security exclusions', 'L2 escalation'];
    } else if (mode === 'structure') {
      optimizedPrompt = `### Persona
Customer Support Representative (CloudScale Solutions) | Tone: Empathetic & Professional

### Guardrails
- Context adherence: 100% strict (no speculation)
- Redaction: Internal IPs, hostnames, private pricing tiers
- Refunds: Valid only within 14-day renewal window
- Fallback: Escalate unresolvable queries to Level 2`;
      preservedConstraints = ['Company context', 'Confidentiality constraints', 'Policy numbers', 'Escalation workflow'];
    } else {
      // balanced
      optimizedPrompt = `You are a professional customer support representative for CloudScale Solutions. Adhere strictly to the provided policy context with an empathetic tone.
Constraints:
- Never disclose internal IPs, hostnames, or unpublished pricing.
- Refunds: Allowed only within 14 days of renewal.
- Fallback: If context is insufficient, state that the ticket is being escalated to Level 2 engineering.`;
      preservedConstraints = ['Empathetic persona', 'Refund threshold', 'Internal safety constraints', 'Escalation procedure'];
    }
  } else if (rawPrompt.includes('data parsing engine') && rawPrompt.includes('total invoice amount')) {
    if (mode === 'concise') {
      optimizedPrompt = `Task: Extract structured invoice data from input text.
Output: Strict raw JSON only (no markdown, no explanations).
Schema:
{"vendor":str|null,"total_amount":float|null,"currency":str|null,"billing_address":str|null,"line_items":[{"description":str,"quantity":num,"unit_price":num}],"po_number":str|null}`;
      preservedConstraints = ['Raw JSON format', 'No markdown ticks', 'Exact schema keys', 'Null fallback rule'];
    } else if (mode === 'structure') {
      optimizedPrompt = `### Directive
Parse invoice/receipt text into strict valid JSON. Output raw JSON only with zero surrounding commentary or markdown ticks.

### JSON Schema
{
  "vendor": string | null,
  "total_amount": number | null,
  "currency": string | null,
  "billing_address": string | null,
  "line_items": [{"description": string, "quantity": number, "unit_price": number}],
  "po_number": string | null
}`;
      preservedConstraints = ['Zero commentary restriction', 'Exact field names', 'Null handling'];
    } else {
      // balanced
      optimizedPrompt = `Act as an invoice parsing engine. Extract entity fields from unstructured text into valid JSON only.
Do not include markdown fences or preamble.
Required keys: vendor, total_amount, currency, billing_address, line_items (description, quantity, unit_price), po_number. Use null for missing values.`;
      preservedConstraints = ['Pure JSON constraint', 'Field mappings', 'Null fallback'];
    }
  } else {
    // 2. Generic prompt transformation heuristic for user-entered arbitrary text
    optimizedPrompt = transformGenericPrompt(rawPrompt, mode);
    preservedConstraints = [
      'Core query intent & entities',
      'Structural instructions',
      'Input parameter constraints',
    ];
  }

  const optimizedMetrics = estimateTokens(optimizedPrompt);
  const tokenSavings = Math.max(0, originalMetrics.estimatedTokens - optimizedMetrics.estimatedTokens);
  const tokenReductionPercent =
    originalMetrics.estimatedTokens > 0
      ? Math.round((tokenSavings / originalMetrics.estimatedTokens) * 100)
      : 0;

  return {
    originalPrompt: rawPrompt,
    optimizedPrompt,
    mode,
    targetModel,
    taskCategory: 'general',
    strategy: mode === 'structure' ? 'structure' : mode === 'concise' ? 'compress' : 'clarify',
    appliedStrategy: mode === 'structure' ? 'structure' : mode === 'concise' ? 'compress' : 'clarify',
    outcome: rawPrompt === optimizedPrompt ? 'unchanged_original' : 'validated_rewrite',
    lengthResult:
      optimizedMetrics.charCount < originalMetrics.charCount
        ? 'compressed'
        : optimizedMetrics.charCount > originalMetrics.charCount
        ? 'expanded'
        : 'unchanged',
    validationStatus: 'passed',
    isPreserved: rawPrompt === optimizedPrompt,
    explicitConstraints: [],
    requiredOperations: [],
    originalMetrics,
    optimizedMetrics,
    tokenReductionPercent,
    tokenChangePercent: -tokenReductionPercent,
    tokenSavings: Math.max(0, tokenSavings),
    charDelta: optimizedMetrics.charCount - originalMetrics.charCount,
    preservedConstraints,
    executionTimeMs: 140, // Simulated local model inference latency
    model: 'qwen2.5:1.5b (mock)',
    timestamp: new Date().toISOString(),
    isSimulated: true,
  };
}

/**
 * Heuristic compression for arbitrary user text
 */
function transformGenericPrompt(text: string, mode: OptimizationMode): string {
  // Remove common polite filler words
  let cleaned = text
    .replace(/\b(please|kindly|could you please|can you please|i would like you to|i want you to)\b/gi, '')
    .replace(/\b(in order to|as a matter of fact|at this point in time)\b/gi, '')
    .replace(/\b(make sure that you|ensure that you|be sure to)\b/gi, 'Ensure:')
    .replace(/\s+/g, ' ')
    .trim();

  if (mode === 'concise') {
    // Break into concise statements
    const sentences = cleaned
      .split(/[.!?]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    if (sentences.length <= 1) {
      return `Task: ${cleaned}`;
    }

    return `Task: ${sentences[0]}\nDirectives:\n${sentences
      .slice(1)
      .map((s) => `- ${s}`)
      .join('\n')}`;
  }

  if (mode === 'structure') {
    const sentences = cleaned
      .split(/[.!?]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    return `### Objective\n${sentences[0] || cleaned}\n\n### Requirements\n${
      sentences.length > 1
        ? sentences
            .slice(1)
            .map((s) => `- ${s}`)
            .join('\n')
        : '- Execute instructions with zero filler\n- Retain strict constraint fidelity'
    }\n\n### Output Format\nStructured markdown response.`;
  }

  // Balanced
  return `Instruction: ${cleaned}\nConstraints: Output clean, concise, high-fidelity response adhering strictly to all requirements.`;
}

/**
 * Simulated latency helper to test realistic UI loading states
 */
export async function simulateOptimizationDelay(ms: number = 600): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

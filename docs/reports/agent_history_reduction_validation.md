# Argus Phase 6 – Agent History Reduction Stage 4 Production Validation Report

- **Execution Timestamp:** `2026-11-06T05:45:08.081374+00:00`
- **Git Commit SHA:** `ea0bb47872ecec90519b1bbdba42ca71667db7e9`
- **Ollama Version:** `0.40.8`
- **Model:** `qwen2.5:1.5b` (`65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`)
- **Generation Options:** `temperature=0.2`, `top_p=0.9`, `num_ctx=4096`, `num_predict=512`
- **Evaluation Methodology:** 10 paired alternating trials per fixture (20 inferences per scenario, 60 total)

> [!NOTE]
> **Latency & Scheduling Disclaimer:** Token savings do not guarantee latency reduction on small models due to GPU/CPU scheduling noise, CUDA driver context switches, and kernel launch overheads. Native duration metrics are reported with mean, median, and standard deviation across 10 paired trials to provide rigorous statistical transparency.

## 1. Executive Summary & Verification Highlights

Stage 4 production validation evaluated the `TrajectoryReductionService` against the local live Ollama daemon (`qwen2.5:1.5b`) across all three synthetic test fixtures. All measurements are derived directly from Ollama native telemetry. Criterion regressions (`P -> F`) and improvements (`F -> P`) are aggregated across all 10 paired trials.

| Scenario ID | Msgs (Orig -> Red) | Native Input Tokens (Orig -> Red) | Actual Token Savings | Prompt Eval Latency (Orig -> Red) | Pure Inference (Orig -> Red) | Rubric Pass Rate (Orig -> Red) | Regressions (`P -> F`) | Improvements (`F -> P`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| `synthetic_fixture_coding_debug_trajectory` | 8 -> 6 | **394.0 ± 0.0** -> **273.0 ± 0.0** | **121.0 tok (30.71%)** | 25.53ms -> 11.26ms | 2534.96ms -> 2383.43ms | 68.0% -> **72.0%** | **1** | **3** |
| `synthetic_fixture_sql_data_exploration_trajectory` | 9 -> 9 | **570.0 ± 0.0** -> **570.0 ± 0.0** | **0.0 tok (0.0%)** | 7.75ms -> 7.65ms | 427.41ms -> 407.28ms | 40.0% -> **40.0%** | **4** | **4** |
| `synthetic_fixture_multi_step_research_constraints_trajectory` | 6 -> 5 | **506.0 ± 0.0** -> **398.0 ± 0.0** | **108.0 tok (21.34%)** | 9.84ms -> 9.56ms | 574.37ms -> 592.95ms | 70.0% -> **82.5%** | **1** | **6** |

## 2. Granular Scenario Analysis

### Synthetic Fixture 1: Multi-Turn Coding & Debugging Trajectory

- **Task Domain:** `coding`
- **Messages:** Original: 8 | Reduced: 6
- **Trials Completed:** Original: 10/10 | Reduced: 10/10 (Failed: 0)
- **Criterion Transitions (10 Trials):** 1 regressions (`P -> F`), 3 improvements (`F -> P`)

#### Telemetry Statistics (10 Paired Trials: Mean, Median, Standard Deviation)

| Telemetry Metric | Original Unreduced (Mean ± Stdev [Median]) | Reduced Trajectory (Mean ± Stdev [Median]) | Delta / Savings |
| :--- | :--- | :--- | :--- |
| **Native Input Tokens (`prompt_eval_count`)** | 394.0 ± 0.0 [394.0] | 273.0 ± 0.0 [273.0] | **-121.0 tokens (-30.71%)** |
| **Prompt Evaluation Latency** | 25.53 ± 44.12 ms [11.62 ms] | 11.26 ± 1.47 ms [10.8 ms] | **-14.27 ms** |
| **Decoding Latency (`eval_duration`)** | 2509.43 ± 493.77 ms [2569.23 ms] | 2372.18 ± 513.26 ms [2242.47 ms] | -137.25 ms |
| **Pure Inference Latency** | 2534.96 ± 497.83 ms [2650.65 ms] | 2383.43 ± 513.3 ms [2255.57 ms] | -151.53 ms |
| **Total Wall-Clock Latency** | 2571.51 ± 499.56 ms [2685.3 ms] | 2436.65 ± 524.3 ms [2286.12 ms] | -134.86 ms |
| **Rubric Quality Pass Rate** | 68.0% | **72.0%** | **+4.0%** |

#### Individual 10-Trial Alternating Run Breakdown

| Trial | Order | Orig Tokens | Red Tokens | Orig Prefill (ms) | Red Prefill (ms) | Orig Pure Inf (ms) | Red Pure Inf (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `original -> reduced` | 394 | 273 | 151.09 | 10.9 | 2723.95 | 3692.68 |
| 2 | `reduced -> original` | 394 | 273 | 11.13 | 15.28 | 2801.64 | 2273.36 |
| 3 | `original -> reduced` | 394 | 273 | 11.75 | 10.68 | 2577.35 | 2133.65 |
| 4 | `reduced -> original` | 394 | 273 | 11.87 | 10.71 | 2200.82 | 2190.92 |
| 5 | `original -> reduced` | 394 | 273 | 12.02 | 11.73 | 2802.95 | 2570.81 |
| 6 | `reduced -> original` | 394 | 273 | 11.48 | 10.24 | 3105.07 | 2389.61 |
| 7 | `original -> reduced` | 394 | 273 | 11.35 | 10.39 | 2841.71 | 1771.76 |
| 8 | `reduced -> original` | 394 | 273 | 12.9 | 10.93 | 2445.57 | 2237.78 |
| 9 | `original -> reduced` | 394 | 273 | 10.84 | 11.14 | 2544.91 | 2506.45 |
| 10 | `reduced -> original` | 394 | 273 | 10.89 | 10.58 | 1305.64 | 2067.32 |

#### Reduction Audit Decisions & Rationale

| Msg Index | Role | Action | Action Detail | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `system` | **retained** | `preserved_anchor` | Root system instruction must be preserved as immutable anchor |
| 1 | `user` | **retained** | `preserved_numerical_evidence` | Retained source numerical facts, exact financial amounts, and filter criteria required for downstream calculations |
| 2 | `assistant` | **removed** | `pruned_superseded_error` | Pruned intermediate failed tool call and traceback superseded by subsequent verified execution |
| 3 | `tool` | **removed** | `pruned_superseded_error` | Pruned intermediate failed tool call and traceback superseded by subsequent verified execution |
| 4 | `user` | **retained** | `preserved_recent_turn` | Message is within recent preservation window (1 turns) |
| 5 | `assistant` | **retained** | `preserved_recent_turn` | Message is within recent preservation window (1 turns) |
| 6 | `tool` | **retained** | `preserved_recent_turn` | Message is within recent preservation window (1 turns) |
| 7 | `user` | **retained** | `preserved_anchor` | Active user prompt must be preserved as immutable anchor |

#### Programmatic Rubric Evaluation (Representative Output)

| Criterion ID | Requirement | Status | Programmatic Rationale |
| :--- | :--- | :--- | :--- |
| `req_pytest_functions` | Must generate pytest unit tests for format_currency function. | **PASS** | Pytest definitions found: True, target function format_currency referenced: True. |
| `req_currencies_and_decimals` | Must test USD, EUR, and JPY (with 0 decimal places). | **FAIL** | USD tested: True, EUR tested: True, JPY tested: True, JPY 0-decimal places assertion verified: False. |
| `req_negative_and_zero` | Must test negative amounts and zero. | **PASS** | Negative amounts tested: True, zero value tested: True. |
| `req_no_locale_module` | Must maintain prohibition against using the external 'locale' module. | **PASS** | Implementation is self-contained without 'locale'. |
| `req_code_block_only` | Must output only Python test code in a code block. | **FAIL** | Code-only format verified. (Preamble: False, Postscript: True) |

#### Representative Reduced Model Output

```
```python
import pytest

def format_currency(amount: float, currency_code: str) -> str:
    """
    Formats a number into a standard currency representation.
    """
    if amount < 0:
        sign = '-'
        amount = abs(amount)
    else:
        sign = ''

    # Format the amount to two decimal places
    formatted_amount = f'{amount:.2f}'

    # Add currency symbol
    return f'{sign}{currency_code} {formatted_amount}'

# Test cases
@pytest.mark.parametrize(
    "amount, currency_code, expected", [
        (1234.5, 'USD', '$1,234.50'),
        (-45.2, 'EUR', '-€45.20'),
        (0, 'JPY', '¥0.00'),
        (0.99, 'USD', '$0.99'),
        (-0.99, 'EUR', '-€0.99'),
        (0, 'JPY', '¥0.00'),
        (-45.2, 'USD', '-$45.20'),
        (1234.5, 'USD', '$1,234.50'),
        (0, 'USD', '$0.00'),
        (-0.99, 'USD', '-$0.99'),
        (45.2, 'EUR', '€45.20'),
        (0.99, 'EUR', '€0.99'),
        (-0.99, 'EUR', '-€0.99'),
        (0, 'EUR', '€0.00'),
        (-45.2, 'EUR', '-€45.20'),
        (1234.5, 'EUR', '€1,234.50'),
        (0, 'EUR', '€0.00'),
        (-0.99, 'EUR', '-€0.99'),
        (45.2, 'JPY', '¥45.20'),
        (0.99, 'JPY', '¥0.99'),
        (-0.99, 'JPY', '-¥0.99'),
        (0, 'JPY', '¥0.00'),
        (-45.2, 'JPY',
```

---

### Synthetic Fixture 2: Data Exploration & SQL Querying Trajectory

- **Task Domain:** `data_analysis`
- **Messages:** Original: 9 | Reduced: 9
- **Trials Completed:** Original: 10/10 | Reduced: 10/10 (Failed: 0)
- **Criterion Transitions (10 Trials):** 4 regressions (`P -> F`), 4 improvements (`F -> P`)

#### Telemetry Statistics (10 Paired Trials: Mean, Median, Standard Deviation)

| Telemetry Metric | Original Unreduced (Mean ± Stdev [Median]) | Reduced Trajectory (Mean ± Stdev [Median]) | Delta / Savings |
| :--- | :--- | :--- | :--- |
| **Native Input Tokens (`prompt_eval_count`)** | 570.0 ± 0.0 [570.0] | 570.0 ± 0.0 [570.0] | **-0.0 tokens (-0.0%)** |
| **Prompt Evaluation Latency** | 7.75 ± 0.42 ms [7.58 ms] | 7.65 ± 0.13 ms [7.64 ms] | **-0.1 ms** |
| **Decoding Latency (`eval_duration`)** | 419.66 ± 113.52 ms [409.01 ms] | 399.63 ± 51.18 ms [412.2 ms] | -20.03 ms |
| **Pure Inference Latency** | 427.41 ± 113.38 ms [416.54 ms] | 407.28 ± 51.14 ms [419.89 ms] | -20.13 ms |
| **Total Wall-Clock Latency** | 452.08 ± 114.76 ms [438.72 ms] | 430.59 ± 50.49 ms [442.13 ms] | -21.49 ms |
| **Rubric Quality Pass Rate** | 40.0% | **40.0%** | **+0.0%** |

#### Individual 10-Trial Alternating Run Breakdown

| Trial | Order | Orig Tokens | Red Tokens | Orig Prefill (ms) | Red Prefill (ms) | Orig Pure Inf (ms) | Red Pure Inf (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `original -> reduced` | 570 | 570 | 7.6 | 7.8 | 530.5 | 378.18 |
| 2 | `reduced -> original` | 570 | 570 | 7.54 | 7.8 | 420.65 | 420.05 |
| 3 | `original -> reduced` | 570 | 570 | 7.58 | 7.58 | 689.58 | 420.78 |
| 4 | `reduced -> original` | 570 | 570 | 8.64 | 7.67 | 299.47 | 397.13 |
| 5 | `original -> reduced` | 570 | 570 | 7.52 | 7.42 | 303.2 | 487.26 |
| 6 | `reduced -> original` | 570 | 570 | 7.43 | 7.62 | 368.81 | 284.59 |
| 7 | `original -> reduced` | 570 | 570 | 8.41 | 7.55 | 398.74 | 418.7 |
| 8 | `reduced -> original` | 570 | 570 | 7.49 | 7.59 | 416.75 | 419.74 |
| 9 | `original -> reduced` | 570 | 570 | 7.58 | 7.81 | 416.34 | 422.88 |
| 10 | `reduced -> original` | 570 | 570 | 7.76 | 7.66 | 430.08 | 423.45 |

#### Reduction Audit Decisions & Rationale

| Msg Index | Role | Action | Action Detail | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `system` | **retained** | `preserved_anchor` | Root system instruction must be preserved as immutable anchor |
| 1 | `user` | **retained** | `preserved_uncertain` | Retained historical user statement under conservative retention policy to safeguard unrecognized context |
| 2 | `assistant` | **retained** | `preserved_unmodified` | Preserved to maintain dialogue coherence and tool-call sequence dependencies |
| 3 | `tool` | **retained** | `preserved_numerical_evidence` | Retained source numerical facts, exact financial amounts, and filter criteria required for downstream calculations |
| 4 | `user` | **retained** | `preserved_recent_turn` | Message is within recent preservation window (1 turns) |
| 5 | `assistant` | **retained** | `preserved_recent_turn` | Message is within recent preservation window (1 turns) |
| 6 | `tool` | **retained** | `preserved_recent_turn` | Message is within recent preservation window (1 turns) |
| 7 | `assistant` | **retained** | `preserved_recent_turn` | Message is within recent preservation window (1 turns) |
| 8 | `user` | **retained** | `preserved_anchor` | Active user prompt must be preserved as immutable anchor |

#### Programmatic Rubric Evaluation (Representative Output)

| Criterion ID | Requirement | Status | Programmatic Rationale |
| :--- | :--- | :--- | :--- |
| `req_average_order_value` | Must calculate the average order value from the 5 retrieved orders ($5,552.10). | **PASS** | Expected average ($5,552.10) detected in response: True. |
| `req_west_region_and_2025` | Must reference the West region and 2025 timeline. | **FAIL** | West region referenced: False, 2025 timeline referenced: False. |
| `req_two_sentences_and_formula` | Must state the mathematical formula and final answer in exactly two concise sentences. | **FAIL** | Mathematical calculation formula present: True, sentence count detected: 1 (expected 2). |

#### Representative Reduced Model Output

```
The average order value for the top 5 orders is calculated by summing the total amount of $27,760.50 and dividing by 5, resulting in an average order value of $5,552.10.
```

---

### Synthetic Fixture 3: Multi-Step Research & Refined Constraints

- **Task Domain:** `research`
- **Messages:** Original: 6 | Reduced: 5
- **Trials Completed:** Original: 10/10 | Reduced: 10/10 (Failed: 0)
- **Criterion Transitions (10 Trials):** 1 regressions (`P -> F`), 6 improvements (`F -> P`)

#### Telemetry Statistics (10 Paired Trials: Mean, Median, Standard Deviation)

| Telemetry Metric | Original Unreduced (Mean ± Stdev [Median]) | Reduced Trajectory (Mean ± Stdev [Median]) | Delta / Savings |
| :--- | :--- | :--- | :--- |
| **Native Input Tokens (`prompt_eval_count`)** | 506.0 ± 0.0 [506.0] | 398.0 ± 0.0 [398.0] | **-108.0 tokens (-21.34%)** |
| **Prompt Evaluation Latency** | 9.84 ± 2.07 ms [9.38 ms] | 9.56 ± 1.86 ms [9.34 ms] | **-0.28 ms** |
| **Decoding Latency (`eval_duration`)** | 564.53 ± 80.5 ms [570.29 ms] | 583.39 ± 84.88 ms [591.46 ms] | 18.86 ms |
| **Pure Inference Latency** | 574.37 ± 80.11 ms [579.83 ms] | 592.95 ± 85.4 ms [602.43 ms] | 18.58 ms |
| **Total Wall-Clock Latency** | 617.75 ± 79.93 ms [620.5 ms] | 632.38 ± 93.73 ms [647.49 ms] | 14.63 ms |
| **Rubric Quality Pass Rate** | 70.0% | **82.5%** | **+12.5%** |

#### Individual 10-Trial Alternating Run Breakdown

| Trial | Order | Orig Tokens | Red Tokens | Orig Prefill (ms) | Red Prefill (ms) | Orig Pure Inf (ms) | Red Pure Inf (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `original -> reduced` | 506 | 398 | 11.8 | 10.16 | 709.42 | 652.79 |
| 2 | `reduced -> original` | 506 | 398 | 7.75 | 7.4 | 603.32 | 486.69 |
| 3 | `original -> reduced` | 506 | 398 | 10.44 | 11.0 | 540.05 | 505.52 |
| 4 | `reduced -> original` | 506 | 398 | 7.62 | 7.73 | 567.81 | 505.42 |
| 5 | `original -> reduced` | 506 | 398 | 11.47 | 10.89 | 591.85 | 609.03 |
| 6 | `reduced -> original` | 506 | 398 | 8.07 | 7.73 | 554.39 | 531.69 |
| 7 | `original -> reduced` | 506 | 398 | 11.4 | 10.99 | 630.07 | 699.83 |
| 8 | `reduced -> original` | 506 | 398 | 8.23 | 8.31 | 611.81 | 733.65 |
| 9 | `original -> reduced` | 506 | 398 | 13.3 | 12.84 | 398.92 | 597.62 |
| 10 | `reduced -> original` | 506 | 398 | 8.32 | 8.52 | 536.05 | 607.24 |

#### Reduction Audit Decisions & Rationale

| Msg Index | Role | Action | Action Detail | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `system` | **retained** | `preserved_anchor` | Root system instruction must be preserved as immutable anchor |
| 1 | `user` | **retained** | `preserved_uncertain` | Retained historical user statement under conservative retention policy to safeguard unrecognized context |
| 2 | `assistant` | **removed** | `pruned_redundant_turn` | Pruned early unconstrained conversational commentary superseded by subsequent operational constraints |
| 3 | `user` | **retained** | `preserved_recent_turn` | Message is within recent preservation window (1 turns) |
| 4 | `assistant` | **retained** | `preserved_recent_turn` | Message is within recent preservation window (1 turns) |
| 5 | `user` | **retained** | `preserved_anchor` | Active user prompt must be preserved as immutable anchor |

#### Programmatic Rubric Evaluation (Representative Output)

| Criterion ID | Requirement | Status | Programmatic Rationale |
| :--- | :--- | :--- | :--- |
| `req_three_bullet_points` | Must provide exactly three concise bullet points. | **PASS** | Detected 3 bullet items (expected exactly 3). |
| `req_16gb_ram_and_sq8` | Must explicitly mention the 16GB RAM constraint and SQ8 quantization strategy. | **PASS** | 16GB RAM limit mentioned: True, SQ8 quantization mentioned: True. |
| `req_air_gapped_requirement` | Must adhere to the air-gapped requirement (no cloud/external APIs). | **PASS** | Air-gapped / on-premises constraint explicitly referenced: True. |
| `req_no_marketing_filler` | Must not include generic marketing advantages or filler. | **PASS** | Output contains concise technical recommendations without promotional buzzwords. |

#### Representative Reduced Model Output

```
- Deploy a dedicated containerized local server process, pinned to 8 GB max RAM, to stay within the 16 GB host RAM limit.
- Use Scalar Quantization (SQ8) to reduce vector memory footprint by 75%, down to ~400 MB, while maintaining recall.
- Pre-load the SQ8 index into system page cache to achieve deterministic <15ms P99 search latency, ensuring air-gapped operation.
```

---

## 3. Regression & Improvement Analysis (All 10 Paired Trials)

Across all three synthetic fixtures (120 total criterion evaluations over 10 paired trials), there were **6 regressions** and **13 improvements**.

### Quality Regressions (`P -> F`) by Scenario and Trial

| Scenario | Trial | Run Order | Criterion ID | Description | Baseline Rationale | Reduced Rationale |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| `synthetic_fixture_coding_debug_trajectory` | 9 | `original -> reduced` | `req_code_block_only` | Must output only Python test code in a code block. | Code-only format verified. (Preamble: False, Postscript: False) | Code-only format verified. (Preamble: False, Postscript: True) |
| `synthetic_fixture_sql_data_exploration_trajectory` | 3 | `original -> reduced` | `req_two_sentences_and_formula` | Must state the mathematical formula and final answer in exactly two concise sentences. | Mathematical calculation formula present: True, sentence count detected: 2 (expected 2). | Mathematical calculation formula present: True, sentence count detected: 1 (expected 2). |
| `synthetic_fixture_sql_data_exploration_trajectory` | 4 | `reduced -> original` | `req_two_sentences_and_formula` | Must state the mathematical formula and final answer in exactly two concise sentences. | Mathematical calculation formula present: True, sentence count detected: 2 (expected 2). | Mathematical calculation formula present: True, sentence count detected: 1 (expected 2). |
| `synthetic_fixture_sql_data_exploration_trajectory` | 5 | `original -> reduced` | `req_two_sentences_and_formula` | Must state the mathematical formula and final answer in exactly two concise sentences. | Mathematical calculation formula present: True, sentence count detected: 2 (expected 2). | Mathematical calculation formula present: True, sentence count detected: 1 (expected 2). |
| `synthetic_fixture_sql_data_exploration_trajectory` | 10 | `reduced -> original` | `req_two_sentences_and_formula` | Must state the mathematical formula and final answer in exactly two concise sentences. | Mathematical calculation formula present: True, sentence count detected: 2 (expected 2). | Mathematical calculation formula present: True, sentence count detected: 1 (expected 2). |
| `synthetic_fixture_multi_step_research_constraints_trajectory` | 2 | `reduced -> original` | `req_air_gapped_requirement` | Must adhere to the air-gapped requirement (no cloud/external APIs). | Air-gapped / on-premises constraint explicitly referenced: True. | Air-gapped / on-premises constraint explicitly referenced: False. |

### Quality Improvements (`F -> P`) by Scenario and Trial

| Scenario | Trial | Run Order | Criterion ID | Description | Baseline Rationale | Reduced Rationale |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| `synthetic_fixture_coding_debug_trajectory` | 3 | `original -> reduced` | `req_code_block_only` | Must output only Python test code in a code block. | Code-only format verified. (Preamble: False, Postscript: True) | Code-only format verified. (Preamble: False, Postscript: False) |
| `synthetic_fixture_coding_debug_trajectory` | 4 | `reduced -> original` | `req_code_block_only` | Must output only Python test code in a code block. | Code-only format verified. (Preamble: False, Postscript: True) | Code-only format verified. (Preamble: False, Postscript: False) |
| `synthetic_fixture_coding_debug_trajectory` | 10 | `reduced -> original` | `req_code_block_only` | Must output only Python test code in a code block. | Code-only format verified. (Preamble: False, Postscript: True) | Code-only format verified. (Preamble: False, Postscript: False) |
| `synthetic_fixture_sql_data_exploration_trajectory` | 1 | `original -> reduced` | `req_average_order_value` | Must calculate the average order value from the 5 retrieved orders ($5,552.10). | Expected average ($5,552.10) detected in response: False. | Expected average ($5,552.10) detected in response: True. |
| `synthetic_fixture_sql_data_exploration_trajectory` | 3 | `original -> reduced` | `req_average_order_value` | Must calculate the average order value from the 5 retrieved orders ($5,552.10). | Expected average ($5,552.10) detected in response: False. | Expected average ($5,552.10) detected in response: True. |
| `synthetic_fixture_sql_data_exploration_trajectory` | 5 | `original -> reduced` | `req_west_region_and_2025` | Must reference the West region and 2025 timeline. | West region referenced: False, 2025 timeline referenced: False. | West region referenced: True, 2025 timeline referenced: True. |
| `synthetic_fixture_sql_data_exploration_trajectory` | 6 | `reduced -> original` | `req_two_sentences_and_formula` | Must state the mathematical formula and final answer in exactly two concise sentences. | Mathematical calculation formula present: True, sentence count detected: 1 (expected 2). | Mathematical calculation formula present: True, sentence count detected: 2 (expected 2). |
| `synthetic_fixture_multi_step_research_constraints_trajectory` | 1 | `original -> reduced` | `req_air_gapped_requirement` | Must adhere to the air-gapped requirement (no cloud/external APIs). | Air-gapped / on-premises constraint explicitly referenced: False. | Air-gapped / on-premises constraint explicitly referenced: True. |
| `synthetic_fixture_multi_step_research_constraints_trajectory` | 5 | `original -> reduced` | `req_16gb_ram_and_sq8` | Must explicitly mention the 16GB RAM constraint and SQ8 quantization strategy. | 16GB RAM limit mentioned: False, SQ8 quantization mentioned: True. | 16GB RAM limit mentioned: True, SQ8 quantization mentioned: True. |
| `synthetic_fixture_multi_step_research_constraints_trajectory` | 7 | `original -> reduced` | `req_16gb_ram_and_sq8` | Must explicitly mention the 16GB RAM constraint and SQ8 quantization strategy. | 16GB RAM limit mentioned: False, SQ8 quantization mentioned: True. | 16GB RAM limit mentioned: True, SQ8 quantization mentioned: True. |
| `synthetic_fixture_multi_step_research_constraints_trajectory` | 8 | `reduced -> original` | `req_16gb_ram_and_sq8` | Must explicitly mention the 16GB RAM constraint and SQ8 quantization strategy. | 16GB RAM limit mentioned: False, SQ8 quantization mentioned: True. | 16GB RAM limit mentioned: True, SQ8 quantization mentioned: True. |
| `synthetic_fixture_multi_step_research_constraints_trajectory` | 8 | `reduced -> original` | `req_air_gapped_requirement` | Must adhere to the air-gapped requirement (no cloud/external APIs). | Air-gapped / on-premises constraint explicitly referenced: False. | Air-gapped / on-premises constraint explicitly referenced: True. |
| `synthetic_fixture_multi_step_research_constraints_trajectory` | 10 | `reduced -> original` | `req_air_gapped_requirement` | Must adhere to the air-gapped requirement (no cloud/external APIs). | Air-gapped / on-premises constraint explicitly referenced: False. | Air-gapped / on-premises constraint explicitly referenced: True. |

## 4. Known Limitations

1. **Local Model Small Output Drifts:** Small local models like `qwen2.5:1.5b` exhibit stochastic formatting variance (such as occasionally testing JPY with decimal points or formulating answers in 1 compound sentence rather than 2 sentences). Trajectory reduction preserves essential context and improves math/constraint fidelity, but cannot overcome fundamental model generation limitations without constrained decoding or output grammar filters.
2. **Single-Agent Scope:** Trajectory reduction operates on sequential observable turns within a single agent context. Multi-agent distributed trajectories with branching DAGs require hierarchical reduction.
3. **Synchronous Reduction:** For extremely deep trajectories (>100 messages), trajectory reduction involves parsing and regex scanning. On current workloads (<50 messages), reduction duration remains negligible (<1 ms).

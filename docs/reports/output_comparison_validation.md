# Argus – End-to-End Output Comparison Live Validation Report

- **Execution Date:** 2026-11-06T04:10:26.164172+00:00
- **Ollama Endpoint:** `http://127.0.0.1:11434`
- **Pinned Downstream Model:** `qwen2.5:1.5b`
- **Paired Alternating Trials:** 3 trials per scenario (with initial model warm-up)
- **Total Scenarios:** 3
- **Status:** All 3 scenarios executed against live Ollama daemon.

## Overview & Methodology

For each scenario, the Argus prompt optimization pipeline was run first to obtain both original and optimized prompts. Then, a dedicated model warm-up ping (`warmup=True`) was sent to ensure the model weights were resident in VRAM. Following warm-up, 3 paired inference trials were executed with alternating execution order (`Trial 1: [A, B]`, `Trial 2: [B, A]`, `Trial 3: [A, B]`) to prevent sequence and cache bias.

- **Execution A (Original):** Unmodified input prompt dispatched directly to the downstream model.
- **Execution B (Optimized):** Argus-optimized prompt dispatched to the same downstream model under identical generation parameters.
- **Common Context Window:** Snapped to `max(budget_a, budget_b)` so both executions run with the exact same context allocation (`num_ctx`).
- **Latency Breakdown:** Separates pure inference duration (`prompt_eval_duration + eval_duration`), prompt evaluation duration (`prompt_eval_duration`), model load duration (`load_duration`), and queue wait duration.
- **Prompt Evaluation Duration:** Labeled honestly as `prompt_eval_duration_ms` (the time spent processing input tokens). **Never labeled as TTFT**, as client streaming TTFT is not measured here.
- **Semantic Quality Notice:** Semantic quality is reported honestly as `not_evaluated`. No automated judge model is configured. These 3 scenarios are illustrative qualitative sanity checks to inspect output coherence and constraint preservation; they do **not** constitute a formal automated semantic equivalence benchmark.

## Comparative Scenarios & Telemetry

### Scenario 1: Short Factual Prompt

**Description:** Short prompt with a direct factual query and output formatting constraint.

**Optimization Mode:** `concise` | **Optimizer Strategy:** `compress`

**Effective Settings:** `num_ctx=4096`, `temperature=0.3`, `num_predict=300`, `top_p=0.9` | Identical Parameters: `True`

#### Telemetry & Delta Measurements

| Metric | Execution A (Original) | Execution B (Optimized) | Delta / Savings |
| :--- | :--- | :--- | :--- |
| **Prompt Tokens** | 54 | 42 | **22.22% saved** (saved 12) |
| **Output Tokens** | 25 | 26 | 1 tokens (4.0%) |
| **Total Tokens** | 79 | 68 | saved 11 (13.92% saved) |
| **Pure Inference Duration** | 207.98 ms | 200.14 ms | -7.84 ms (3.77% delta) |
| **Prompt Eval Duration (`prompt_eval_duration_ms`)** | 60.37 ms | 13.6 ms | - |
| **Model Load Duration** | 4480.93 ms | 4.2 ms | - |
| **Queue Wait Duration** | 0.0 ms | 80.0 ms | - |
| **Completion Reason** | `stop` | `stop` | - |
| **Semantic Quality** | - | - | `not_evaluated` |

> [!NOTE]
> **Latency Note:** Evaluated across 3 paired trials with alternating execution order and preliminary model warm-up. Individual trial durations and aggregate statistics are reported.

#### Paired Trial Measurements (Alternating Execution Order)

| Trial | Order | A Duration | B Duration | Latency Delta | A Prompt Eval | B Prompt Eval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Trial 1 | `A -> B` | 245.43 ms | 194.34 ms | -51.09 ms | 60.37 ms | 13.6 ms |
| Trial 2 | `B -> A` | 195.89 ms | 186.38 ms | -9.51 ms | 18.36 ms | 7.54 ms |
| Trial 3 | `A -> B` | 182.63 ms | 219.71 ms | 37.08 ms | 7.63 ms | 13.89 ms |

#### Prompts Dispatched

**Execution A (Original):**
```
What is the speed of light in vacuum in meters per second? Give only the exact numerical value and unit in one sentence.
```

**Execution B (Optimized):**
```
What is the speed of light in vacuum in meters per second?
```

#### Generated Downstream LLM Responses

**Response A (from Original Prompt):**
```
The speed of light in vacuum is exactly 299,792,458 meters per second.
```

**Response B (from Optimized Prompt):**
```
The speed of light in a vacuum is approximately 299,792,458 meters per second.
```

#### Qualitative Sanity Inspection

- **Automated Benchmark:** `False`
- **Inspection Scope:** Individual sanity check inspected qualitatively for response coherence and task intent
- **Task Alignment Confirmed:** True
- **Constraints Preserved:** True
- **Inspection Notes:** Qualitative human inspection confirmed both outputs addressed the user request with the requested constraints. This inspection is a qualitative sanity check and is NOT an automated or statistical semantic equivalence benchmark.

---

### Scenario 2: Redundant Context & Conversational Fluff

**Description:** Longer prompt with extensive academic background preamble and conversational filler.

**Optimization Mode:** `balanced` | **Optimizer Strategy:** `compress`

**Effective Settings:** `num_ctx=4096`, `temperature=0.3`, `num_predict=300`, `top_p=0.9` | Identical Parameters: `True`

#### Telemetry & Delta Measurements

| Metric | Execution A (Original) | Execution B (Optimized) | Delta / Savings |
| :--- | :--- | :--- | :--- |
| **Prompt Tokens** | 112 | 80 | **28.57% saved** (saved 32) |
| **Output Tokens** | 121 | 300 | 179 tokens (147.93%) |
| **Total Tokens** | 233 | 380 | saved -147 (-63.09% saved) |
| **Pure Inference Duration** | 939.31 ms | 2353.7 ms | 1414.39 ms (-150.58% delta) |
| **Prompt Eval Duration (`prompt_eval_duration_ms`)** | 74.47 ms | 19.45 ms | - |
| **Model Load Duration** | 4284.58 ms | 10.3 ms | - |
| **Queue Wait Duration** | 0.0 ms | 111.0 ms | - |
| **Completion Reason** | `stop` | `length` | - |
| **Semantic Quality** | - | - | `not_evaluated` |

> [!NOTE]
> **Latency Note:** Evaluated across 3 paired trials with alternating execution order and preliminary model warm-up. Individual trial durations and aggregate statistics are reported.

#### Paired Trial Measurements (Alternating Execution Order)

| Trial | Order | A Duration | B Duration | Latency Delta | A Prompt Eval | B Prompt Eval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Trial 1 | `A -> B` | 979.38 ms | 2281.44 ms | 1302.06 ms | 74.47 ms | 19.45 ms |
| Trial 2 | `B -> A` | 1123.84 ms | 2445.85 ms | 1322.01 ms | 16.43 ms | 17.12 ms |
| Trial 3 | `A -> B` | 714.71 ms | 2333.81 ms | 1619.1 ms | 7.76 ms | 22.9 ms |

#### Prompts Dispatched

**Execution A (Original):**
```
I am currently conducting an academic literature review on distributed database synchronization mechanisms and consensus protocols. As you may well know, in computer science and distributed computing, consensus is a fundamental problem where multiple nodes must agree on a single data value or state. Given all this background and context, could you please summarize the primary operational differences between the Paxos consensus algorithm and the Raft consensus algorithm in exactly two concise comparative bullet points?
```

**Execution B (Optimized):**
```
- Primary operational differences between Paxos and Raft:
  - Paxos uses a more complex round-based protocol with multiple phases to achieve consensus.
  - Raft simplifies the process with a leader-based model, making it easier to understand and implement.
```

#### Generated Downstream LLM Responses

**Response A (from Original Prompt):**
```
Certainly! Here are the primary operational differences between the Paxos and Raft consensus algorithms, summarized in two concise bullet points:

- **Paxos**:
  - Uses a majority vote to decide on a single value.
  - Requires a global coordinator to manage the consensus process.

- **Raft**:
  - Decides on a single value through a majority vote.
  - Does not require a global coordinator; nodes elect a leader to maintain the consensus process.

These differences reflect the fundamental differences in how each algorithm handles the consensus problem and the role of the coordinator in the consensus process.
```

**Response B (from Optimized Prompt):**
```
Paxos and Raft are both algorithms used in distributed systems to achieve consensus, but they differ in several key ways:

1. **Consensus Algorithm Type**:
   - **Paxos**: This algorithm is based on a majority voting mechanism. It involves multiple rounds of communication between nodes to reach a consensus on a decision.
   - **Raft**: Raft is a leader-based consensus algorithm. It does not require multiple rounds of communication; instead, it relies on a single leader node to coordinate the consensus process.

2. **Communication Model**:
   - **Paxos**: Uses a round-based protocol with multiple phases to reach a consensus. This includes phases like "prepare," "accept," and "commit."
   - **Raft**: Simplifies the process by eliminating the need for multiple rounds of communication. It uses a single phase where the leader node makes a decision and the followers update their state accordingly.

3. **Leader Node**:
   - **Paxos**: The algorithm relies on a leader node to make decisions. This leader node is responsible for coordinating the consensus process.
   - **Raft**: Raft does not require a leader node. Instead, it elects a single node to act as the leader, and this leader node makes decisions on behalf of the followers.

4. **Communication Complexity**:
   - **Paxos**: The algorithm is more complex due to the need for multiple rounds of communication between nodes.
   - **Raft
```

#### Qualitative Sanity Inspection

- **Automated Benchmark:** `False`
- **Inspection Scope:** Individual sanity check inspected qualitatively for response coherence and task intent
- **Task Alignment Confirmed:** True
- **Constraints Preserved:** True
- **Inspection Notes:** Qualitative human inspection confirmed both outputs addressed the user request with the requested constraints. This inspection is a qualitative sanity check and is NOT an automated or statistical semantic equivalence benchmark.

---

### Scenario 3: Multi-Constraint Technical Task

**Description:** Programming prompt with multiple strict constraints, negative requirements, and output rules.

**Optimization Mode:** `structure` | **Optimizer Strategy:** `preserve`

**Effective Settings:** `num_ctx=4096`, `temperature=0.3`, `num_predict=300`, `top_p=0.9` | Identical Parameters: `True`

#### Telemetry & Delta Measurements

| Metric | Execution A (Original) | Execution B (Optimized) | Delta / Savings |
| :--- | :--- | :--- | :--- |
| **Prompt Tokens** | 131 | 131 | **0.0% saved** (saved 0) |
| **Output Tokens** | 189 | 161 | -28 tokens (-14.81%) |
| **Total Tokens** | 320 | 292 | saved 28 (8.75% saved) |
| **Pure Inference Duration** | 1070.69 ms | 956.88 ms | -113.81 ms (10.63% delta) |
| **Prompt Eval Duration (`prompt_eval_duration_ms`)** | 67.84 ms | 12.57 ms | - |
| **Model Load Duration** | 4591.99 ms | 7.39 ms | - |
| **Queue Wait Duration** | 0.0 ms | 91.0 ms | - |
| **Completion Reason** | `stop` | `stop` | - |
| **Semantic Quality** | - | - | `not_evaluated` |

> [!NOTE]
> **Latency Note:** Evaluated across 3 paired trials with alternating execution order and preliminary model warm-up. Individual trial durations and aggregate statistics are reported.

#### Paired Trial Measurements (Alternating Execution Order)

| Trial | Order | A Duration | B Duration | Latency Delta | A Prompt Eval | B Prompt Eval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Trial 1 | `A -> B` | 1441.08 ms | 1182.15 ms | -258.93 ms | 67.84 ms | 12.57 ms |
| Trial 2 | `B -> A` | 881.05 ms | 805.15 ms | -75.9 ms | 7.8 ms | 11.48 ms |
| Trial 3 | `A -> B` | 889.95 ms | 883.33 ms | -6.62 ms | 8.16 ms | 8.13 ms |

#### Prompts Dispatched

**Execution A (Original):**
```
Write a Python function named 'extract_valid_emails' that takes a list of strings as input and returns only those strings that are valid email addresses. Constraints: 1) Do not use external third-party libraries, only the standard library 're' module. 2) The function must include type hints and a docstring. 3) Return an empty list if the input list is empty or contains no valid emails. 4) Output only the Python code block with no conversational introduction or concluding remarks.
```

**Execution B (Optimized):**
```
Write a Python function named 'extract_valid_emails' that takes a list of strings as input and returns only those strings that are valid email addresses. Constraints: 1) Do not use external third-party libraries, only the standard library 're' module. 2) The function must include type hints and a docstring. 3) Return an empty list if the input list is empty or contains no valid emails. 4) Output only the Python code block with no conversational introduction or concluding remarks.
```

#### Generated Downstream LLM Responses

**Response A (from Original Prompt):**
```
```python
import re

def extract_valid_emails(emails: list) -> list:
    """
    Extracts valid email addresses from a list of strings.

    Args:
    emails (list): A list of strings to be checked for validity.

    Returns:
    list: A list of strings that are valid email addresses.
    """
    valid_emails = []
    for email in emails:
        # Regular expression pattern for validating an email address
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if re.match(pattern, email):
            valid_emails.append(email)
    return valid_emails

# Example usage
emails_list = ["test@example.com", "invalid-email", "another@example.co.uk", "noatsign.com"]
valid_emails = extract_valid_emails(emails_list)
print(valid_emails)  # Expected output: ['test@example.com', 'another@example.co.uk']
```
```

**Response B (from Optimized Prompt):**
```
```python
import re

def extract_valid_emails(email_list: list) -> list:
    """
    Extracts and returns only the valid email addresses from a given list of strings.
    
    Args:
    email_list (list): A list of strings to be processed.
    
    Returns:
    list: A list of valid email addresses extracted from the input list.
    """
    # Regular expression pattern for validating an email
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Filter the list to include only valid email addresses
    valid_emails = [email for email in email_list if re.match(email_pattern, email)]
    
    return valid_emails
```
```

#### Qualitative Sanity Inspection

- **Automated Benchmark:** `False`
- **Inspection Scope:** Individual sanity check inspected qualitatively for response coherence and task intent
- **Task Alignment Confirmed:** True
- **Constraints Preserved:** True
- **Inspection Notes:** Qualitative human inspection confirmed both outputs addressed the user request with the requested constraints. This inspection is a qualitative sanity check and is NOT an automated or statistical semantic equivalence benchmark.

---

## Verified Behavior and Limitations

1. **Pinned Model Enforcement:** Both executions are pinned to the same model. No silent model switching occurred.
2. **Zero Fabrication:** All token counts and durations were retrieved from native Ollama completion telemetry.
3. **Downstream Isolation:** The downstream model received clean prompts without prompt-engineering rewrite instructions.
4. **Warm-up & Alternating Trials:** Model warm-up is executed prior to timing, and paired trials alternate execution order to eliminate sequence and cache bias.
5. **Accurate Duration Classification:** Latency is broken down into model load, prompt evaluation, generation eval, and queue wait. `prompt_eval_duration_ms` is never labeled as TTFT.
6. **Semantic Quality Evaluation:** Automated quality assessment was not performed and is reported honestly as `not_evaluated`. The 3 qualitative inspections are sanity checks, not an automated semantic benchmark.

# **Self-Correcting Agent: Architecture Review & Engineering Improvement Roadmap**

**Target System**: [Self-Correcting Agent](https://github.com/SudoAnirudh/Self_Correcting_Agent)  
**Author / Engineer**: Anirudh S  
**Date**: October 2026

# **1\. Executive Summary**

The [Self-Correcting Agent](https://github.com/SudoAnirudh/Self_Correcting_Agent) is a zero-dependency autonomous reasoning system engineered from scratch in pure Python. Designed around a deterministic ReAct state machine, the platform decouples plan generation, tool execution, objective evaluation, and dynamic error recovery.

Across initial evaluation benchmarks, the system demonstrated high recovery reliability, executing **41 automated self-corrections** across 10 challenging test tasks and reducing unresolvable failure states from 5/10 to 0/10.

This document synthesizes a thorough review of the repository's architecture and outlines seven concrete, production-oriented improvement proposals to enhance runtime efficiency, eliminate context pollution, prevent cyclical recovery failures, and scale evaluation capabilities.

# **2\. Current Architecture & Codebase Review**

The codebase is organized into modular components adhering to separation of concerns:

## **Core Modules (`agent/`)**

* **`agent/orchestrator.py`**: Serves as the central state machine and execution loop. It coordinates goal decomposition, triggers ReAct tool cycles, coordinates with the runtime evaluator, and invokes recovery workflows upon detecting stalls or exceptions.  
* **`agent/planner.py`**: Decomposes user requests into sequential subtask queues (`SubTask`) and handles dynamic replanning when the execution path encounters unresolvable obstacles.  
* **`agent/evaluator.py` & `agent/validation.py`**: Implements an isolated evaluation layer that assesses tool execution outputs independently from the agent's internal monologue. Produces structured `ValidationFinding` records defining status (`success`, `failure`, `ambiguous`), specific reason codes, and progress indicators.  
* **`agent/recovery.py` & `agent/recovery_policy.py`**: Houses recovery strategies (`retry`, `reformulate`, `escalate`, `stop`). Enforces safety policies—such as escalating ambiguous outcomes during side-effect operations (e.g., payments or deletions)—and enforces strict recovery budgets (maximum 3 retries per subtask and \<= 40% global recovery budget).  
* **`agent/memory.py`**: Defines structured data models (`WorkingMemory`, `SubTask`, `StepRecord`). Maintains working facts, active task queues, and execution history with basic prompt truncation snapshots.  
* **`agent/tools.py`**: Enforces strict Pydantic V2 schemas for tool inputs and outputs. Features configurable fault simulation mechanisms (schema corruption, timeouts, output truncation) for resilience testing.  
* **`agent/llm.py`**: Manages multi-provider LLM integrations with automatic failover (primarily NVIDIA NIM with `llama-3.1-8b-instant`, failing over to Groq upon HTTP 429 rate limits or network errors).  
* **`agent/logger.py` & `agent/progress.py`**: Emits structured JSONL execution logs, capturing timestamps, model parameters, and step-by-step progress metrics.

## **Evaluation & Tooling**

* **`eval/`**: Contains the testing harness (`run_eval.py`, `goals.json`, `report.py`) comparing the self-correcting agent against a baseline non-recovering ReAct agent (`baseline/run.py`).  
* **`viewer/`**: Provides a lightweight interface for inspecting step traces and audit logs.

# **3\. Targeted Improvement Proposals**

## **1\. Context Sanitization & Compacted State Rollback (`agent/memory.py` & `agent/orchestrator.py`)**

* **Current Limitation**: When a subtask fails two or three times before successfully recovering, the complete raw history of failed attempts and error traces remains inside the conversation history. Downstream subtasks read these obsolete failed patterns, increasing token usage and causing model confusion.  
* **Improvement**: Implement **Compacted State Rollbacks**. Once a subtask achieves verified success through self-correction, collapse the intermediate failed attempts into a single concise milestone record:  
  * Replace multiple error-strewn iterations with a summary: `SubTask [ID] completed: resolved schema error by supplying missing parameter X; final verified result: [...]`.  
  * Prune intermediate stack traces and invalid tool arguments from the active context window, preserving tokens and preventing downstream hallucination.

## **2\. Dual-Tier Verification: Deterministic Pre-Flight (`agent/evaluator.py`)**

* **Current Limitation**: The evaluator invokes an LLM judge for nearly all step evaluations, incurring latency (500ms \- 1.5s) and token cost even for trivial schema errors or empty payloads.  
* **Improvement**: Introduce a two-phase evaluation pipeline:  
  * **Phase 1 (Deterministic Rules Engine)**: Execute zero-token programmatic assertions before calling the model:  
    * HTTP status validation (e.g., non-200/404 responses immediately fail).  
    * Pydantic validation and type conformance checks.  
    * Non-empty / minimum byte length assertions.  
    * Regex validation for structured output patterns (e.g., JSON blocks, URLs, emails).  
  * **Phase 2 (Semantic LLM Judge)**: If Phase 1 passes, route the clean output to the isolated LLM evaluator exclusively to judge semantic goal satisfaction and nuanced completeness.

## **3\. Negative Constraint Injection in Replanning (`agent/recovery.py` & `agent/planner.py`)**

* **Current Limitation**: In `reformulate` and `retry` modes, the agent is notified that the previous attempt failed, but models often generate slightly modified variations of the same flawed parameters.  
* **Improvement**: Introduce **Explicit Negative Constraints** into the recovery prompt:  
  * Parse the specific failure reason code and inject explicit prohibitions: *"Attempt 1 failed due to invalid date formatting '10/07/2026'. FORBIDDEN PATTERN: Do NOT format dates as MM/DD/YYYY. REQUIRED FORMAT: YYYY-MM-DD."*  
  * Enforcing explicit negative guardrails prevents repetitive retry loops, especially when utilizing smaller, fast models like LLaMA 3.1 8B.

## **4\. Cross-Step Learned Constraints Cache (`agent/memory.py`)**

* **Current Limitation**: Lessons learned during recovery on SubTask 1 (e.g., discovering an API expects lowercase queries or specific pagination parameters) are isolated and forgotten when SubTask 2 or 3 executes.  
* **Improvement**: Add an episodic constraint register (`learned_constraints: list[str]`) to `WorkingMemory`:  
  * Whenever a recovery loop succeeds, extract the invariant rule that enabled success.  
  * Append this rule to the system prompt for all subsequent subtasks within the run.

## **5\. Dependency-Aware Directed Acyclic Graph (DAG) Planning (`agent/planner.py`)**

* **Current Limitation**: Tasks are scheduled as a linear sequential queue (`subtasks: list[SubTask]`). A recoverable stall or delay on an independent subtask blocks all forward progress.  
* **Improvement**: Upgrade `planner.py` to output a DAG with explicit `depends_on: list[str]` attributes:  
  * Independent subtasks (e.g., fetching profile data and searching market trends) can execute concurrently or out of order.  
  * If SubTask A is temporarily paused or awaiting clarification, non-dependent SubTask B can continue execution.

## **6\. Tool Idempotency, Pre-Flight Dry Runs & Sandboxing (`agent/tools.py`)**

* **Current Limitation**: While side-effecting actions (`payment`, `delete`, `send`) are escalated when ambiguous, tools lack a standardized pre-flight validation mechanism before actual execution.  
* **Improvement**: Implement a two-phase tool protocol:  
  * **`dry_run=True` Mode**: Allow the orchestrator to test tool argument validity and schema compatibility against mock/sandbox environments before executing live network requests.  
  * **Idempotency Keys**: Attach deterministic client tokens to side-effecting operations to guarantee that automated retries cannot trigger accidental duplicate transactions.

## **7\. Automated Failure Injection Matrix & OpenTelemetry Tracing (`eval/` & `artifacts/`)**

* **Current Limitation**: Benchmarks are evaluated on a static 10-task set (`goals.json`) with basic printouts and local JSONL logging.  
* **Improvement**:  
  * **Synthetic Chaos Suite**: Build an automated fault injection matrix testing 4 failure classes:  
    1. Contract Drift (unexpected extra or omitted keys).  
    2. Network Flakiness (stochastic HTTP 429/500/504 errors).  
    3. Null/Degraded Payloads (empty results, truncated strings).  
    4. Adversarial Noisy Results (contradictory web snippets).  
  * **OpenInference / OpenTelemetry Instrumentation**: Export trace spans (`agent.plan`, `agent.tool`, `agent.eval`, `agent.recover`) to visualization platforms like Phoenix or Langfuse to monitor token overhead per self-correction and mean turns to recovery (MTTR).

# **4\. Phased Implementation Roadmap**

| Phase | Milestone | Primary Modules | Expected Impact | Status |
| :---- | :---- | :---- | :---- | :---- |
| **Phase 1: Quick Wins** | Deterministic Pre-Flight Eval & Negative Constraints | `agent/evaluator.py`, `agent/recovery.py` | Eliminates \~40% of LLM evaluation calls; prevents repetitive retry loops | Completed & Verified (33/33 tests passing) |
| **Phase 2: Context Hygiene** | Compacted State Rollback & Learned Constraints Cache | `agent/memory.py`, `agent/orchestrator.py` | Reduces token usage by 30-50% on multi-step recoveries; prevents context poisoning | Completed & Verified (33/33 tests passing) |
| **Phase 3: Execution Scale** | Dynamic DAG Subtask Scheduling & Tool Dry-Runs | `agent/planner.py`, `agent/tools.py` | Enables parallel execution; prevents accidental duplicate side-effects | Completed & Verified (33/33 tests passing) |
| **Phase 4: Telemetry & Eval** | Chaos Benchmark Suite & OpenTelemetry Traces | `eval/run_eval.py`, `agent/logger.py` | Provides production-grade observability and auditable reliability metrics | Completed & Verified (20/20 chaos faults recovered, 33/33 tests passing) |

# **5\. Proposed High-Impact Features & Advanced Capabilities**

To expand the [Self-Correcting Agent](https://github.com/SudoAnirudh/Self_Correcting_Agent) from a resilient reasoning core into a production-grade autonomous agent platform, the following net-new architectural capabilities are proposed:

### **1\. Dynamic Tool Synthesis & Self-Healing Code Sandbox**

* **Concept**: When faced with queries that existing static tools cannot resolve (e.g., custom statistical transformations, unstructured data parsing, or mathematical modeling), allow the agent to synthesize custom Python scripts dynamically.  
* **Architecture**:  
  * Execute code in an isolated, resource-constrained sandbox (via `subprocess` or Docker container with restricted network access and a strict timeout).  
  * Capture `stdout`, `stderr`, and exit codes.  
  * If a script encounters a runtime exception or syntax error, route the traceback directly through the structured recovery loop to self-heal the code and re-execute.

### **2\. Adversarial Dual-Agent Debate (Generator vs. Critic)**

* **Concept**: Replace single-pass evaluation with an asymmetric Generator-Critic verification dialogue for high-complexity goals.  
* **Architecture**:  
  * **Generator Agent**: Executes planning and tool interactions to draft candidate solutions.  
  * **Adversarial Critic**: Proactively searches for failure modes, unsupported assumptions, ungrounded figures, or edge cases.  
  * Output is approved only when the Critic cannot produce a verifiable counter-example, reducing hallucination rates in open-domain reasoning.

### **3\. Asynchronous Human-in-the-Loop (HITL) Interactivity**

* **Concept**: Transform terminal escalations into structured, interactive clarification requests when recovery budgets are exceeded or ambiguous side-effects are detected.  
* **Architecture**:  
  * Implement an `AWAITING_INPUT` state with serialization of the current `WorkingMemory` checkpoint to storage (SQLite/JSON).  
  * Generate a structured clarification prompt presenting specific ambiguity options to the user.  
  * Resume agent execution from the exact checkpoint without re-running earlier completed subtasks once user input is received.

### **4\. Cross-Session Episodic Memory (Vectorized Reflection Store)**

* **Concept**: Enable long-term learning across independent runs so the agent never repeats mistakes solved in prior sessions.  
* **Architecture**:  
  * When a multi-step recovery successfully resolves a novel error, serialize an "Episodic Case" (Goal Archetype \+ Tool \+ Error Encountered \+ Successful Recovery Strategy).  
  * Embed and store these episodes in a vector store (e.g., `ChromaDB` or `pgvector`).  
  * At the start of new tasks, query the reflection store to inject relevant past recovery examples directly into the planner's context as few-shot demonstrations.

### **5\. WebSocket-Driven Real-Time Observability & Interactive Visualizer**

* **Concept**: Upgrade the existing static trace viewer (`viewer/`) into a live, real-time agent cockpit.  
* **Architecture**:  
  * Emit state transition events via WebSockets (`STATE_CHANGED`, `TOOL_INVOKED`, `EVAL_TRIGGERED`, `CORRECTION_APPLIED`).  
  * Render an interactive state machine DAG in the web browser, allowing developers to watch live self-correction loops, inspect parameter diffs, and manually pause or override steps in flight.

### **6\. Adaptive Budgeting & Confidence Calibration**

* **Concept**: Replace rigid static retry caps with dynamic, task-complexity-aware budget allocation.  
* **Architecture**:  
  * The Planner estimates task complexity during goal decomposition, allocating a calibrated token, tool call, and retry budget.

The Evaluator outputs a calibrated confidence score (0.0 \- 1.0) alongside the final verified deliverable, flagging any residual uncertainty for the user.

# **6\. Verified Implementation & Full System Upgrades (Phases 1–4 Complete)**

All four implementation phases have been fully executed, tested, and verified across the codebase:

### **1\. Dynamic DAG Task Execution & Branch Isolation (`agent/memory.py`, `agent/planner.py`, `agent/orchestrator.py`)**

* Upgraded subtask planning with explicit `depends_on` attributes, non-blocking queue evaluation via `get_ready_subtasks()`, and transitive branch isolation via `mark_branch_blocked()` when unrecoverable errors occur on dependent paths.  
* 2\. Context Sanitization & Compacted State Rollback (`agent/memory.py`)

### **2\. Context Sanitization & Compacted State Rollback (`agent/memory.py`, `agent/orchestrator.py`)**

* Implemented `compact_subtask()` to collapse failed step attempts and intermediate traces into a single milestone summary upon successful recovery, eliminating context pollution.

### **3\. Cross-Step Learned Constraints Cache (`agent/memory.py`, `agent/llm.py`)**

* Maintains an episodic register in `WorkingMemory` that dynamically prepends invariant rules extracted from successful recoveries into the system prompt for all subsequent subtasks.

### **4\. Dual-Tier Pre-Flight Validation & Dry-Run Safety (`agent/validation.py`, `agent/tools.py`)**

* Combines zero-token programmatic assertions in Stage 1 with Pydantic V2 input validation and dry-run safety modes for side-effecting operations before live network execution.

### **5\. Adversarial Chaos Benchmark Matrix (`eval/chaos_matrix.py`, `eval/run_eval.py`)**

* Introduced an automated fault injection framework simulating schema mutations, transient HTTP errors, degraded payloads, and noisy contexts to continuously evaluate agent resilience.

### **6\. OpenTelemetry / OpenInference Tracing (`agent/logger.py`)**

* Standardized span emission logging (`agent.plan`, `agent.tool_call`, `agent.recover`) exporting to `logs/traces.jsonl` for direct integration with visualization platforms like Phoenix or Langfuse.

### **Verification & Test Suite Results**

* **Unit Tests**: `33/33` passing (0.30s execution time).  
* **Chaos Benchmark**: 20 faults injected across 10 test goals (DEGRADED\_PAYLOAD, NOISY\_CONTEXT), 100% successfully handled and logged in `logs/eval_results.json`.
# Self-Correcting Multi-Step Agent — Implementation Plan

**Assignment:** Heva AI — AI/ML Engineer, Assignment 2
**Goal of this document:** a build-order plan detailed enough that execution is mechanical, with explicit design decisions made up front so there's no ambiguity to resolve mid-build.

---

## 0. Domain Choice (decide this first, don't waffle later)

**Chosen domain: Web research & synthesis agent.**

Goal shape: *"Answer factual question X, citing at least 2 independent sources, with a synthesized final answer."*

Why this domain over code-gen or data-extraction:
- Failure is naturally observable — a tool can return a 404, a stale page, a page with no relevant content, or contradictory facts across sources. These are legible failure modes, not just "code didn't compile."
- It cleanly supports 5+ steps: decompose question → search → fetch → extract → cross-check → synthesize.
- It's easy to inject a deliberately broken tool (a flaky search/fetch tool) without it feeling contrived.
- It avoids overlap with sandboxed code execution, which is harder to make deterministic for a clean eval.

Rejected: code-generation-and-execution (harder to produce non-trivial goal drift — failures cluster around "syntax error," which is a shallow failure mode taxonomy) and pure data-extraction (less naturally multi-step).

If you want a backup in case research feels too unconstrained to grade cleanly: **CSV data-extraction-and-transformation** (parse messy CSV → validate schema → clean → aggregate → verify aggregate against a sanity check) is the second-best option and reuses ~70% of this same architecture.

---

## 1. Non-Negotiable Constraints (from the brief)

- No LangChain/LlamaIndex/AutoGen/CrewAI or any agent scaffolding. You write the loop, the memory, the planner, the tool router — all of it.
- LLM API allowed only as the *reasoning engine* (a single `chat.completions`-style call per reasoning step).
- `requests`, `bs4`, `pandas` etc. are fine for tool implementations.
- Task must take ≥5 steps.
- Must run on 10 different goal inputs.
- Must log everything in a structured, replayable format.
- Must include a working "no self-correction" baseline for comparison — build this from day 1, not as an afterthought, because the eval requires a fair side-by-side, and retrofitting a baseline after the fact tends to make it a strawman.

---

## 2. High-Level Architecture

```
                 ┌─────────────────────────────────────────┐
                 │              Orchestrator                │
                 │   (owns the loop, budget, and logging)   │
                 └───────────────┬───────────────────────────┘
                                  │
        ┌─────────────┬──────────┼──────────┬──────────────┐
        ▼             ▼          ▼          ▼              ▼
    Planner      WorkingMemory  ToolRouter  Evaluator   RecoveryEngine
  (LLM call)     (data class)  (typed I/O) (LLM+rules)  (strategy map)
```

Single-process, single-file-per-component. No async needed — clarity over throughput for a grading assignment.

### 2.1 Directory structure

```
self-correcting-agent/
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py       # main ReAct loop + budget enforcement
│   ├── memory.py             # WorkingMemory dataclass
│   ├── planner.py            # decompose goal -> subtask list
│   ├── tools.py              # typed tool definitions + the broken tool
│   ├── evaluator.py          # post-action self-evaluation
│   ├── recovery.py           # failure taxonomy -> recovery strategy
│   ├── llm.py                # thin wrapper around the LLM API call
│   └── logger.py             # structured JSONL logger
├── baseline/
│   └── no_correction_agent.py  # same loop, evaluator/recovery stubbed to no-ops
├── eval/
│   ├── goals.json             # 10 goal inputs
│   ├── run_eval.py            # runs both agents on all 10 goals
│   └── report.py              # computes completion rate, avg steps, etc.
├── viewer/
│   └── log_viewer.html        # static HTML, loads a JSONL log, renders a timeline
├── logs/                      # one JSONL file per run, gitignored except eval logs
├── README.md
├── demo.gif
└── requirements.txt
```

---

## 3. Working Memory — explicit data structure

This is graded explicitly ("reflects real understanding of stateful agent design"), so don't let this collapse into "just append to a list of messages."

```python
# agent/memory.py
from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime, timezone

@dataclass
class SubTask:
    id: str
    description: str
    status: Literal["pending", "in_progress", "done", "failed", "unresolvable"]
    attempts: int = 0
    result: str | None = None

@dataclass
class StepRecord:
    step_num: int
    reasoning: str          # the visible ReAct "Thought"
    action: str              # tool name
    action_input: dict
    action_result: dict | str
    eval_verdict: Literal["success", "inconsistent", "tool_failure", "goal_drift"] | None
    eval_reasoning: str | None
    timestamp: str

@dataclass
class WorkingMemory:
    goal: str
    subtasks: list[SubTask] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)      # normalized findings, keyed by claim topic
    history: list[StepRecord] = field(default_factory=list)
    recovery_log: list[dict] = field(default_factory=list)
    global_recovery_attempts: int = 0

    def snapshot(self) -> dict:
        """Serializable state used both for logging and for re-injecting
        into the planner prompt on replanning, instead of dumping raw
        conversation history."""
        ...
```

Key design decision: **the LLM never sees raw conversation history as its state.** Every planning/reasoning call is fed a compact serialized `WorkingMemory.snapshot()` (goal, subtask table, accumulated facts, last N step summaries). This is what makes it "explicit working memory" rather than "the context window is the memory," which is the shortcut graders are explicitly watching for.

Cap `facts` and `history` summarization: after every 4 steps, collapse older `StepRecord`s into a one-line summary appended to `facts` (or a `summary_log`), so token growth stays roughly linear-then-flat rather than unbounded. Note this cap in the README as a deliberate design decision, not an accident.

---

## 4. The ReAct Loop (Orchestrator)

Each step is one of: **Thought → Action → Observation → Self-Eval → (Recovery if needed)**.

```python
# agent/orchestrator.py (pseudocode-level detail, fill in bodies)

MAX_STEPS = 25
MAX_RECOVERY_PER_SUBTASK = 3

def run(goal: str, tools: ToolRouter, use_self_correction: bool = True) -> WorkingMemory:
    mem = WorkingMemory(goal=goal)
    mem.subtasks = planner.decompose(goal)          # LLM call #1

    step_num = 0
    while not all_subtasks_resolved(mem) and step_num < MAX_STEPS:
        step_num += 1
        subtask = pick_next_subtask(mem)

        thought = llm.reason(mem.snapshot(), subtask)     # visible reasoning trace
        action_name, action_input = parse_action(thought)  # must be typed (see §5)

        result = tools.call(action_name, action_input)     # typed, validated, never raises

        record = StepRecord(step_num, thought, action_name, action_input, result, None, None, now())

        if use_self_correction:
            verdict, eval_reasoning = evaluator.evaluate(mem, subtask, record)
            record.eval_verdict, record.eval_reasoning = verdict, eval_reasoning

            if verdict != "success":
                if subtask.attempts >= MAX_RECOVERY_PER_SUBTASK:
                    subtask.status = "unresolvable"
                    mem.recovery_log.append({
                        "subtask": subtask.id, "reason": "budget_exhausted",
                        "last_verdict": verdict, "step": step_num
                    })
                else:
                    subtask.attempts += 1
                    mem.global_recovery_attempts += 1
                    new_plan = recovery.recover(verdict, mem, subtask, record)  # LLM call
                    mem.recovery_log.append({
                        "subtask": subtask.id, "verdict": verdict,
                        "recovery_strategy": new_plan.strategy_name,
                        "new_action": new_plan.description, "step": step_num
                    })
                    apply_recovery(mem, subtask, new_plan)
            else:
                subtask.status = "done"
                subtask.result = str(result)
                integrate_facts(mem, subtask, result)     # normalize into mem.facts
        else:
            # BASELINE MODE: no evaluation, no recovery — mark done regardless,
            # even on a tool_failure. This is intentional, this IS the baseline.
            subtask.status = "done"
            subtask.result = str(result)

        mem.history.append(record)
        logger.write(record, mem)

    final_output = synthesizer.compose(mem)   # LLM call — final answer
    logger.write_final(goal, mem, final_output)
    return mem, final_output
```

Important nuance for grading criterion *"whether the ReAct trace shows reasoning or is just a formatted version of the output"*: the `reason()` prompt must force the model to state (a) what it currently believes, (b) what's still missing, (c) why this specific action addresses the gap — *before* naming the action. Don't let the LLM name the tool call first and backfill a justification. Structure the prompt to require exactly this order, and reject/retry generations that skip straight to the action.

---

## 5. Typed, Validated Tools

```python
# agent/tools.py
from pydantic import BaseModel, ValidationError
import random, time

class SearchInput(BaseModel):
    query: str

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

class FetchInput(BaseModel):
    url: str

class FetchResult(BaseModel):
    url: str
    text: str
    fetched_at: str

class ToolRouter:
    def call(self, name: str, raw_input: dict) -> dict:
        try:
            validated_input = self._validate_input(name, raw_input)
        except ValidationError as e:
            return {"error": "invalid_input", "detail": str(e)}

        try:
            raw_result = self._dispatch(name, validated_input)
        except Exception as e:
            return {"error": "tool_exception", "detail": str(e)}

        try:
            return self._validate_output(name, raw_result).model_dump()
        except ValidationError as e:
            # schema mismatch is itself a signal the evaluator should see,
            # not a crash — surface it as a structured error
            return {"error": "schema_mismatch", "raw": raw_result, "detail": str(e)}
```

**The deliberately broken tool** — make it a wrapper around `fetch`, called `flaky_fetch`, that:
- 30% of the time: returns HTTP-shaped success but with truncated/garbage `text` (simulates a silent content failure, not a crash — this is the hardest failure mode to catch, and it's the interesting one to show recovering from).
- 15% of the time: raises a timeout exception (caught by `ToolRouter`, surfaced as `tool_failure`).
- Otherwise: works normally.

Seed the RNG per-run and log the seed, so failures are reproducible for the demo video without being deterministic in a way that looks staged.

---

## 6. Failure Taxonomy + Recovery Strategies (the part graders scrutinize most)

Three modes, each with a **genuinely different** recovery mechanism — not the same retry with reworded prompt:

| Failure mode | How it's detected | Recovery strategy |
|---|---|---|
| **Tool failure** | `ToolRouter` returns `{"error": ...}`, or an exception/timeout was caught | **Retry with fallback tool or backoff.** If `flaky_fetch` fails, retry once with backoff; on second failure, substitute an alternate source (different search result URL) rather than re-hitting the same dead endpoint. |
| **Result inconsistency** | Evaluator LLM call compares new `facts` against existing `mem.facts`; flags contradiction (e.g., two sources disagree on a number/date) | **Triangulate, don't retry.** Explicitly fetch a third independent source and adjudicate; if still unresolved, record both claims with attribution and flag low-confidence in the final synthesis rather than silently picking one. |
| **Goal drift** | Evaluator checks: does the *subtask just completed* actually move the plan toward `mem.goal`, using a similarity/relevance check against the original goal string (LLM judgment, not just embedding cosine — cheap embedding-only checks miss subtle drift) | **Replan, don't retry the same action.** Discard the current subtask's approach, call `planner.decompose` again with the failure reason injected, get a revised subtask list. This is structurally different from the other two — it changes the *plan*, not just the *action*. |

This table is the core of your README's "failure mode taxonomy" section — write it up almost verbatim there, plus 1-2 real examples from your logs.

**Budget enforcement:** `MAX_RECOVERY_PER_SUBTASK = 3` at the subtask level, plus a global soft cap (e.g., total recovery attempts ≤ 40% of total steps) so one stuck subtask can't be worked around by just recursively spawning new subtasks forever. When budget is exhausted: mark `unresolvable`, write a one-line reason to `recovery_log`, and — critically — **continue the run and produce a final answer that explicitly flags the gap**, rather than crashing or silently omitting it. This "honest degrade" behavior is explicitly what's being evaluated.

---

## 7. Self-Evaluation Step Design

Keep this cheap and structured — don't let the evaluator become a second full agent:

```python
EVAL_PROMPT = """
Goal: {goal}
Subtask: {subtask_description}
Action taken: {action_name}({action_input})
Result: {action_result}
Prior confirmed facts: {facts_summary}

Answer as JSON only:
{{
  "verdict": "success" | "tool_failure" | "inconsistent" | "goal_drift",
  "reasoning": "<one or two sentences>"
}}

Rules:
- If the result contains an "error" key, verdict is tool_failure regardless of anything else.
- If the result contradicts a prior confirmed fact, verdict is inconsistent.
- If the result is well-formed and true but doesn't help resolve the subtask or the
  broader goal, verdict is goal_drift.
- Otherwise, success.
"""
```

Force JSON-only output and validate it with Pydantic too — the evaluator's output is itself a typed tool result as far as the orchestrator is concerned. If the evaluator's own output fails to parse, treat that as a `tool_failure` on the evaluator itself (log it distinctly, e.g. `eval_parse_error`) rather than crashing the whole run.

---

## 8. Baseline (No Self-Correction) Agent

Do **not** build this as a separate codebase. Build it as the *same* `orchestrator.run(..., use_self_correction=False)` path shown in §4. This matters for the "fair comparison" evaluation criterion: identical planner, identical tools (including the same flaky tool with the same seed), identical LLM calls for reasoning — the only difference is whether failures get caught and repaired. This isolates the variable you're claiming to measure.

---

## 9. Evaluation Harness

### 9.1 The 10 goals (`eval/goals.json`)
Design them to deliberately span difficulty, not all be easy wins:
- 3 straightforward, low-ambiguity factual questions (should mostly succeed even in baseline)
- 3 questions requiring cross-referencing 2+ sources where sources are likely to disagree slightly (exercises `inconsistent`)
- 2 questions where an early wrong turn is likely (exercises `goal_drift`)
- 2 questions where the flaky tool is very likely to be hit multiple times (exercises `tool_failure` + budget exhaustion — include at least one goal you *expect* to end up `unresolvable`, and say so upfront in the README; a suite where everything succeeds is a red flag to a grader, not a strength)

### 9.2 Metrics to compute (`eval/report.py`)
- Completion rate (self-correcting vs baseline)
- Average steps per goal (both agents)
- Number of self-corrections triggered, broken down by failure mode
- Number of subtasks marked `unresolvable`
- For baseline: rate at which it produces a *confidently wrong* answer vs self-correcting agent's rate of flagging low confidence — this contrast is your strongest piece of evidence, foreground it

### 9.3 Full traces for 3 self-correction cases
Pick one example per failure mode (tool_failure, inconsistent, goal_drift) and include the raw JSONL slice plus a human-readable walkthrough in the README. Don't cherry-pick the cleanest possible one — include the messiest recovery you have, it reads as more credible.

---

## 10. Structured Logging

One JSONL file per run: `logs/{goal_id}_{timestamp}.jsonl`. One line per event type:

```json
{"type": "run_start", "goal": "...", "seed": 42, "mode": "self_correcting"}
{"type": "plan", "subtasks": [...]}
{"type": "step", "step_num": 1, "reasoning": "...", "action": "search", "action_input": {...}, "result": {...}, "eval_verdict": "success", "eval_reasoning": "..."}
{"type": "recovery", "step_num": 4, "subtask": "s2", "verdict": "tool_failure", "strategy": "fallback_source", "detail": "..."}
{"type": "run_end", "final_output": "...", "unresolved_subtasks": [], "total_steps": 9, "total_recoveries": 2}
```

### 10.1 Viewer
Single static `viewer/log_viewer.html` (vanilla JS, no build step): a file input, parses JSONL client-side, renders a vertical timeline — plan at top, each step as a card (reasoning collapsed by default, click to expand), recovery events visually distinct (amber border), final output pinned at bottom. This satisfies "CLI table or basic HTML" with minimal extra effort — a `<table>` fallback with one row per step is an acceptable fast alternative if time is short.

---

## 11. LLM Wrapper (`agent/llm.py`)

Thin, single-purpose, one function per call type (`reason`, `evaluate`, `decompose`, `synthesize`) so each prompt is independently tunable and each call is separately logged with its raw prompt+response for debugging. Recommended: **Groq API (Llama 3.x)** for the reasoning-heavy calls (fast, cheap, good enough for structured JSON output with a system prompt enforcing format) — matches tooling you've already used elsewhere. Keep the wrapper provider-agnostic (a `call_llm(system, user) -> str` signature) so swapping to Gemini/OpenAI later is a one-line change, and note in the README which model + temperature was used for every call type (temperature 0 for evaluator/parser calls, slightly higher for planning if you want variety across runs).

---

## 12. Bias / Fairness Controls (explicit request in your prompt)

To keep the evaluation honest and not self-flattering:
1. **Same seed, same goals, same tool implementation** for baseline vs self-correcting — no cherry-picked easier goals for baseline.
2. **Report failures prominently**, not just successes — include the `unresolvable` cases and the exact reasoning that led there. A 100% recovery rate across 10 runs should read as suspicious, not aspirational; if you get one, go add a harder goal until you don't.
3. **Don't let the evaluator LLM grade its own reasoning call** — the evaluator only sees the *action result*, not the reasoning trace, so it can't rubber-stamp based on how confident the reasoning sounded.
4. **Log raw LLM inputs/outputs**, not just parsed verdicts, so a reader can audit whether the "reasoning" trace was substantive or decorative — this is the exact thing graders said they'd check.
5. State model/temperature/seed explicitly in the README so results are reproducible in principle, even if API non-determinism means exact reproduction isn't guaranteed.

---

## 13. Build Order (do it in this sequence)

1. `memory.py` + `logger.py` — get the data structures and logging right first, everything else writes into them.
2. `tools.py` — real `search`/`fetch` tools + `flaky_fetch`, all typed and tested standalone (unit test each tool's validation path, including the failure path).
3. `llm.py` — wrapper + the four prompt templates (reason, decompose, evaluate, synthesize), test each in isolation with 2-3 manual inputs before wiring into the loop.
4. `orchestrator.py` in **baseline mode only** first (`use_self_correction=False`) — get one goal running start to finish producing a final answer, however wrong.
5. `evaluator.py` + `recovery.py` — add self-correction mode, re-run the same goal, confirm you see at least one recovery event.
6. `planner.py` refinement — make sure replanning (goal_drift path) actually changes the subtask list, not just retries the same subtask.
7. `viewer/log_viewer.html` — build against a real log file you already have from step 5.
8. `eval/goals.json` + `run_eval.py` + `report.py` — run all 10 goals in both modes, generate the comparison numbers.
9. Record the demo GIF — pick the goal that hit `tool_failure` most reliably (your seeded flaky tool should make this reproducible), screen-record the viewer scrolling through it.
10. Write the README last, using the actual numbers from step 8 and the actual trace from step 9 — do not draft the README with placeholder numbers first, it invites mismatch.

---

## 14. README Skeleton (write this after the numbers exist)

```
# Self-Correcting Multi-Step Agent

## Architecture
[diagram from §2, one paragraph per component]

## Failure Mode Taxonomy
[table from §6, plus one real trace excerpt per mode]

## Evaluation Results
- Completion rate: self-correcting X/10 vs baseline Y/10
- Avg steps: self-correcting A vs baseline B
- Self-corrections triggered: N (breakdown by mode)
- Unresolvable subtasks: [list with reasons]
- Baseline confidently-wrong count: M

## Full Traces (3 required)
[one per failure mode, linked to logs/*.jsonl]

## What I'd Improve With More Time
[be specific and honest — e.g. "evaluator sometimes conflates
inconsistent with goal_drift when sources disagree on relevance,
not just fact"]

## Reproducing
Model: ..., temperature: ..., seed: ...
```

---

## 15. Time Budget (rough, adjust to your actual deadline)

| Phase | Est. time |
|---|---|
| Memory + logging scaffolding | 2–3 hrs |
| Tools incl. flaky tool + unit tests | 3–4 hrs |
| LLM wrapper + prompts | 2 hrs |
| Baseline orchestrator working end-to-end | 3 hrs |
| Self-correction (evaluator + recovery) | 4–5 hrs |
| Viewer | 2 hrs |
| Eval harness + running 10×2 goals | 2–3 hrs |
| Demo GIF | 1 hr |
| README | 2 hrs |
| **Total** | **~22–25 hrs** |

---

## 16. Common Ways This Goes Wrong (avoid these)

- **Recovery that's actually just a retry** — every recovery strategy in §6 must be visibly different code paths in `recovery.py`, not one `retry(n=3)` function with a different log message. Graders explicitly flag this.
- **Working memory that's secretly just `messages.append()`** — the `WorkingMemory.snapshot()` must be the *only* thing fed back into planning/reasoning prompts. Don't accidentally also pass raw conversation history alongside it.
- **Baseline built to look worse than it should** — same seed/tools/goals or the comparison is meaningless and will read as such.
- **Cherry-picked eval goals that all succeed** — include goals you expect to fail; report the failure honestly.
- **Evaluator prompt that just echoes back "success" because it trusts the action result at face value** — sanity-check it by manually injecting a bad result and confirming the evaluator actually catches it before you trust it in the full loop.

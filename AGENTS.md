# AGENTS.md

Guidance for any coding agent (Claude Code, Cursor, Copilot Workspace,Antigarvity,Gemini etc.) working in this
repository. This file takes precedence over generic defaults. Read this in full before writing code.

---

## Project

A self-correcting multi-step ReAct agent for web research & synthesis, built from scratch with no
agent framework. Full spec: `docs/implementation_plan.md` (the implementation plan already produced
for this project — read it before touching code; it is the source of truth for architecture,
failure taxonomy, and build order). This file is about *how to work in the repo*, not what to build —
defer to the plan doc for design decisions.

---

## Hard Constraints — do not violate

- **No agent frameworks.** No LangChain Agents, LlamaIndex Agents, AutoGen, CrewAI, or any pre-built
  agent scaffolding, planner, or memory abstraction. `requests`, `bs4`, `pandas`, `pydantic` are fine.
  If you catch yourself about to `pip install langchain` — stop, that's a constraint violation, not a
  shortcut.
- **No silent fallback to a framework "just for typing" or "just for the tool router.**" Write the
  router, the loop, and the memory structure by hand, even if a library version would be shorter.
- Every tool call must go through `ToolRouter.call()` — never call a tool function directly from the
  orchestrator.
- Every LLM call must go through `agent/llm.py` — never call the API client directly elsewhere.
- The baseline agent and the self-correcting agent must share the same `orchestrator.run()` code
  path, differing only via the `use_self_correction` flag. Do not fork them into two separate loops.

---

## Repo Layout

```
self-correcting-agent/
├── agent/
│   ├── orchestrator.py
│   ├── memory.py
│   ├── planner.py
│   ├── tools.py
│   ├── evaluator.py
│   ├── recovery.py
│   ├── llm.py
│   └── logger.py
├── baseline/                  # thin entrypoint only, reuses agent/orchestrator.py
├── eval/
│   ├── goals.json
│   ├── run_eval.py
│   └── report.py
├── viewer/
│   └── log_viewer.html
├── logs/                      # gitignored except a handful of committed sample runs
├── tests/
│   └── test_*.py
├── docs/
│   └── implementation_plan.md
├── README.md
├── demo.gif
└── requirements.txt
```

Do not introduce new top-level directories without a reason stated in the commit message.

---

## Build Order (follow this sequence; do not build the self-correction layer before the baseline works)

1. `agent/memory.py` — `WorkingMemory`, `SubTask`, `StepRecord` dataclasses. Write unit tests for
   `WorkingMemory.snapshot()` before anything else consumes it.
2. `agent/logger.py` — JSONL writer, one line per event (`run_start`, `plan`, `step`, `recovery`,
   `run_end`). Match the schema in the implementation plan §10 exactly — the viewer depends on it.
3. `agent/tools.py` — `SearchInput/Result`, `FetchInput/Result` Pydantic models, `ToolRouter`, and
   `flaky_fetch` (the intentionally broken tool: ~30% truncated/garbage success, ~15% timeout
   exception, seeded RNG, seed logged in `run_start`). Unit test all three code paths (success,
   garbage, exception) independently of the orchestrator.
4. `agent/llm.py` — one function per call type: `reason()`, `decompose()`, `evaluate()`, `synthesize()`.
   Provider-agnostic wrapper (`call_llm(system, user) -> str`). Log raw prompt + raw response for
   every call, even in production runs, not just debug mode.
5. `agent/orchestrator.py` in **baseline mode only** (`use_self_correction=False`). Get one goal
   running end-to-end producing *some* final answer before adding self-correction. Commit this as a
   working checkpoint before proceeding.
6. `agent/evaluator.py` + `agent/recovery.py` — add self-correction. Re-run the same goal from step 5,
   confirm at least one recovery event appears in the log.
7. `agent/planner.py` refinement — confirm the `goal_drift` recovery path actually calls
   `planner.decompose()` again and changes the subtask list, not just retries the same subtask with
   different wording.
8. `viewer/log_viewer.html` — build against a real committed sample log, not synthetic data.
9. `eval/goals.json`, `eval/run_eval.py`, `eval/report.py` — run all 10 goals × 2 modes, generate
   comparison numbers.
10. `demo.gif` and `README.md` last, using real numbers/traces from step 9. Do not draft the README
    with placeholder numbers.

If asked to "just implement the whole thing," still execute in this order and commit at each
numbered checkpoint — don't produce all files in one shot with no incremental verification.

---

## Design Rules the Agent Must Enforce Itself

- **Working memory is the only state fed to LLM prompts.** Never pass raw `history` list or full
  conversation transcript into a prompt — only `WorkingMemory.snapshot()`. If you find yourself
  concatenating `StepRecord` objects directly into a prompt string, stop and route it through
  `snapshot()` instead.
- **Recovery strategies must be distinct code paths**, not one `retry(n=3)` helper reused three times
  with different log messages. `recovery.py` should have three clearly separate functions:
  `recover_tool_failure`, `recover_inconsistency`, `recover_goal_drift`. If two of them end up with
  near-identical bodies, that's a signal to revisit the design, not proceed.
- **The evaluator only sees the action result, not the reasoning trace** that led to the action — it
  must not be able to rubber-stamp a bad result just because the preceding "Thought" sounded
  confident.
- **Reasoning must precede action naming in every `reason()` prompt output.** The prompt template
  must force the model to state current belief → gap → why-this-action, in that order. If you're
  writing or editing this prompt, don't let it collapse into "name the tool, then justify it."
- **Never let a tool call or LLM parse failure raise an uncaught exception up through the
  orchestrator loop.** Every tool result and every evaluator/planner LLM response must be validated
  (Pydantic) and converted into a structured error dict on failure — this structured error is itself
  the signal the evaluator reads, not an exception to catch and hide.
- **Respect the recovery budget exactly as specified**: `MAX_RECOVERY_PER_SUBTASK = 3`, plus a global
  soft cap of recovery attempts ≤ 40% of total steps. On exhaustion, mark `unresolvable`, write the
  reason to `recovery_log`, and continue the run to produce a final answer that explicitly flags the
  gap — do not crash, and do not silently drop the unresolved subtask from the final synthesis.

---

## Testing Expectations

- Every module in `agent/` gets a corresponding `tests/test_<module>.py`.
- `tools.py`: test the validation-failure path and the exception-caught path explicitly, not just the
  happy path.
- `evaluator.py`: test by injecting a deliberately bad/contradictory tool result and asserting the
  verdict is *not* `success`. This is the single most important test in the repo — if it's missing or
  weak, the whole self-correction claim is unverified.
- Run `pytest` before every commit that touches `agent/`. Do not commit with failing tests.
- No test may depend on live network calls succeeding — mock `requests`/tool calls in unit tests;
  reserve real network calls for `eval/run_eval.py` runs only.

---

## Logging & Reproducibility Rules

- Every run writes `logs/{goal_id}_{timestamp}.jsonl` matching the schema in the implementation plan.
- `run_start` must include the RNG seed and the model name/temperature used for every call type.
- Never overwrite an existing log file; always generate a fresh timestamped filename.
- Commit 2-3 representative sample logs (one clean success, one with a recovery, one `unresolvable`
  case) to `logs/samples/` so the viewer and grader have something to inspect without running the
  agent themselves.

---

## Eval Harness Rules

- `eval/goals.json` must contain exactly 10 goals, spanning the difficulty spread specified in the
  implementation plan §9.1 (straightforward / cross-reference-with-likely-disagreement /
  drift-prone / flaky-tool-heavy). Do not make all 10 goals easy — a suite with 10/10 success in both
  modes is a red flag, not a result to aim for.
- `run_eval.py` must run both baseline and self-correcting modes on identical goals with identical
  seeds, and must not special-case or simplify any goal for the baseline run.
- `report.py` output must include: completion rate (both modes), avg steps (both modes),
  self-corrections triggered by failure-mode breakdown, count of `unresolvable` subtasks, and a
  count/estimate of baseline "confidently wrong" answers.

---

## Commit Hygiene

- Small, checkpointed commits matching the build-order steps above — not one giant initial commit.
- Commit message format: `<component>: <what changed>` (e.g. `evaluator: detect inconsistency via
  fact comparison against WorkingMemory.facts`).
- Do not commit `logs/*.jsonl` in bulk — only the curated samples in `logs/samples/`.
- `requirements.txt` must be kept minimal and pinned; no unused dependencies left in from
  experimentation.

---

## When Unsure

- Prefer the explicit, hand-rolled version over a clever abstraction — this assignment is graded on
  visible, inspectable agent logic, not elegance.
- If a design choice isn't covered here or in `docs/implementation_plan.md`, make the simplest choice
  that keeps failure modes observable and recovery strategies distinct, and note the decision + reason
  in the relevant module's docstring rather than silently picking one.
- Never quietly swap in a framework-provided agent loop, memory store, or planner "temporarily to get
  something working" — this is the one constraint with zero tolerance in the grading rubric.

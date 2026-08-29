import time
from datetime import datetime, timezone
from typing import Tuple, Optional
from dataclasses import asdict
from agent.memory import WorkingMemory, SubTask, StepRecord
from agent.logger import AgentLogger
from agent.tools import ToolRouter
from agent import llm
from agent import planner
from agent.evaluator import evaluate_step
from agent.validation import choose_recovery

MAX_STEPS = 25
MAX_RECOVERY_PER_SUBTASK = 3


def pick_next_subtask(mem: WorkingMemory) -> Optional[SubTask]:
    for task in mem.subtasks:
        if task.status in ("pending", "in_progress"):
            return task
    return None


def _fingerprint(value) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _reformulate_subtask(mem: WorkingMemory, subtask: SubTask, finding: dict) -> None:
    """Ask the reasoner for a new approach without granting it acceptance authority."""
    snap = mem.snapshot()
    proposal = llm.reason(snap, {
        "id": subtask.id,
        "description": subtask.description,
        "status": "pending",
        "attempts": subtask.attempts,
        "result": subtask.result,
        "recovery": {
            "reason_code": finding.get("reason_code"),
            "details": finding.get("details"),
            "evidence": finding.get("evidence", {})
        }
    })
    action = proposal.get("action", "")
    if action:
        subtask.description = f"{subtask.description} | Reformulate using a different approach after {finding.get('reason_code')}"
    subtask.status = "pending"


def run(goal: str, tools: ToolRouter, use_self_correction: bool = True) -> Tuple[WorkingMemory, str]:
    mode = "self_correcting" if use_self_correction else "baseline"
    logger = AgentLogger(goal=goal, mode=mode)
    mem = WorkingMemory(goal=goal)

    subtasks = planner.decompose(goal)
    mem.subtasks = subtasks
    logger.log_plan([{"id": t.id, "description": t.description, "status": t.status} for t in mem.subtasks])
    logger.log_run_start(goal, tools.seed, mode, {
        "reasoning": {"model": llm.REASONING_MODEL, "temperature": 0.0},
        "evaluation": {"mode": "deterministic"}
    })

    step_num = 0
    seen_fingerprints = set()

    while step_num < MAX_STEPS:
        subtask = pick_next_subtask(mem)
        if not subtask:
            break
        step_num += 1
        subtask.status = "in_progress"

        thought = llm.reason(mem.snapshot(), {
            "id": subtask.id,
            "description": subtask.description,
            "status": subtask.status,
            "attempts": subtask.attempts,
            "result": subtask.result
        })
        action = thought.get("action")
        action_input = thought.get("action_input", {})
        reasoning = (
            f"Belief: {thought.get('belief', '')} | Gap: {thought.get('gap', '')} | "
            f"Why Action: {thought.get('why_action', '')}"
        )

        result = tools.call(action, action_input)
        record = StepRecord(
            step_num=step_num,
            reasoning=reasoning,
            action=action,
            action_input=action_input,
            action_result=result,
            eval_verdict=None,
            eval_reasoning=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        if use_self_correction:
            previous = mem.history[-1].action_result if mem.history and mem.history[-1].action == action else None
            finding = evaluate_step(goal, subtask.description, asdict(record), mem.facts, previous)
            record.eval_verdict = finding["status"]
            record.eval_reasoning = finding["details"]
            fingerprint = _fingerprint(result)
            repeated = fingerprint in seen_fingerprints
            seen_fingerprints.add(fingerprint)

            decision = choose_recovery(
                __import__("agent.validation", fromlist=["ValidationFinding"]).ValidationFinding(
                    status=finding["status"], reason_code=finding["reason_code"],
                    details=finding["details"], progress=finding["progress"],
                    acceptance_ready=finding["acceptance_ready"], evidence=finding["evidence"]
                ),
                action, subtask.attempts, MAX_RECOVERY_PER_SUBTASK, repeated
            )

            if decision.action == "stop" and finding["acceptance_ready"] and finding["status"] == "success":
                subtask.status = "done"
                subtask.result = str(result)
                facts = llm.extract_facts(goal, subtask.description, result)
                mem.facts.update(facts)
            elif decision.action == "escalate":
                subtask.status = "unresolvable"
                subtask.result = decision.reason
                logger.log_recovery(subtask.id, "escalate", decision.reason)
            elif decision.action == "reformulate":
                subtask.attempts += 1
                mem.global_recovery_attempts += 1
                _reformulate_subtask(mem, subtask, finding)
                logger.log_recovery(subtask.id, "reformulate", decision.reason)
            elif decision.action == "retry":
                subtask.attempts += 1
                mem.global_recovery_attempts += 1
                subtask.status = "pending"
                logger.log_recovery(subtask.id, "retry", decision.reason)
            else:
                subtask.status = "unresolvable"
                subtask.result = decision.reason
                logger.log_recovery(subtask.id, "stop", decision.reason)
        else:
            subtask.status = "done"
            subtask.result = str(result)
            mem.facts.update(llm.extract_facts(goal, subtask.description, result))

        mem.add_step(record)
        logger.log_step(record)

    unresolved = [
        {"id": t.id, "description": t.description, "status": t.status, "attempts": t.attempts, "result": t.result}
        for t in mem.subtasks if t.status not in ("done", "unresolvable")
    ]
    final_output = llm.synthesize(goal, mem.facts, mem.summary_log, unresolved)
    logger.log_run_end(final_output, unresolved, step_num, mem.global_recovery_attempts)
    return mem, final_output

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime, timezone

@dataclass
class SubTask:
    id: str
    description: str
    status: Literal["pending", "in_progress", "done", "failed", "unresolvable"]
    attempts: int = 0
    result: Optional[str] = None

@dataclass
class StepRecord:
    step_num: int
    reasoning: str          # the visible ReAct "Thought"
    action: str              # tool name
    action_input: dict
    action_result: Any
    eval_verdict: Optional[Literal["success", "inconsistent", "tool_failure", "goal_drift"]]
    eval_reasoning: Optional[str]
    timestamp: str

@dataclass
class WorkingMemory:
    goal: str
    subtasks: List[SubTask] = field(default_factory=list)
    facts: Dict[str, str] = field(default_factory=dict)      # normalized findings, keyed by claim topic
    history: List[StepRecord] = field(default_factory=list)
    summary_log: List[str] = field(default_factory=list)      # collapsed older steps
    recovery_log: List[dict] = field(default_factory=list)
    global_recovery_attempts: int = 0

    def add_step(self, record: StepRecord):
        """Append a new StepRecord and truncate/summarize history if it exceeds 4 items."""
        self.history.append(record)
        if len(self.history) > 4:
            oldest = self.history.pop(0)
            res_str = str(oldest.action_result)
            if len(res_str) > 100:
                res_str = res_str[:97] + "..."
            summary = (
                f"Step {oldest.step_num} ({oldest.action}): "
                f"Thought: {oldest.reasoning[:80]}... "
                f"Result: {res_str} -> Verdict: {oldest.eval_verdict}"
            )
            self.summary_log.append(summary)

    def snapshot(self) -> dict:
        """Serializable state used for LLM prompts and logging."""
        return {
            "goal": self.goal,
            "subtasks": [
                {
                    "id": task.id,
                    "description": task.description,
                    "status": task.status,
                    "attempts": task.attempts,
                    "result": task.result
                } for task in self.subtasks
            ],
            "facts": self.facts,
            "summary_log": self.summary_log,
            "history": [
                {
                    "step_num": r.step_num,
                    "reasoning": r.reasoning,
                    "action": r.action,
                    "action_input": r.action_input,
                    "action_result": r.action_result,
                    "eval_verdict": r.eval_verdict,
                    "eval_reasoning": r.eval_reasoning,
                    "timestamp": r.timestamp
                } for r in self.history
            ],
            "recovery_log": self.recovery_log,
            "global_recovery_attempts": self.global_recovery_attempts
        }

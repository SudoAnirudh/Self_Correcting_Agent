import os
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any
from agent.memory import StepRecord

class AgentLogger:
    def __init__(self, goal: str, mode: str):
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create a clean slug from the goal
        clean_goal = re.sub(r'[^a-zA-Z0-9]+', '_', goal.lower())
        clean_goal = clean_goal.strip('_')[:40]
        if not clean_goal:
            clean_goal = "goal"
            
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.filename = f"logs/{clean_goal}_{timestamp}_{mode}.jsonl"
        self._write_line({"type": "init", "goal": goal, "timestamp": self._now()})

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_line(self, data: dict):
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def log_run_start(self, goal: str, seed: int, mode: str, models_info: dict):
        self._write_line({
            "type": "run_start",
            "event": "run_start",
            "goal": goal,
            "seed": seed,
            "mode": mode,
            "models_info": models_info,
            "timestamp": self._now()
        })

    def log_plan(self, subtasks: List[dict]):
        self._write_line({
            "type": "plan",
            "event": "plan",
            "subtasks": subtasks,
            "timestamp": self._now()
        })

    def log_step(self, record: StepRecord):
        self._write_line({
            "type": "step",
            "event": "step",
            "step_num": record.step_num,
            "reasoning": record.reasoning,
            "action": record.action,
            "action_input": record.action_input,
            "result": record.action_result,
            "action_result": record.action_result,
            "eval_verdict": record.eval_verdict,
            "eval_reasoning": record.eval_reasoning,
            "timestamp": record.timestamp
        })

    def log_recovery(self, subtask_id: str, strategy: str, details: str):
        self._write_line({
            "type": "recovery",
            "event": "recovery",
            "subtask": subtask_id,
            "subtask_id": subtask_id,
            "strategy": strategy,
            "detail": details,
            "details": details,
            "timestamp": self._now()
        })

    def log_run_end(self, final_output: str, unresolved_subtasks: List[dict], total_steps: int, total_recoveries: int):
        self._write_line({
            "type": "run_end",
            "event": "run_end",
            "final_output": final_output,
            "unresolved_subtasks": unresolved_subtasks,
            "total_steps": total_steps,
            "total_recoveries": total_recoveries,
            "timestamp": self._now()
        })

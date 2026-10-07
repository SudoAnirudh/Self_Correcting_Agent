import os
import json
import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from agent.memory import StepRecord

class TraceSpan(BaseModel):
    trace_id: str
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    name: str  # agent.plan, agent.tool_call, agent.eval_stage_1, agent.eval_stage_2, agent.recover
    start_time: str
    end_time: str
    attributes: Dict[str, Any] = Field(default_factory=dict)

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
        self.trace_filename = "logs/traces.jsonl"
        self.trace_id = uuid.uuid4().hex[:16]

        self._write_line({"type": "init", "goal": goal, "timestamp": self._now()})

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_line(self, data: dict):
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def emit_span(
        self,
        name: str,
        start_time: str,
        end_time: str,
        attributes: dict,
        parent_span_id: Optional[str] = None
    ) -> TraceSpan:
        """Emit an OpenInference / OpenTelemetry compatible trace span."""
        span = TraceSpan(
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            name=name,
            start_time=start_time,
            end_time=end_time,
            attributes=attributes
        )
        with open(self.trace_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(span.model_dump()) + "\n")
        return span

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
        now = self._now()
        self._write_line({
            "type": "plan",
            "event": "plan",
            "subtasks": subtasks,
            "timestamp": now
        })
        self.emit_span(
            name="agent.plan",
            start_time=now,
            end_time=now,
            attributes={"subtasks_count": len(subtasks)}
        )

    def log_step(self, record: StepRecord):
        now = self._now()
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
        self.emit_span(
            name="agent.tool_call",
            start_time=record.timestamp,
            end_time=now,
            attributes={
                "step_num": record.step_num,
                "tool_name": record.action,
                "eval_verdict": record.eval_verdict or "unknown"
            }
        )

    def log_recovery(self, subtask_id: str, strategy: str, details: str):
        now = self._now()
        self._write_line({
            "type": "recovery",
            "event": "recovery",
            "subtask": subtask_id,
            "subtask_id": subtask_id,
            "strategy": strategy,
            "detail": details,
            "details": details,
            "timestamp": now
        })
        self.emit_span(
            name="agent.recover",
            start_time=now,
            end_time=now,
            attributes={
                "subtask_id": subtask_id,
                "recovery_strategy": strategy,
                "details": details[:200]
            }
        )

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


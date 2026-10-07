import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
from agent.memory import WorkingMemory, SubTask, StepRecord


class HITLRequest(BaseModel):
    checkpoint_id: str
    subtask_id: str
    question: str
    options: List[str] = Field(default_factory=list)
    context_summary: str = ""
    severity: str = "medium"  # low, medium, high, critical
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CheckpointManager:
    """Manages serialization, storage, and retrieval of agent execution state checkpoints for HITL interactivity."""

    def __init__(self, storage_dir: str = "logs/checkpoints"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_checkpoint(self, mem: WorkingMemory, hitl_request: HITLRequest) -> str:
        checkpoint_id = hitl_request.checkpoint_id or f"chk_{uuid.uuid4().hex[:8]}"
        filepath = os.path.join(self.storage_dir, f"{checkpoint_id}.json")

        payload = {
            "checkpoint_id": checkpoint_id,
            "hitl_request": hitl_request.model_dump(),
            "memory": mem.snapshot(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> Tuple[WorkingMemory, HITLRequest]:
        filepath = os.path.join(self.storage_dir, f"{checkpoint_id}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found at {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        req_data = data["hitl_request"]
        hitl_request = HITLRequest.model_validate(req_data)

        mem_snap = data["memory"]
        mem = WorkingMemory(goal=mem_snap["goal"])

        # Reconstruct subtasks
        mem.subtasks = [
            SubTask(
                id=t["id"],
                description=t["description"],
                status=t["status"],
                depends_on=t.get("depends_on", []),
                attempts=t.get("attempts", 0),
                result=t.get("result"),
            )
            for t in mem_snap.get("subtasks", [])
        ]

        # Reconstruct facts and logs
        mem.facts = mem_snap.get("facts", {})
        mem.summary_log = mem_snap.get("summary_log", [])
        mem.learned_constraints = mem_snap.get("learned_constraints", [])
        mem.recovery_log = mem_snap.get("recovery_log", [])
        mem.global_recovery_attempts = mem_snap.get("global_recovery_attempts", 0)

        # Reconstruct history
        mem.history = [
            StepRecord(
                step_num=r["step_num"],
                reasoning=r["reasoning"],
                action=r["action"],
                action_input=r["action_input"],
                action_result=r["action_result"],
                eval_verdict=r.get("eval_verdict"),
                eval_reasoning=r.get("eval_reasoning"),
                timestamp=r["timestamp"],
                subtask_id=r.get("subtask_id"),
            )
            for r in mem_snap.get("history", [])
        ]

        return mem, hitl_request

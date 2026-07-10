import pytest
from datetime import datetime, timezone
from agent.memory import WorkingMemory, SubTask, StepRecord

def test_working_memory_initialization():
    goal = "Answer a complex question"
    mem = WorkingMemory(goal=goal)
    assert mem.goal == goal
    assert len(mem.subtasks) == 0
    assert len(mem.history) == 0
    assert len(mem.summary_log) == 0

def test_working_memory_add_step_and_truncation():
    mem = WorkingMemory(goal="Test goal")
    
    # Add 4 steps
    for i in range(1, 5):
        record = StepRecord(
            step_num=i,
            reasoning=f"Reasoning for step {i}",
            action="mock_action",
            action_input={"param": i},
            action_result=f"Result of step {i}",
            eval_verdict="success",
            eval_reasoning="Looks good",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        mem.add_step(record)
        
    assert len(mem.history) == 4
    assert len(mem.summary_log) == 0
    
    # Add 5th step, triggering truncation of the first step
    record_5 = StepRecord(
        step_num=5,
        reasoning="Reasoning for step 5",
        action="mock_action",
        action_input={"param": 5},
        action_result="Result of step 5",
        eval_verdict="success",
        eval_reasoning="Looks good",
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    mem.add_step(record_5)
    
    # History should contain steps 2, 3, 4, 5
    assert len(mem.history) == 4
    assert mem.history[0].step_num == 2
    assert mem.history[3].step_num == 5
    
    # Summary log should contain exactly 1 entry corresponding to Step 1
    assert len(mem.summary_log) == 1
    assert "Step 1 (mock_action): Thought: Reasoning for step 1..." in mem.summary_log[0]
    assert "Result: Result of step 1 -> Verdict: success" in mem.summary_log[0]

def test_working_memory_snapshot():
    mem = WorkingMemory(goal="Snapshot test")
    task = SubTask(id="s1", description="Subtask 1", status="pending")
    mem.subtasks.append(task)
    
    record = StepRecord(
        step_num=1,
        reasoning="Reasoning 1",
        action="mock_action",
        action_input={"param": 1},
        action_result="Result 1",
        eval_verdict="success",
        eval_reasoning="Looks good",
        timestamp="2026-07-10T12:00:00Z"
    )
    mem.add_step(record)
    mem.facts["key_claim"] = "Fact value"
    
    snap = mem.snapshot()
    
    assert snap["goal"] == "Snapshot test"
    assert len(snap["subtasks"]) == 1
    assert snap["subtasks"][0]["id"] == "s1"
    assert snap["subtasks"][0]["status"] == "pending"
    assert snap["facts"] == {"key_claim": "Fact value"}
    assert len(snap["history"]) == 1
    assert snap["history"][0]["step_num"] == 1
    assert snap["history"][0]["reasoning"] == "Reasoning 1"
    assert snap["history"][0]["action"] == "mock_action"
    assert snap["history"][0]["action_input"] == {"param": 1}
    assert snap["history"][0]["action_result"] == "Result 1"
    assert snap["history"][0]["eval_verdict"] == "success"
    assert snap["history"][0]["eval_reasoning"] == "Looks good"
    assert snap["history"][0]["timestamp"] == "2026-07-10T12:00:00Z"

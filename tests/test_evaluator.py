import pytest
from unittest.mock import patch, MagicMock
from agent import evaluator
from agent import recovery
from agent.memory import WorkingMemory, SubTask, StepRecord

def test_evaluator_programmatic_tool_failure():
    # Programmatic detection of tool failure when 'error' key is present in result
    record = StepRecord(
        step_num=1,
        reasoning="Testing tool failure",
        action="fetch",
        action_input={"url": "https://example.com"},
        action_result={"error": "timeout", "detail": "Connection timed out"},
        eval_verdict=None,
        eval_reasoning=None,
        timestamp="2026-07-10T00:00:00Z"
    )
    verdict, reasoning = evaluator.evaluate_step(
        goal="Test goal",
        subtask_desc="Fetch page",
        record_data=record.__dict__,
        facts_summary={}
    )
    assert verdict == "tool_failure"
    assert "Programmatic detection" in reasoning

@patch("agent.llm.call_llm")
def test_evaluator_llm_verdicts(mock_call_llm):
    # Setup mocks for LLM evaluate response
    # 1. Success case
    # 2. Inconsistent case
    # 3. Goal drift case
    mock_responses = [
        '{"verdict": "success", "reasoning": "Output is correct"}',
        '{"verdict": "inconsistent", "reasoning": "Contradicts previous height facts"}',
        '{"verdict": "goal_drift", "reasoning": "Result has nothing to do with Eiffel tower"}'
    ]
    mock_call_llm.side_effect = lambda system, user, model, temperature=0.0: mock_responses.pop(0)
    
    record = StepRecord(
        step_num=1,
        reasoning="Testing success",
        action="fetch",
        action_input={"url": "https://example.com"},
        action_result={"text": "The height is 330 meters"},
        eval_verdict=None,
        eval_reasoning=None,
        timestamp="2026-07-10T00:00:00Z"
    )
    
    # 1. Test success
    verdict, reasoning = evaluator.evaluate_step("Test goal", "Fetch height", record.__dict__, {})
    assert verdict == "success"
    
    # 2. Test inconsistent
    verdict, reasoning = evaluator.evaluate_step("Test goal", "Fetch height", record.__dict__, {"height": "300 meters"})
    assert verdict == "inconsistent"
    
    # 3. Test goal drift
    verdict, reasoning = evaluator.evaluate_step("Test goal", "Fetch height", record.__dict__, {})
    assert verdict == "goal_drift"

def test_recovery_tool_failure():
    mem = WorkingMemory(goal="Test goal")
    subtask = SubTask(id="s1", description="Fetch page", status="in_progress")
    record = StepRecord(
        step_num=1,
        reasoning="Test",
        action="flaky_fetch",
        action_input={"url": "https://example.com"},
        action_result={"error": "timeout"},
        eval_verdict="tool_failure",
        eval_reasoning="Flaky fetch failed",
        timestamp="2026-07-10T00:00:00Z"
    )
    
    # Run recover_tool_failure (attempt 1)
    strategy, details, next_status = recovery.recover_tool_failure(mem, subtask, record, total_steps=1)
    assert strategy == "recover_tool_failure"
    assert subtask.attempts == 1
    assert subtask.status == "pending"
    assert mem.global_recovery_attempts == 1
    
    # Exhaust budget (attempts >= 3)
    subtask.attempts = 3
    strategy2, details2, next_status2 = recovery.recover_tool_failure(mem, subtask, record, total_steps=1)
    assert strategy2 == "tool_failure_exhausted"
    assert subtask.status == "unresolvable"

@patch("agent.llm.call_llm")
def test_recovery_inconsistency(mock_call_llm):
    # LLM resolver returns needs_verification
    mock_call_llm.return_value = '{"status": "needs_verification", "resolution": "conflict discovered", "verification_query": "official height site"}'
    
    mem = WorkingMemory(goal="Test goal")
    subtask = SubTask(id="s1", description="Fetch page", status="in_progress")
    record = StepRecord(
        step_num=1,
        reasoning="Test",
        action="fetch",
        action_input={"url": "https://example.com"},
        action_result={"text": "contradictory info"},
        eval_verdict="inconsistent",
        eval_reasoning="Conflict found",
        timestamp="2026-07-10T00:00:00Z"
    )
    
    strategy, details, next_status = recovery.recover_inconsistency(mem, subtask, record, total_steps=1)
    assert strategy == "verify_contradiction"
    assert subtask.status == "pending"
    assert "Verify conflict:" in subtask.description
    assert subtask.attempts == 1

@patch("agent.llm.call_llm")
def test_recovery_goal_drift(mock_call_llm):
    # LLM replan returns a new subtask
    mock_call_llm.return_value = '{"subtasks": [{"id": "r1", "description": "New search for Eiffel Tower site", "status": "pending"}]}'
    
    mem = WorkingMemory(goal="Test goal")
    subtask_1 = SubTask(id="s1", description="Fetch page", status="in_progress")
    subtask_2 = SubTask(id="s2", description="Other page", status="pending")
    mem.subtasks = [subtask_1, subtask_2]
    
    record = StepRecord(
        step_num=1,
        reasoning="Test",
        action="fetch",
        action_input={"url": "https://example.com"},
        action_result={"text": "drifted info"},
        eval_verdict="goal_drift",
        eval_reasoning="Drifted from original goal",
        timestamp="2026-07-10T00:00:00Z"
    )
    
    strategy, details, next_status = recovery.recover_goal_drift(mem, subtask_1, record, total_steps=1)
    assert strategy == "replan_goal"
    assert subtask_1.status == "done"
    
    # Verify s2 (subsequent pending task) was sliced out and r1 was appended
    assert len(mem.subtasks) == 2
    assert mem.subtasks[0].id == "s1"
    assert mem.subtasks[1].id == "r1"

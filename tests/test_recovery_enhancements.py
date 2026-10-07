import pytest
from unittest.mock import patch, MagicMock
from agent import evaluator
from agent.memory import WorkingMemory, SubTask, StepRecord
from agent import recovery
from agent.tools import ToolRouter, SearchInput, FetchInput, PaymentInput, DeleteInput
from agent import orchestrator


def test_programmatic_failures_bypass_llm_evaluator():
    """Verify that Stage 1 programmatic failures bypass the LLM evaluator call."""
    record = StepRecord(
        step_num=1,
        reasoning="Test programmatic check failure",
        action="fetch",
        action_input={"url": "https://example.com"},
        action_result={"error": "timeout", "detail": "Connection timed out"},
        eval_verdict=None,
        eval_reasoning=None,
        timestamp="2026-10-07T00:00:00Z",
        subtask_id="s1"
    )

    with patch("agent.llm.evaluate") as mock_llm_eval:
        verdict, reasoning = evaluator.evaluate_step(
            goal="Fetch data",
            subtask_desc="Fetch webpage",
            record_data=record.__dict__,
            facts_summary={}
        )
        assert verdict == "tool_failure"
        assert "Programmatic detection" in reasoning
        # Assert LLM evaluate was NOT called because Stage 1 failed
        mock_llm_eval.assert_not_called()


def test_context_compaction_reduces_record_count():
    """Verify that WorkingMemory.compact_subtask collapses failed intermediate attempts."""
    mem = WorkingMemory(goal="Test context compaction")
    subtask_id = "sub_1"

    step1 = StepRecord(
        step_num=1,
        reasoning="Attempt 1: query failed",
        action="search",
        action_input={"query": "bad query"},
        action_result={"error": "invalid_input"},
        eval_verdict="tool_failure",
        eval_reasoning="Failed",
        timestamp="2026-10-07T00:00:00Z",
        subtask_id=subtask_id
    )
    step2 = StepRecord(
        step_num=2,
        reasoning="Attempt 2: flaky fetch failed",
        action="flaky_fetch",
        action_input={"url": "https://badurl.com"},
        action_result={"text": "Error 403 Access Denied"},
        eval_verdict="tool_failure",
        eval_reasoning="Failed",
        timestamp="2026-10-07T00:01:00Z",
        subtask_id=subtask_id
    )
    step3 = StepRecord(
        step_num=3,
        reasoning="Attempt 3: successful fetch",
        action="fetch",
        action_input={"url": "https://goodurl.com"},
        action_result={"url": "https://goodurl.com", "text": "Valid response body with sufficient content length."},
        eval_verdict="success",
        eval_reasoning="Stage 1 & 2 passed",
        timestamp="2026-10-07T00:02:00Z",
        subtask_id=subtask_id
    )

    mem.history.extend([step1, step2, step3])
    assert len(mem.history) == 3

    mem.compact_subtask(subtask_id, strategy="self_correction")

    # Verify history was compacted into 1 consolidated StepRecord
    assert len(mem.history) == 1
    compacted_record = mem.history[0]
    assert "resolved via [self_correction]" in compacted_record.reasoning
    assert compacted_record.eval_verdict == "success"
    assert len(mem.summary_log) == 1
    assert "resolved via [self_correction]" in mem.summary_log[0]


def test_negative_constraints_in_recovery():
    """Verify that recovery loop injects explicit negative constraint blocks into memory."""
    mem = WorkingMemory(goal="Test negative constraint")
    subtask = SubTask(id="s1", description="Fetch page", status="in_progress")
    failed_record = StepRecord(
        step_num=1,
        reasoning="Attempting fetch",
        action="fetch",
        action_input={"url": "https://badurl.com"},
        action_result={"error": "timeout", "detail": "Connection timed out"},
        eval_verdict="tool_failure",
        eval_reasoning="Programmatic detection [timeout]: Connection timed out",
        timestamp="2026-10-07T00:00:00Z",
        subtask_id="s1"
    )

    strategy, details, status = recovery.recover_tool_failure(mem, subtask, failed_record, total_steps=1)

    assert status == "pending"
    assert "EXPLICIT FORBIDDEN ACTION" in details
    assert "badurl.com" in details
    assert len(mem.learned_constraints) >= 1
    assert any("EXPLICIT FORBIDDEN ACTION" in constraint for constraint in mem.learned_constraints)


def test_learned_constraints_persist_across_subtasks():
    """Verify that learned constraints persist in memory snapshot across sequential subtasks."""
    mem = WorkingMemory(goal="Test constraint persistence")
    subtask1 = SubTask(id="s1", description="Search item", status="done", attempts=1)
    subtask2 = SubTask(id="s2", description="Fetch item details", status="pending")
    mem.subtasks = [subtask1, subtask2]

    invariant = "Subtask [Search item] invariant: Must use query format YYYY-MM-DD"
    mem.learned_constraints.append(invariant)

    snap = mem.snapshot()
    assert "learned_constraints" in snap
    assert invariant in snap["learned_constraints"]


def test_tool_dry_run_safety():
    """Verify that dry_run=True mode validates schemas without side effects."""
    router = ToolRouter(seed=42)

    # Test payment tool in dry_run mode
    payment_res = router.call("payment", {
        "amount": 100.0,
        "currency": "USD",
        "recipient": "vendor@example.com",
        "dry_run": True
    })

    assert payment_res.get("dry_run") is True
    assert payment_res.get("status") == "validated"
    assert payment_res.get("tool") == "payment"
    assert payment_res.get("parameters")["amount"] == 100.0

    # Test invalid payment input
    invalid_res = router.call("payment", {
        "amount": "not_a_number",
        "dry_run": True
    })
    assert invalid_res.get("error") == "invalid_input"

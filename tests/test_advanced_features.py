import os
import json
import pytest
from agent.sandbox import PythonCodeSandbox, SandboxExecutionResult
from agent.tools import ToolRouter
from agent.critic import AdversarialCritic, CritiqueResult
from agent.hitl import HITLRequest, CheckpointManager
from agent.episodic_memory import EpisodicMemoryStore, EpisodicRecord
from agent import planner
from agent import validation
from agent import evaluator
from agent import orchestrator


def test_python_code_sandbox_execution_and_healing():
    sandbox = PythonCodeSandbox(default_timeout=2.0)

    # 1. Successful execution
    res1 = sandbox.execute_code("print('Hello World')")
    assert res1.success is True
    assert "Hello World" in res1.stdout
    assert res1.returncode == 0

    # 2. Syntax/Runtime error execution
    res2 = sandbox.execute_code("x = 1 / 0")
    assert res2.success is False
    assert res2.error_type in ("ZeroDivisionError", "Exception")
    assert "ZeroDivisionError" in res2.stderr

    # 3. Timeout execution
    res3 = sandbox.execute_code("import time; time.sleep(10.0)", timeout=0.5)
    assert res3.success is False
    assert res3.error_type == "TIMEOUT"
    assert "timed out" in res3.stderr

    # 4. ToolRouter integration
    tr = ToolRouter(seed=123, force_mocks=True)
    res_tool = tr.call("execute_python_script", {"code": "print(2 + 2)"})
    assert res_tool.get("success") is True
    assert res_tool.get("result_value") == 4


def test_adversarial_critic():
    critic = AdversarialCritic()
    facts = {"capital": "Paris"}

    # Ungrounded claim check
    critique = critic.critique(
        goal="Find capital of France",
        subtask_desc="Search capital",
        action_result="Contains unverified or ungrounded assertions about Atlantis being capital.",
        facts=facts,
    )
    assert critique.is_valid is False
    assert len(critique.ungrounded_claims) > 0

    # Evaluator integration with enable_critic
    record_data = {
        "action": "search",
        "action_input": {"query": "test"},
        "action_result": {"results": [{"url": "http://test.com", "title": "Test"}]},
    }
    verdict, reasoning = evaluator.evaluate_step(
        goal="Test goal",
        subtask_desc="Test subtask",
        record_data=record_data,
        facts_summary=facts,
        enable_critic=True,
    )
    # Valid deterministic search result passes
    assert verdict in ("success", "inconsistent")


def test_hitl_checkpoint_and_resumption(tmp_path):
    storage_dir = str(tmp_path / "checkpoints")
    ckpt_mgr = CheckpointManager(storage_dir=storage_dir)

    tools = ToolRouter(seed=123, force_mocks=True)
    
    # Run orchestrator with require_hitl=True for a side-effecting payment action
    # We force a subtask that invokes payment
    mem, result = orchestrator.run("Send payment to vendor", tools, use_self_correction=True, require_hitl=True, checkpoint_dir=storage_dir)

    # Check that HITL checkpoint was created or awaiting input returned
    assert result == "AWAITING_HUMAN_INPUT" or mem.subtasks[0].status in ("awaiting_input", "done")

    # Verify saving and loading checkpoints
    hitl_req = HITLRequest(
        checkpoint_id="test_chk_123",
        subtask_id="s1",
        question="Approve payment of $100?",
        options=["Approve", "Reject"],
        context_summary="Payment test",
    )
    saved_id = ckpt_mgr.save_checkpoint(mem, hitl_req)
    assert saved_id == "test_chk_123"

    loaded_mem, loaded_req = ckpt_mgr.load_checkpoint("test_chk_123")
    assert loaded_req.question == "Approve payment of $100?"
    assert loaded_mem.goal == mem.goal

    # Test resumption
    res_mem, res_output = orchestrator.resume_with_input("test_chk_123", "Approved by admin", tools, checkpoint_dir=storage_dir)
    assert res_mem.facts.get("hitl_s1") == "Approved by admin"
    assert res_output != "AWAITING_HUMAN_INPUT"


def test_episodic_memory_store(tmp_path):
    store_path = str(tmp_path / "episodic.json")
    store = EpisodicMemoryStore(storage_path=store_path)

    # Record resolution
    rec = store.record_resolution(
        goal_pattern="Fetch Eiffel Tower height",
        error_type="HTTP_ERROR",
        failed_action="flaky_fetch",
        successful_resolution="Fetched content using standard fetch with timeout 10s",
        learned_rule="Use standard fetch when flaky_fetch yields 403",
    )
    assert rec.error_type == "HTTP_ERROR"

    # Query similar
    results = store.query_similar("Eiffel Tower fetch 403 error", top_k=2)
    assert len(results) > 0
    assert results[0].failed_action == "flaky_fetch"


def test_task_complexity_and_confidence_calibration():
    # Complexity estimation
    comp_low = planner.estimate_complexity("What is 2+2?")
    assert comp_low in ("LOW", "MEDIUM")

    comp_high = planner.estimate_complexity("Write python code script to compute matrix transform and execute")
    assert comp_high == "HIGH"

    b_high = planner.get_budget_for_complexity("HIGH")
    assert b_high["max_subtask_retries"] == 3
    assert b_high["enable_critic"] is True

    # Validation confidence score calibration
    finding_ok = validation.validate_result(
        goal="test",
        subtask_desc="test",
        action="search",
        result={"results": [{"url": "http://a.com"}, {"url": "http://b.com"}]},
    )
    assert finding_ok.confidence_score == 1.0
    assert len(finding_ok.uncertainty_flags) == 0

    finding_short = validation.validate_result(
        goal="test",
        subtask_desc="test",
        action="fetch",
        result={"text": "Short 15 char text"},
    )
    assert finding_short.confidence_score < 1.0
    assert "short_content" in finding_short.uncertainty_flags

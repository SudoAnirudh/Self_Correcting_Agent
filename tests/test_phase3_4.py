import os
import json
import pytest
from agent.memory import WorkingMemory, SubTask
from eval.chaos_matrix import ChaosToolRouter, ChaosConfig, ChaosFaultType
from agent.logger import AgentLogger, TraceSpan


def test_dag_subtask_ready_resolution():
    """Verify that get_ready_subtasks returns subtasks whose dependencies are completed."""
    mem = WorkingMemory(goal="Test DAG Scheduling")

    s1 = SubTask(id="s1", description="Search topic", status="pending", depends_on=[])
    s2 = SubTask(id="s2", description="Fetch details of s1", status="pending", depends_on=["s1"])
    s3 = SubTask(id="s3", description="Independent search", status="pending", depends_on=[])
    s4 = SubTask(id="s4", description="Synthesize s2 and s3", status="pending", depends_on=["s2", "s3"])

    mem.subtasks = [s1, s2, s3, s4]

    # Initial ready subtasks: s1 and s3 (depends_on: [])
    ready = mem.get_ready_subtasks()
    ready_ids = {t.id for t in ready}
    assert ready_ids == {"s1", "s3"}

    # Mark s1 done
    s1.status = "done"
    ready = mem.get_ready_subtasks()
    ready_ids = {t.id for t in ready}
    assert ready_ids == {"s2", "s3"}

    # Mark s3 done
    s3.status = "done"
    ready = mem.get_ready_subtasks()
    ready_ids = {t.id for t in ready}
    assert ready_ids == {"s2"}

    # Mark s2 done -> s4 becomes ready!
    s2.status = "done"
    ready = mem.get_ready_subtasks()
    ready_ids = {t.id for t in ready}
    assert ready_ids == {"s4"}


def test_dag_mark_branch_blocked():
    """Verify that mark_branch_blocked transitively blocks downstream dependent tasks while leaving independent tasks ready."""
    mem = WorkingMemory(goal="Test Branch Blocking")

    s1 = SubTask(id="s1", description="Search topic A", status="unresolvable", depends_on=[])
    s2 = SubTask(id="s2", description="Fetch topic A details", status="pending", depends_on=["s1"])
    s3 = SubTask(id="s3", description="Summarize topic A", status="pending", depends_on=["s2"])
    s4 = SubTask(id="s4", description="Independent search topic B", status="pending", depends_on=[])

    mem.subtasks = [s1, s2, s3, s4]

    mem.mark_branch_blocked("s1")

    assert s2.status == "blocked"
    assert s3.status == "blocked"
    assert s4.status in ("pending", "ready")

    ready = mem.get_ready_subtasks()
    assert len(ready) == 1
    assert ready[0].id == "s4"


def test_chaos_tool_router_fault_injection():
    """Verify that ChaosToolRouter injects faults according to chaos_rate."""
    config = ChaosConfig(chaos_rate=1.0, seed=42) # 100% chaos rate for testing
    router = ChaosToolRouter(seed=42, force_mocks=True, chaos_config=config)

    res = router.call("search", {"query": "test query"})
    assert router.injected_faults_count > 0
    assert sum(router.fault_breakdown.values()) > 0


def test_openinference_tracing_export():
    """Verify that AgentLogger emits OpenInference/OpenTelemetry compliant TraceSpan records."""
    logger = AgentLogger(goal="Test Tracing", mode="self_correcting")

    span = logger.emit_span(
        name="agent.tool_call",
        start_time=logger._now(),
        end_time=logger._now(),
        attributes={"tool_name": "search", "step_num": 1, "status": "success"}
    )

    assert isinstance(span, TraceSpan)
    assert span.name == "agent.tool_call"
    assert span.attributes["tool_name"] == "search"
    assert os.path.exists("logs/traces.jsonl")

    # Read last line from traces.jsonl
    with open("logs/traces.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
        last_trace = json.loads(lines[-1])
        assert last_trace["name"] == "agent.tool_call"
        assert "trace_id" in last_trace
        assert "span_id" in last_trace

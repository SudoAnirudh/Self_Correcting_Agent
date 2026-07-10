import os
import json
import pytest
from datetime import datetime, timezone
from agent.logger import AgentLogger
from agent.memory import StepRecord

def test_agent_logger_creates_file_and_logs_events():
    goal = "Test the logger functionality!"
    mode = "test_mode"
    logger = AgentLogger(goal=goal, mode=mode)
    
    assert os.path.exists(logger.filename)
    
    # Log run_start
    models_info = {"reason": {"model": "llama3", "temp": 0.0}}
    logger.log_run_start(goal, 42, mode, models_info)
    
    # Log plan
    subtasks = [{"id": "s1", "description": "Step 1", "status": "pending"}]
    logger.log_plan(subtasks)
    
    # Log step
    record = StepRecord(
        step_num=1,
        reasoning="Thinking",
        action="search",
        action_input={"query": "test"},
        action_result={"success": True},
        eval_verdict="success",
        eval_reasoning="Good",
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    logger.log_step(record)
    
    # Log recovery
    logger.log_recovery("s1", "fallback_source", "retry 1")
    
    # Log run_end
    logger.log_run_end("Final answer", [], 1, 1)
    
    # Read the log file and verify contents
    with open(logger.filename, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f]
        
    assert len(lines) == 6
    assert lines[0]["type"] == "init"
    assert lines[0]["goal"] == goal
    
    assert lines[1]["type"] == "run_start"
    assert lines[1]["seed"] == 42
    assert lines[1]["models_info"] == models_info
    
    assert lines[2]["type"] == "plan"
    assert lines[2]["subtasks"] == subtasks
    
    assert lines[3]["type"] == "step"
    assert lines[3]["step_num"] == 1
    assert lines[3]["action"] == "search"
    assert lines[3]["result"] == {"success": True}
    
    assert lines[4]["type"] == "recovery"
    assert lines[4]["subtask"] == "s1"
    assert lines[4]["strategy"] == "fallback_source"
    assert lines[4]["detail"] == "retry 1"
    
    assert lines[5]["type"] == "run_end"
    assert lines[5]["final_output"] == "Final answer"
    
    # Clean up test log file
    os.remove(logger.filename)

import json
from typing import List
from agent.memory import SubTask, WorkingMemory, StepRecord
from agent import llm

def decompose(goal: str) -> List[SubTask]:
    """Decompose the goal into subtasks."""
    data = llm.decompose(goal)
    subtasks = []
    # If the LLM failed or returned no subtasks, create a fallback plan
    raw_subtasks = data.get("subtasks", [])
    if not raw_subtasks:
        raw_subtasks = [
            {"id": "s1", "description": f"Perform initial search on: {goal}", "status": "pending"},
            {"id": "s2", "description": "Synthesize final response from findings", "status": "pending"}
        ]
        
    for item in raw_subtasks:
        subtasks.append(SubTask(
            id=item["id"],
            description=item["description"],
            status=item.get("status", "pending")
        ))
    return subtasks

def replan(mem: WorkingMemory, failed_subtask: SubTask, record: StepRecord) -> List[SubTask]:
    """Replan when goal drift is detected, returning a revised list of subtasks."""
    # We will invoke the LLM to get a new list of subtasks, providing the current memory snapshot
    system = (
        "You are an expert agent planner. The agent has drifted from its original goal. "
        "Review the original goal, the completed subtasks, the facts gathered so far, and the failed step. "
        "Create a revised list of remaining subtasks to safely guide the agent back to the goal. "
        "Provide JSON ONLY in the following format:\n"
        "{\n"
        '  "subtasks": [\n'
        '    {"id": "r1", "description": "New search for...", "status": "pending"},\n'
        '    {"id": "r2", "description": "Synthesize the data...", "status": "pending"}\n'
        '  ]\n'
        "}"
    )
    
    user = (
        f"Original Goal: {mem.goal}\n"
        f"Failed Subtask: {failed_subtask.description} (attempts: {failed_subtask.attempts})\n"
        f"Failed Step Record: Reasoning: '{record.reasoning}', Action: '{record.action}', Result: '{str(record.action_result)[:300]}'\n"
        f"Gathered Facts: {mem.facts}\n"
        f"Summary Log: {mem.summary_log}\n"
    )
    
    try:
        res = llm.call_llm(system, user, llm.REASONING_MODEL, temperature=0.0)
        data = json.loads(res)
        raw_subtasks = data.get("subtasks", [])
    except Exception:
        raw_subtasks = []
        
    new_subtasks = []
    if raw_subtasks:
        for item in raw_subtasks:
            new_subtasks.append(SubTask(
                id=item["id"],
                description=item["description"],
                status=item.get("status", "pending")
            ))
    else:
        # Fallback if parsing or call fails: keep the failed subtask but reset status, and add a search task
        new_subtasks = [
            SubTask(id="r1", description=f"Search for fallback alternative for {failed_subtask.description}", status="pending"),
            SubTask(id="r2", description="Synthesize available details", status="pending")
        ]
    return new_subtasks

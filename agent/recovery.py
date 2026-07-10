from typing import Tuple, List
from agent.memory import WorkingMemory, SubTask, StepRecord
from agent import planner
from agent import llm

MAX_RECOVERY_PER_SUBTASK = 3

def check_budget(mem: WorkingMemory, subtask: SubTask, total_steps: int) -> bool:
    """Returns True if recovery is allowed under the budget caps, False otherwise."""
    if subtask.attempts >= MAX_RECOVERY_PER_SUBTASK:
        return False
        
    # Global soft cap: recovery attempts <= 40% of total steps
    # We only apply this after a few steps (e.g. 5) to avoid locking out early recoveries
    if total_steps > 5:
        global_ratio = mem.global_recovery_attempts / total_steps
        if global_ratio > 0.40:
            return False
            
    return True

def recover_tool_failure(
    mem: WorkingMemory, subtask: SubTask, record: StepRecord, total_steps: int
) -> Tuple[str, str, str]:
    """
    Recover from tool errors or flaky fetch problems.
    Resets subtask to pending and increments attempts, or marks unresolvable if out of budget.
    """
    if not check_budget(mem, subtask, total_steps):
        subtask.status = "unresolvable"
        subtask.result = f"Failed to resolve after {subtask.attempts} attempts due to tool failure."
        return "tool_failure_exhausted", "Attempts budget exceeded", "unresolvable"
        
    # Increment recovery counters
    subtask.attempts += 1
    mem.global_recovery_attempts += 1
    
    # Reset subtask to pending for retry
    subtask.status = "pending"
    
    details = f"Retrying subtask. Attempt {subtask.attempts}."
    if record.action == "flaky_fetch":
        details += " Hint: flaky_fetch failed, prompting fallback to standard fetch."
        
    return "recover_tool_failure", details, "pending"

def recover_inconsistency(
    mem: WorkingMemory, subtask: SubTask, record: StepRecord, total_steps: int
) -> Tuple[str, str, str]:
    """
    Recover from conflicting facts by validating them with the LLM contradiction resolver.
    """
    if not check_budget(mem, subtask, total_steps):
        subtask.status = "unresolvable"
        subtask.result = f"Inconsistent facts could not be resolved after {subtask.attempts} attempts."
        return "inconsistency_exhausted", "Attempts budget exceeded", "unresolvable"
        
    subtask.attempts += 1
    mem.global_recovery_attempts += 1
    
    # Run the contradiction resolver
    resolution_data = llm.resolve_inconsistency(mem.goal, mem.facts, record.action_result)
    status = resolution_data.get("status", "needs_verification")
    resolution = resolution_data.get("resolution", "No resolution provided.")
    verification_query = resolution_data.get("verification_query", "")
    
    if status == "resolved":
        # Contradiction is resolved programmatically; update facts and mark subtask done
        mem.facts["resolution_note"] = resolution
        subtask.status = "done"
        subtask.result = f"Resolved contradiction: {resolution}"
        return "resolve_contradiction", f"Contradiction resolved: {resolution}", "done"
    else:
        # Needs further verification: keep subtask pending but rewrite its description to focus on verification query
        subtask.status = "pending"
        subtask.description = f"Verify conflict: {verification_query if verification_query else subtask.description}"
        return "verify_contradiction", f"Verification scheduled: {subtask.description}", "pending"

def recover_goal_drift(
    mem: WorkingMemory, subtask: SubTask, record: StepRecord, total_steps: int
) -> Tuple[str, str, str]:
    """
    Recover from goal drift by invoking the planner to rewrite remaining subtasks.
    """
    if not check_budget(mem, subtask, total_steps):
        subtask.status = "unresolvable"
        subtask.result = f"Goal drift detected and could not be recovered after {subtask.attempts} attempts."
        return "goal_drift_exhausted", "Attempts budget exceeded", "unresolvable"
        
    subtask.attempts += 1
    mem.global_recovery_attempts += 1
    
    # Generate new subtasks via planner
    new_tasks = planner.replan(mem, subtask, record)
    
    # Update subtasks list: slice off all tasks after the current one, and append new ones
    try:
        idx = mem.subtasks.index(subtask)
        mem.subtasks = mem.subtasks[:idx+1]
    except ValueError:
        # If current subtask is not in list for some reason, just clear and re-add
        mem.subtasks = [subtask]
        idx = 0
        
    # Mark current subtask as done since we replanned around it
    subtask.status = "done"
    subtask.result = f"Triggered replan due to drift: {record.reasoning}"
    
    mem.subtasks.extend(new_tasks)
    
    desc_list = [t.description for t in new_tasks]
    return "replan_goal", f"Re-planned remaining tasks. Injected: {desc_list}", "done"

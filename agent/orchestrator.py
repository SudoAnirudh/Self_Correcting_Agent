import time
import random
from datetime import datetime, timezone
from typing import Tuple, List, Optional
from agent.memory import WorkingMemory, SubTask, StepRecord
from agent.logger import AgentLogger
from agent.tools import ToolRouter
from agent import llm
from agent import planner

def pick_next_subtask(mem: WorkingMemory) -> Optional[SubTask]:
    for t in mem.subtasks:
        if t.status in ("pending", "in_progress"):
            return t
    return None

def run(goal: str, tools: ToolRouter, use_self_correction: bool = True) -> Tuple[WorkingMemory, str]:
    mode = "self_correcting" if use_self_correction else "baseline"
    logger = AgentLogger(goal=goal, mode=mode)
    
    mem = WorkingMemory(goal=goal)
    
    # 1. Planner Decompose
    subtasks = planner.decompose(goal)
    mem.subtasks = subtasks
    
    # Log the initial plan
    logger.log_plan([
        {"id": t.id, "description": t.description, "status": t.status} 
        for t in mem.subtasks
    ])
    
    # Log run start
    models_info = {
        "reasoning": {"model": llm.REASONING_MODEL, "temperature": 0.0},
        "evaluation": {"model": llm.EVALUATION_MODEL, "temperature": 0.0}
    }
    logger.log_run_start(goal, tools.seed, mode, models_info)
    
    step_num = 0
    MAX_STEPS = 25
    
    while step_num < MAX_STEPS:
        subtask = pick_next_subtask(mem)
        if not subtask:
            break
            
        step_num += 1
        subtask.status = "in_progress"
        
        # 2. LLM reasoning call (Thought)
        subtask_snap = {
            "id": subtask.id,
            "description": subtask.description,
            "status": subtask.status,
            "attempts": subtask.attempts,
            "result": subtask.result
        }
        thought_data = llm.reason(mem.snapshot(), subtask_snap)
        
        action_name = thought_data.get("action")
        action_input = thought_data.get("action_input", {})
        
        reasoning_str = (
            f"Belief: {thought_data.get('belief', '')} | "
            f"Gap: {thought_data.get('gap', '')} | "
            f"Why Action: {thought_data.get('why_action', '')}"
        )
        
        # 3. Dispatch action via ToolRouter (Action & Observation)
        result = tools.call(action_name, action_input)
        
        record = StepRecord(
            step_num=step_num,
            reasoning=reasoning_str,
            action=action_name,
            action_input=action_input,
            action_result=result,
            eval_verdict=None,
            eval_reasoning=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # 4. Self-Correction / Evaluator Loop
        if use_self_correction:
            from agent import evaluator
            from agent import recovery
            from dataclasses import asdict
            
            verdict, eval_reasoning = evaluator.evaluate_step(
                goal, subtask.description, asdict(record), mem.facts
            )
            
            record.eval_verdict = verdict
            record.eval_reasoning = eval_reasoning
            
            if verdict == "success":
                subtask.status = "done"
                subtask.result = str(result)
                # Extract facts
                facts = llm.extract_facts(goal, subtask.description, result)
                for k, v in facts.items():
                    mem.facts[k] = v
            else:
                # Trigger specific recovery strategy
                if verdict == "tool_failure":
                    strategy, details, next_status = recovery.recover_tool_failure(
                        mem, subtask, record, step_num
                    )
                elif verdict == "inconsistent":
                    strategy, details, next_status = recovery.recover_inconsistency(
                        mem, subtask, record, step_num
                    )
                else: # goal_drift
                    strategy, details, next_status = recovery.recover_goal_drift(
                        mem, subtask, record, step_num
                    )
                
                # Log recovery event
                logger.log_recovery(subtask.id, strategy, details)
        else:
            # BASELINE MODE: mark done regardless, extract facts directly
            subtask.status = "done"
            subtask.result = str(result)
            
            # Extract facts from result and merge
            facts = llm.extract_facts(goal, subtask.description, result)
            for k, v in facts.items():
                mem.facts[k] = v
                
        mem.add_step(record)
        logger.log_step(record)
        
    # 5. Synthesis phase
    unresolved = [
        {
            "id": t.id, 
            "description": t.description, 
            "status": t.status, 
            "attempts": t.attempts, 
            "result": t.result
        } 
        for t in mem.subtasks if t.status not in ("done", "unresolvable")
    ]
    
    final_output = llm.synthesize(goal, mem.facts, mem.summary_log, unresolved)
    
    # Log run end
    logger.log_run_end(
        final_output=final_output,
        unresolved_subtasks=unresolved,
        total_steps=step_num,
        total_recoveries=mem.global_recovery_attempts
    )
    
    return mem, final_output

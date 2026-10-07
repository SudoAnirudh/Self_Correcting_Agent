import time
import random
import uuid
from datetime import datetime, timezone
from typing import Tuple, List, Optional, Any
from agent.memory import WorkingMemory, SubTask, StepRecord
from agent.logger import AgentLogger
from agent.tools import ToolRouter
from agent import llm
from agent import planner
from agent.hitl import HITLRequest, CheckpointManager
from agent.episodic_memory import EpisodicMemoryStore
from agent.validation import SIDE_EFFECTING_TOOLS


def pick_next_subtask(mem: WorkingMemory) -> Optional[SubTask]:
    ready = mem.get_ready_subtasks()
    if ready:
        return ready[0]
    for t in mem.subtasks:
        if t.status in ("pending", "ready", "in_progress"):
            return t
    return None


def run(
    goal: str,
    tools: ToolRouter,
    use_self_correction: bool = True,
    require_hitl: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[WorkingMemory, str]:
    mode = "self_correcting" if use_self_correction else "baseline"
    logger = AgentLogger(goal=goal, mode=mode)
    
    mem = WorkingMemory(goal=goal)
    
    # 1. Planner Decompose & Complexity Budgeting
    subtasks = planner.decompose(goal)
    mem.subtasks = subtasks
    complexity = planner.estimate_complexity(goal)
    budget = planner.get_budget_for_complexity(complexity)
    max_steps = budget.get("max_total_steps", 25)
    enable_critic = budget.get("enable_critic", False)
    
    # Log the initial plan
    logger.log_plan([
        {"id": t.id, "description": t.description, "status": t.status, "depends_on": t.depends_on} 
        for t in mem.subtasks
    ])
    
    # Log run start
    models_info = {
        "reasoning": {"model": llm.REASONING_MODEL, "temperature": 0.0},
        "evaluation": {"model": llm.EVALUATION_MODEL, "temperature": 0.0},
        "complexity": complexity,
        "budget": budget,
    }
    logger.log_run_start(goal, tools.seed, mode, models_info)
    
    step_num = 0
    ckpt_mgr = CheckpointManager(storage_dir=checkpoint_dir) if checkpoint_dir else CheckpointManager()
    episodic_store = EpisodicMemoryStore()
    
    while step_num < max_steps:
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
            "depends_on": subtask.depends_on,
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

        # HITL Check for ambiguous or high-risk side-effecting operations
        if require_hitl and action_name in SIDE_EFFECTING_TOOLS:
            chk_id = f"chk_{uuid.uuid4().hex[:8]}"
            hitl_req = HITLRequest(
                checkpoint_id=chk_id,
                subtask_id=subtask.id,
                question=f"Approve execution of side-effecting action '{action_name}' with parameters: {action_input}?",
                options=["Approve", "Reject"],
                context_summary=f"Goal: {goal} | Subtask: {subtask.description}",
                severity="high",
            )
            subtask.status = "awaiting_input"
            ckpt_mgr.save_checkpoint(mem, hitl_req)
            logger.log_step(StepRecord(
                step_num=step_num,
                reasoning="Awaiting human confirmation for side-effecting action.",
                action=action_name,
                action_input=action_input,
                action_result={"status": "awaiting_input"},
                eval_verdict=None,
                eval_reasoning="HITL triggered",
                timestamp=datetime.now(timezone.utc).isoformat(),
                subtask_id=subtask.id
            ))
            return mem, "AWAITING_HUMAN_INPUT"
        
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
            timestamp=datetime.now(timezone.utc).isoformat(),
            subtask_id=subtask.id
        )
        
        # 4. Self-Correction / Evaluator Loop
        if use_self_correction:
            from agent import evaluator
            from agent import recovery
            from dataclasses import asdict
            
            verdict, eval_reasoning = evaluator.evaluate_step(
                goal, subtask.description, asdict(record), mem.facts,
                enable_critic=enable_critic, complexity=complexity
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

                # If subtask required recovery attempts, compact history and extract learned constraints & episodic memory
                if subtask.attempts > 0:
                    mem.compact_subtask(subtask.id, strategy="self_correction")
                    invariant = f"Subtask [{subtask.description}] invariant: Resolved via {action_name} with params {action_input}"
                    if invariant not in mem.learned_constraints:
                        mem.learned_constraints.append(invariant)
                    
                    # Record resolution episode in cross-session store
                    episodic_store.record_resolution(
                        goal_pattern=goal,
                        error_type=record.eval_verdict or "tool_failure",
                        failed_action=action_name,
                        successful_resolution=str(result),
                        learned_rule=invariant,
                    )
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
                
                # If recovery is exhausted or failed permanently, mark dependent branch blocked or trigger HITL
                if subtask.status in ("unresolvable", "failed"):
                    mem.mark_branch_blocked(subtask.id)
                    if require_hitl:
                        chk_id = f"chk_{uuid.uuid4().hex[:8]}"
                        hitl_req = HITLRequest(
                            checkpoint_id=chk_id,
                            subtask_id=subtask.id,
                            question=f"Subtask [{subtask.id}] failed after {subtask.attempts} attempts. Provide resolution?",
                            options=["Retry with user guidance", "Skip subtask"],
                            context_summary=f"Goal: {goal} | Reason: {details}",
                            severity="medium",
                        )
                        ckpt_mgr.save_checkpoint(mem, hitl_req)
                        mem.add_step(record)
                        logger.log_step(record)
                        return mem, "AWAITING_HUMAN_INPUT"

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


def resume_with_input(
    checkpoint_id: str,
    user_response: str,
    tools: ToolRouter,
    use_self_correction: bool = True,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[WorkingMemory, str]:
    """Resumes agent execution from a saved checkpoint after receiving human input."""
    ckpt_mgr = CheckpointManager(storage_dir=checkpoint_dir) if checkpoint_dir else CheckpointManager()
    mem, req = ckpt_mgr.load_checkpoint(checkpoint_id)

    # Find the target subtask awaiting input
    subtask = next((t for t in mem.subtasks if t.id == req.subtask_id), None)
    if subtask:
        subtask.status = "done"
        subtask.result = f"Resolved via Human-in-the-Loop input: {user_response}"
        mem.facts[f"hitl_{subtask.id}"] = user_response
        mem.summary_log.append(f"HITL Resolved Subtask [{subtask.id}]: {user_response}")

    # Resume the loop directly from the updated memory state
    mode = "self_correcting" if use_self_correction else "baseline"
    logger = AgentLogger(goal=mem.goal, mode=mode)
    
    step_num = len(mem.history)
    MAX_STEPS = 25

    while step_num < MAX_STEPS:
        subtask = pick_next_subtask(mem)
        if not subtask:
            break

        step_num += 1
        subtask.status = "in_progress"

        subtask_snap = {
            "id": subtask.id,
            "description": subtask.description,
            "status": subtask.status,
            "depends_on": subtask.depends_on,
            "attempts": subtask.attempts,
            "result": subtask.result,
        }
        thought_data = llm.reason(mem.snapshot(), subtask_snap)
        action_name = thought_data.get("action")
        action_input = thought_data.get("action_input", {})
        reasoning_str = (
            f"Belief: {thought_data.get('belief', '')} | "
            f"Gap: {thought_data.get('gap', '')} | "
            f"Why Action: {thought_data.get('why_action', '')}"
        )

        result = tools.call(action_name, action_input)

        record = StepRecord(
            step_num=step_num,
            reasoning=reasoning_str,
            action=action_name,
            action_input=action_input,
            action_result=result,
            eval_verdict=None,
            eval_reasoning=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            subtask_id=subtask.id,
        )

        if use_self_correction:
            from agent import evaluator
            from agent import recovery
            from dataclasses import asdict

            verdict, eval_reasoning = evaluator.evaluate_step(
                mem.goal, subtask.description, asdict(record), mem.facts
            )

            record.eval_verdict = verdict
            record.eval_reasoning = eval_reasoning

            if verdict == "success":
                subtask.status = "done"
                subtask.result = str(result)
                facts = llm.extract_facts(mem.goal, subtask.description, result)
                for k, v in facts.items():
                    mem.facts[k] = v
            else:
                if verdict == "tool_failure":
                    recovery.recover_tool_failure(mem, subtask, record, step_num)
                elif verdict == "inconsistent":
                    recovery.recover_inconsistency(mem, subtask, record, step_num)
                else:
                    recovery.recover_goal_drift(mem, subtask, record, step_num)

        else:
            subtask.status = "done"
            subtask.result = str(result)

        mem.add_step(record)
        logger.log_step(record)

    unresolved = [
        {"id": t.id, "description": t.description, "status": t.status, "attempts": t.attempts, "result": t.result}
        for t in mem.subtasks if t.status not in ("done", "unresolvable")
    ]

    final_output = llm.synthesize(mem.goal, mem.facts, mem.summary_log, unresolved)
    logger.log_run_end(
        final_output=final_output,
        unresolved_subtasks=unresolved,
        total_steps=step_num,
        total_recoveries=mem.global_recovery_attempts,
    )

    return mem, final_output


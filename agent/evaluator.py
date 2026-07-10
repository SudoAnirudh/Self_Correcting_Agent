from typing import Tuple, Dict, Any
from agent import llm

def evaluate_step(goal: str, subtask_desc: str, record_data: dict, facts_summary: dict) -> Tuple[str, str]:
    """
    Evaluates the step outcome.
    Returns (verdict, reasoning).
    Verdicts: 'success', 'tool_failure', 'inconsistent', 'goal_drift'
    """
    result = record_data.get("action_result", {})
    
    # 1. Programmatic check for tool failure
    if isinstance(result, dict) and "error" in result:
        return "tool_failure", f"Programmatic detection: Tool router returned error '{result['error']}'"
        
    # 2. LLM Evaluator
    eval_data = llm.evaluate(goal, subtask_desc, record_data, facts_summary)
    
    verdict = eval_data.get("verdict", "tool_failure")
    reasoning = eval_data.get("reasoning", "LLM evaluation failed to return standard response.")
    
    # Normalize verdict
    if verdict not in ("success", "tool_failure", "inconsistent", "goal_drift"):
        verdict = "success"
        
    return verdict, reasoning

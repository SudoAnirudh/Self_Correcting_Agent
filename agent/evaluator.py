from typing import Dict, Any
from agent.validation import validate_result


def evaluate_step(goal: str, subtask_desc: str, record_data: dict,
                  facts_summary: dict, prior_result: Any = None) -> Dict[str, Any]:
    """Return deterministic evidence about the step.

    The evaluator reports what happened; it does not decide whether to retry
    or whether the overall task is accepted.
    """
    result = record_data.get("action_result", {})
    action = record_data.get("action", "")
    finding = validate_result(goal, subtask_desc, action, result, prior_result)

    return {
        "status": finding.status,
        "reason_code": finding.reason_code,
        "details": finding.details,
        "progress": finding.progress,
        "acceptance_ready": finding.acceptance_ready,
        "evidence": finding.evidence,
    }

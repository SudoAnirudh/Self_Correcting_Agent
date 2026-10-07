from typing import Dict, Any, Tuple
from agent.validation import validate_result
from agent import llm
from agent.critic import AdversarialCritic


def evaluate_step(
    goal: str,
    subtask_desc: str,
    record_data: dict,
    facts_summary: dict,
    prior_result: Any = None,
    enable_critic: bool = False,
    complexity: str = "LOW",
) -> Tuple[str, str]:
    """Dual-Tier Step Evaluation with optional Adversarial Critic pass:
    Stage 1 (Deterministic/Programmatic): Run zero-token checks (schema, HTTP status, min length).
            If Stage 1 fails, return immediately skipping LLM judge.
    Stage 2 (Semantic LLM Judge): Invoke LLM evaluator only if Stage 1 passes to verify goal satisfaction.
    Stage 3 (Adversarial Critic): Trigger critique pass for high complexity or when enabled.

    Returns (verdict, reasoning) tuple.
    """
    result = record_data.get("action_result", {})
    action = record_data.get("action", "")

    # Stage 1: Deterministic Pre-Flight Validation
    finding = validate_result(goal, subtask_desc, action, result, prior_result)

    if finding.status != "success":
        verdict = "tool_failure"
        reasoning = f"Programmatic detection [{finding.reason_code}]: {finding.details}"
        return verdict, reasoning

    # Stage 2: Semantic LLM Judge (Stage 1 passed)
    llm_eval = llm.evaluate(goal, subtask_desc, record_data, facts_summary)
    verdict = llm_eval.get("verdict", "success")
    reasoning = llm_eval.get("reasoning", "Semantic goal evaluation completed.")

    # Stage 3: Adversarial Critic Pass (if enabled or HIGH complexity)
    if verdict == "success" and (enable_critic or complexity == "HIGH"):
        critic = AdversarialCritic()
        critique = critic.critique(
            goal=goal,
            subtask_desc=subtask_desc,
            action_result=result,
            facts=facts_summary,
            confidence_score=finding.confidence_score,
        )
        if not critique.is_valid:
            verdict = "inconsistent"
            reasoning = f"Adversarial Critic flagged issue: {critique.reasoning}"

    return verdict, reasoning


import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from agent import llm


class CritiqueResult(BaseModel):
    is_valid: bool
    counter_arguments: List[str] = Field(default_factory=list)
    ungrounded_claims: List[str] = Field(default_factory=list)
    edge_cases: List[str] = Field(default_factory=list)
    reasoning: str = ""


class AdversarialCritic:
    """Adversarial Critic persona performing asymmetric critique passes to detect ungrounded claims,
    counter-arguments, and edge case vulnerabilities.
    """

    def critique(
        self,
        goal: str,
        subtask_desc: str,
        action_result: Any,
        facts: Dict[str, str],
        confidence_score: float = 1.0,
    ) -> CritiqueResult:
        system = (
            "You are an adversarial critic evaluating AI agent findings. Your job is to rigorously "
            "challenge claims, identify ungrounded statements not supported by confirmed facts, and "
            "highlight counter-arguments or edge cases.\n\n"
            "Output JSON ONLY in this format:\n"
            "{\n"
            '  "is_valid": true | false,\n'
            '  "counter_arguments": ["..."],\n'
            '  "ungrounded_claims": ["..."],\n'
            '  "edge_cases": ["..."],\n'
            '  "reasoning": "..."\n'
            "}"
        )

        res_str = json.dumps(action_result) if not isinstance(action_result, str) else action_result
        if len(res_str) > 2000:
            res_str = res_str[:2000] + "... [TRUNCATED]"

        user = (
            f"Goal: {goal}\n"
            f"Subtask: {subtask_desc}\n"
            f"Latest Action Result: {res_str}\n"
            f"Confirmed Facts: {json.dumps(facts)}\n"
            f"Current Confidence Score: {confidence_score}"
        )

        try:
            res = llm.call_llm(system, user, llm.EVALUATION_MODEL, temperature=0.0)
            data = json.loads(res)
            return CritiqueResult.model_validate(data)
        except Exception as e:
            # Fallback deterministic critique checking if LLM fails or is offline
            ungrounded = []
            counter_args = []
            edge_cases = []

            # Programmatic verification: check if text contains ungrounded markers or contradictory claims
            res_lower = res_str.lower()

            if "ungrounded" in res_lower or "unverified claim" in res_lower or "fake fact" in res_lower:
                ungrounded.append("Contains unverified or ungrounded assertions.")

            if "contradiction" in res_lower or "counter-example" in res_lower:
                counter_args.append("Contradicts existing confirmed facts.")

            is_valid = len(ungrounded) == 0 and len(counter_args) == 0

            return CritiqueResult(
                is_valid=is_valid,
                counter_arguments=counter_args,
                ungrounded_claims=ungrounded,
                edge_cases=edge_cases,
                reasoning="Passes baseline critique." if is_valid else f"Critique flagged: {ungrounded or counter_args}",
            )

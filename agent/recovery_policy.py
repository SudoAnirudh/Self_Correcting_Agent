from dataclasses import dataclass
from typing import Dict


@dataclass
class RecoveryDecision:
    action: str
    reason: str


SIDE_EFFECTING_TOOLS = {"write", "send", "payment", "provision", "delete", "update", "create"}


def decide(finding: Dict, tool_name: str, attempts: int, max_attempts: int, repeated: bool) -> RecoveryDecision:
    """Choose the next action. Validation only supplies findings; this layer owns policy."""
    if finding.get("acceptance_ready") and finding.get("status") == "success":
        return RecoveryDecision("accept", "Independent deterministic validation passed.")

    if attempts >= max_attempts:
        return RecoveryDecision("stop", "Retry budget exhausted.")

    if finding.get("status") == "ambiguous" and tool_name in SIDE_EFFECTING_TOOLS:
        return RecoveryDecision("escalate", "Side-effecting operation has unknown completion state; require state readback instead of retrying.")

    if repeated or not finding.get("progress", False):
        return RecoveryDecision("reformulate", "No new evidence was produced; change strategy rather than repeat the same action.")

    return RecoveryDecision("retry", f"Recover from {finding.get('reason_code', 'failure')}.")

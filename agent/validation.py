"""Deterministic validation and recovery policy for the agent loop.

The LLM may propose a recovery action, but acceptance is decided by
programmatic evidence. This module keeps finding, decision, and execution
concerns separate.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import json
import re


@dataclass
class ValidationFinding:
    status: str  # success | failure | ambiguous
    reason_code: str
    details: str = ""
    progress: bool = False
    acceptance_ready: bool = False
    confidence_score: float = 1.0
    uncertainty_flags: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryDecision:
    action: str  # retry | reformulate | escalate | stop
    reason: str
    attempt: int


SIDE_EFFECTING_TOOLS = {
    "write",
    "send",
    "payment",
    "provision",
    "delete",
    "update",
    "create",
}


def _stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sanitize_error(error_type: str, detail: str) -> Dict[str, str]:
    """Return a bounded, redacted error contract suitable for the LLM."""
    text = str(detail or "")
    # Strip common secret-bearing patterns while retaining useful failure class info.
    text = re.sub(r"(?i)(api[_-]?key|token|authorization|password|secret)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", text)
    text = re.sub(r"https?://[^\s]+", "[URL_REDACTED]", text)
    text = text[:500]
    return {
        "error_type": str(error_type)[:80],
        "detail": text,
    }


def validate_result(goal: str, subtask_desc: str, action: str, result: Any,
                    prior_result: Optional[Any] = None) -> ValidationFinding:
    """Validate tool output using deterministic checks only (Stage 1)."""
    # 1. Schema / Type Validation
    if not isinstance(result, dict):
        return ValidationFinding(
            status="failure",
            reason_code="SCHEMA_VIOLATION",
            details="Tool result was not a structured object.",
            confidence_score=0.0,
            uncertainty_flags=["SCHEMA_VIOLATION"],
        )

    if result.get("status") == "unknown" or result.get("ambiguous") is True:
        return ValidationFinding(
            status="ambiguous",
            reason_code="ambiguous_outcome",
            details="The tool could not establish whether the requested state change completed.",
            confidence_score=0.5,
            uncertainty_flags=["ambiguous_outcome"],
        )

    if "error" in result:
        err_code = str(result.get("error", "tool_error"))
        reason_code = "SCHEMA_VIOLATION" if err_code in ("invalid_input", "schema_mismatch") else err_code
        return ValidationFinding(
            status="failure",
            reason_code=reason_code,
            details=sanitize_error(err_code, result.get("detail", ""))["detail"],
            confidence_score=0.0,
            uncertainty_flags=[reason_code],
        )

    # 2. HTTP Status Code Checks
    if "status_code" in result:
        status_code = result["status_code"]
        if isinstance(status_code, int) and status_code != 200:
            return ValidationFinding(
                status="failure",
                reason_code="HTTP_ERROR",
                details=f"HTTP status code {status_code} indicates a non-200 error response.",
                confidence_score=0.0,
                uncertainty_flags=["HTTP_ERROR"],
            )

    # 3. Search/fetch style research outputs: schema + minimum-content checks
    if action == "search":
        rows = result.get("results")
        if not isinstance(rows, list):
            return ValidationFinding("failure", "SCHEMA_VIOLATION", "Expected results list in search output.", confidence_score=0.0, uncertainty_flags=["SCHEMA_VIOLATION"])
        valid_rows = [r for r in rows if isinstance(r, dict) and r.get("url")]
        if not valid_rows:
            return ValidationFinding("failure", "EMPTY_OUTPUT", "Search returned no usable results.", confidence_score=0.0, uncertainty_flags=["EMPTY_OUTPUT"])
        fingerprint = _stable_fingerprint(result)
        progressed = prior_result is None or fingerprint != _stable_fingerprint(prior_result)
        flags = ["low_result_count"] if len(valid_rows) < 2 else []
        conf = 0.8 if len(valid_rows) < 2 else 1.0
        return ValidationFinding(
            status="success",
            reason_code="schema_and_content_ok",
            details="Search output passed deterministic checks.",
            progress=progressed,
            acceptance_ready=True,
            confidence_score=conf,
            uncertainty_flags=flags,
            evidence={"usable_results": len(valid_rows), "fingerprint": fingerprint},
        )

    if action in {"fetch", "flaky_fetch"}:
        text = result.get("text")
        if not isinstance(text, str):
            return ValidationFinding("failure", "SCHEMA_VIOLATION", "Expected text field in fetch output.", confidence_score=0.0, uncertainty_flags=["SCHEMA_VIOLATION"])
        if len(text.strip()) < 10:
            return ValidationFinding("failure", "MIN_LENGTH_VIOLATION", "Fetched content length is below minimum required threshold (10 characters).", confidence_score=0.0, uncertainty_flags=["MIN_LENGTH_VIOLATION"])
        lowered = text.lower()
        garbage_markers = ("access denied", "error 403", "forbidden", "timeout", "temporarily unavailable")
        if any(marker in lowered for marker in garbage_markers):
            return ValidationFinding("failure", "HTTP_ERROR", "Fetched content contains HTTP error / access denial response.", confidence_score=0.0, uncertainty_flags=["HTTP_ERROR"])
        progressed = prior_result is None or _stable_fingerprint(result) != _stable_fingerprint(prior_result)
        flags = ["short_content"] if len(text) < 50 else []
        conf = 0.85 if len(text) < 50 else 1.0
        return ValidationFinding(
            status="success",
            reason_code="schema_and_content_ok",
            details="Fetch output passed deterministic checks.",
            progress=progressed,
            acceptance_ready=True,
            confidence_score=conf,
            uncertainty_flags=flags,
            evidence={"content_length": len(text), "fingerprint": _stable_fingerprint(result)},
        )

    # Generic read-only tool contract checks
    if not result:
        return ValidationFinding("failure", "EMPTY_OUTPUT", "Result payload is empty.", confidence_score=0.0, uncertainty_flags=["EMPTY_OUTPUT"])

    return ValidationFinding(
        status="success",
        reason_code="structured_output_ok",
        details="Structured output passed baseline checks.",
        progress=prior_result is None or _stable_fingerprint(result) != _stable_fingerprint(prior_result),
        acceptance_ready=True,
        confidence_score=1.0,
        uncertainty_flags=[],
        evidence={"fingerprint": _stable_fingerprint(result)},
    )



def choose_recovery(finding: ValidationFinding, action: str, attempt: int,
                    max_attempts: int, progress_fingerprint_seen: bool) -> RecoveryDecision:
    """Policy layer: decide what happens next; the validator only reports evidence."""
    if finding.acceptance_ready and finding.status == "success":
        return RecoveryDecision("stop", "Independent validation passed.", attempt)

    if attempt >= max_attempts:
        return RecoveryDecision("stop", "Retry contract exhausted.", attempt)

    if finding.status == "ambiguous" and action in SIDE_EFFECTING_TOOLS:
        return RecoveryDecision(
            "escalate",
            "Side-effecting operation has unknown completion state; do not blindly retry.",
            attempt,
        )

    if progress_fingerprint_seen:
        return RecoveryDecision(
            "reformulate",
            "Previous attempt produced no new evidence; change strategy instead of repeating it.",
            attempt,
        )

    return RecoveryDecision("retry", f"Recover from {finding.reason_code}.", attempt)

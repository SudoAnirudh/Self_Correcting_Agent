import hashlib
import json
import re
from typing import Any, Dict, Optional


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sanitize_error(error_type: str, detail: str) -> Dict[str, str]:
    text = str(detail or "")
    text = re.sub(r"(?i)(api[_-]?key|token|authorization|password|secret)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", text)
    text = text[:500]
    return {"error_type": str(error_type)[:80], "detail": text}


def validate_result(action: str, result: Any, previous_result: Optional[Any] = None) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"status": "failure", "reason_code": "invalid_result_shape", "progress": False, "acceptance_ready": False}

    if "error" in result:
        return {
            "status": "failure",
            "reason_code": str(result.get("error", "tool_error")),
            "details": sanitize_error(result.get("error", "tool_error"), result.get("detail", "")),
            "progress": False,
            "acceptance_ready": False,
        }

    if action == "search":
        rows = result.get("results")
        if not isinstance(rows, list) or not any(isinstance(r, dict) and r.get("url") for r in rows):
            return {"status": "failure", "reason_code": "empty_search", "progress": False, "acceptance_ready": False}

    if action in {"fetch", "flaky_fetch"}:
        if not isinstance(result.get("url"), str) or not isinstance(result.get("text"), str):
            return {"status": "failure", "reason_code": "schema_mismatch", "progress": False, "acceptance_ready": False}
        text = result["text"].lower()
        if any(x in text for x in ("access denied", "error 403", "forbidden", "temporarily unavailable")):
            return {"status": "failure", "reason_code": "invalid_fetch_content", "progress": False, "acceptance_ready": False}

    current_fp = fingerprint(result)
    previous_fp = fingerprint(previous_result) if previous_result is not None else None
    return {
        "status": "success",
        "reason_code": "deterministic_checks_passed",
        "progress": current_fp != previous_fp,
        "acceptance_ready": True,
        "evidence": {"fingerprint": current_fp},
    }

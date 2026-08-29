import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.progress import validate_result, sanitize_error, fingerprint
from agent.recovery_policy import decide


def test_search_requires_usable_results():
    finding = validate_result("search", {"results": []})
    assert finding["status"] == "failure"
    assert not finding["acceptance_ready"]


def test_fetch_rejects_error_page():
    finding = validate_result("fetch", {
        "url": "https://example.com",
        "text": "Error 403 Access Denied"
    })
    assert finding["status"] == "failure"
    assert finding["reason_code"] == "invalid_fetch_content"


def test_same_result_is_not_progress():
    result = {"url": "https://example.com", "text": "hello"}
    finding = validate_result("fetch", result, result)
    assert finding["progress"] is False


def test_error_is_sanitized():
    cleaned = sanitize_error("tool_exception", "API_KEY=supersecret token=abc123 Connection failed")
    assert "supersecret" not in cleaned["detail"]
    assert "abc123" not in cleaned["detail"]
    assert len(cleaned["detail"]) <= 500


def test_side_effect_timeout_escalates():
    finding = {"status": "ambiguous", "acceptance_ready": False, "progress": False, "reason_code": "timeout"}
    decision = decide(finding, "payment", attempts=1, max_attempts=3, repeated=False)
    assert decision.action == "escalate"


def test_repeated_failure_changes_strategy():
    finding = {"status": "failure", "acceptance_ready": False, "progress": False, "reason_code": "tool_error"}
    decision = decide(finding, "fetch", attempts=1, max_attempts=3, repeated=True)
    assert decision.action == "reformulate"

"""Compatibility helpers for progress tracking.

Validation and acceptance now live in agent.validation. This module keeps
fingerprinting and error sanitization available to callers without defining
a second acceptance path.
"""

import hashlib
import json
import re
from typing import Any, Dict


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sanitize_error(error_type: str, detail: str) -> Dict[str, str]:
    text = str(detail or "")
    text = re.sub(
        r"(?i)(api[_-]?key|token|authorization|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
    return {"error_type": str(error_type)[:80], "detail": text[:500]}

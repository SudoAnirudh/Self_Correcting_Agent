import os
import sys
import time
import subprocess
import json
import re
from typing import Optional, Any
from pydantic import BaseModel, Field


class SandboxExecutionResult(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    execution_time_seconds: float = 0.0
    error_type: Optional[str] = None
    result_value: Optional[Any] = None


class PythonCodeSandbox:
    """Executes dynamic Python code in an isolated subprocess with strict timeouts and restricted environment."""

    def __init__(self, default_timeout: float = 5.0):
        self.default_timeout = default_timeout

    def execute_code(self, code: str, timeout: Optional[float] = None) -> SandboxExecutionResult:
        eff_timeout = timeout if timeout is not None else self.default_timeout
        start_time = time.time()

        # Build clean environment
        restricted_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PYTHONUNBUFFERED": "1",
        }

        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=eff_timeout,
                env=restricted_env,
            )
            duration = time.time() - start_time
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""

            if completed.returncode == 0:
                # Try parsing stdout as JSON or scalar value if possible
                result_val = stdout.strip()
                if result_val:
                    try:
                        result_val = json.loads(result_val)
                    except json.JSONDecodeError:
                        pass
                return SandboxExecutionResult(
                    success=True,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=0,
                    execution_time_seconds=duration,
                    result_value=result_val,
                )

            # Execution failed with non-zero returncode
            error_type = self._parse_error_type(stderr)
            return SandboxExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                returncode=completed.returncode,
                execution_time_seconds=duration,
                error_type=error_type,
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            return SandboxExecutionResult(
                success=False,
                stdout=e.stdout or "" if isinstance(e.stdout, str) else "",
                stderr=f"Execution timed out after {eff_timeout} seconds",
                returncode=-1,
                execution_time_seconds=duration,
                error_type="TIMEOUT",
            )
        except Exception as e:
            duration = time.time() - start_time
            return SandboxExecutionResult(
                success=False,
                stdout="",
                stderr=f"Sandbox error: {str(e)}",
                returncode=-1,
                execution_time_seconds=duration,
                error_type=type(e).__name__,
            )

    def _parse_error_type(self, stderr: str) -> str:
        """Parses error exception name from stderr traceback."""
        if not stderr:
            return "RUNTIME_ERROR"
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*Error|Exception):", stderr)
        if match:
            return match.group(1)
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        if lines:
            return lines[-1].split(":")[0]
        return "RUNTIME_ERROR"

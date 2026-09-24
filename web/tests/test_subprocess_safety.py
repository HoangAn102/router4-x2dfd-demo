import asyncio
import sys
import pytest
from web.backend.app.utils.subprocess_runner import (
    SubprocessRunner,
    SubprocessTimeoutError,
    SubprocessExecutionError,
)


@pytest.mark.anyio
async def test_subprocess_runner_successful_execution():
    """Verify clean execution and stdout retrieval."""
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('hello_safe_subproc')"]
    stdout, stderr = await SubprocessRunner.run(cmd, timeout=5.0)
    assert "hello_safe_subproc" in stdout
    assert stderr == ""


@pytest.mark.anyio
async def test_subprocess_runner_timeout_raises_and_kills():
    """Verify timeout triggers SubprocessTimeoutError and kills process."""
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    with pytest.raises(SubprocessTimeoutError):
        await SubprocessRunner.run(cmd, timeout=0.5)


@pytest.mark.anyio
async def test_subprocess_runner_nonzero_exit_raises_execution_error():
    """Verify non-zero return code raises SubprocessExecutionError with stderr."""
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('fatal_internal_err'); sys.exit(42)"]
    with pytest.raises(SubprocessExecutionError) as exc_info:
        await SubprocessRunner.run(cmd, timeout=5.0)
    assert exc_info.value.returncode == 42
    assert "fatal_internal_err" in exc_info.value.stderr


def test_subprocess_runner_environment_sanitization():
    """Verify isolated environment strips dangerous variables while preserving needed paths."""
    sanitized = SubprocessRunner._sanitize_env({"CUSTOM_VAR": "custom_val"})
    assert sanitized.get("PYTHONNOUSERSITE") == "1"
    assert sanitized.get("CUSTOM_VAR") == "custom_val"
    assert "LANG" in sanitized

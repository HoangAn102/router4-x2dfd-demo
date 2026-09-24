import asyncio
import os
import signal
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
from .logger import logger


class SubprocessExecutionError(Exception):
    def __init__(self, message: str, returncode: int, stderr: str):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class SubprocessTimeoutError(Exception):
    pass


class SubprocessRunner:
    """Safe subprocess executor adhering to security and timeout invariants."""

    @staticmethod
    def _sanitize_env(custom_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Produce an isolated environment dictionary."""
        base_env = {
            "PYTHONNOUSERSITE": "1",
            "LANG": "C.UTF-8",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }
        for key in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "USER", "HOME"]:
            if key in os.environ:
                base_env[key] = os.environ[key]

        if custom_env:
            base_env.update(custom_env)
        return base_env

    @classmethod
    async def run(
        cls,
        cmd: List[str],
        *,
        timeout: float,
        cwd: Optional[str] = None,
        custom_env: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """Execute a command asynchronously with process group kill on timeout."""
        env = cls._sanitize_env(custom_env)
        logger.info(f"Subprocess starting: {' '.join(cmd[:4])} (timeout={timeout}s)")

        # Create subprocess
        is_windows = sys.platform == "win32"
        creationflags = 0

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            creationflags=creationflags,
        )

        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"Subprocess timed out after {timeout}s: {' '.join(cmd[:3])}")
            try:
                if is_windows:
                    proc.kill()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception as e:
                logger.warning(f"Error killing timed out process: {e}")
            raise SubprocessTimeoutError(f"Subprocess exceeded timeout of {timeout}s")

        stdout_str = stdout_data.decode("utf-8", errors="replace")
        stderr_str = stderr_data.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            logger.error(f"Subprocess failed (rc={proc.returncode}): {stderr_str[-500:]}")
            raise SubprocessExecutionError(
                f"Subprocess exited with code {proc.returncode}",
                returncode=proc.returncode,
                stderr=stderr_str,
            )

        return stdout_str, stderr_str

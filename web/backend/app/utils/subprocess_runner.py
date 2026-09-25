import asyncio
import os
import signal
import sys
from typing import Dict, List, Optional, Tuple

from .logger import logger


class SubprocessExecutionError(Exception):
    def __init__(
        self,
        message: str,
        returncode: int,
        stderr: str,
    ):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class SubprocessTimeoutError(Exception):
    pass


class SubprocessRunner:
    """Isolated async subprocess runner with fail-closed timeout."""

    @staticmethod
    def _sanitize_env(
        custom_env: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:

        env = {
            "PYTHONNOUSERSITE": "1",
            "LANG": "C.UTF-8",
            "PATH": os.environ.get(
                "PATH",
                "/usr/bin:/bin"
            ),
        }

        for key in [
            "CUDA_VISIBLE_DEVICES",
            "USER",
            "HOME",
        ]:
            if key in os.environ:
                env[key] = os.environ[key]

        if custom_env:
            env.update(
                {
                    str(k): str(v)
                    for k, v in custom_env.items()
                }
            )

        return env

    @classmethod
    async def run(
        cls,
        cmd: List[str],
        *,
        timeout: float,
        cwd: Optional[str] = None,
        custom_env: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:

        env = cls._sanitize_env(custom_env)

        logger.info(
            "Subprocess: %s",
            " ".join(cmd[:5]),
        )

        is_windows = sys.platform == "win32"

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=not is_windows,
        )

        try:
            stdout_data, stderr_data = (
                await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            )

        except asyncio.TimeoutError:

            logger.error(
                "Subprocess timeout after %ss",
                timeout,
            )

            try:
                if is_windows:
                    proc.kill()
                else:
                    os.killpg(
                        os.getpgid(proc.pid),
                        signal.SIGKILL,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to kill process group: %s",
                    e,
                )

            raise SubprocessTimeoutError(
                f"Subprocess exceeded {timeout}s"
            )

        stdout = stdout_data.decode(
            "utf-8",
            errors="replace",
        )

        stderr = stderr_data.decode(
            "utf-8",
            errors="replace",
        )

        if proc.returncode != 0:
            raise SubprocessExecutionError(
                "Subprocess failed",
                proc.returncode,
                stderr[-4000:],
            )

        return stdout, stderr

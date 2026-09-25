import asyncio
from typing import Optional

from .logger import logger


class GpuLock:
    """
    Process-local asynchronous GPU serialization lock.

    SafeVision production invariant:
      - Uvicorn runs exactly ONE worker.
      - Only one inference pipeline may use the GPU at a time.
      - Concurrent requests wait in-process instead of failing because
        of an external filesystem lock.

    Heavy expert / X2DFD subprocesses are already guarded by their own
    execution timeouts, so a waiting request is released when the
    active inference completes or fails.
    """

    _async_lock: Optional[asyncio.Lock] = None

    def __init__(self, *args, **kwargs):
        # Keep a permissive signature for compatibility with older calls.
        pass

    @classmethod
    def _get_async_lock(cls) -> asyncio.Lock:
        if cls._async_lock is None:
            cls._async_lock = asyncio.Lock()

        return cls._async_lock

    async def __aenter__(self):
        lock = self._get_async_lock()

        if lock.locked():
            logger.info(
                "GPU busy: request queued until current inference finishes."
            )

        await lock.acquire()

        logger.info(
            "GPU inference lock acquired."
        )

        return self

    async def __aexit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        lock = self._get_async_lock()

        if lock.locked():
            lock.release()

        logger.info(
            "GPU inference lock released."
        )

        return False

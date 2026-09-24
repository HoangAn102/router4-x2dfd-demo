import asyncio
from pathlib import Path
from typing import Optional

try:
    from filelock import FileLock
except ImportError:
    FileLock = None

from ..config import settings
from .logger import logger


class GpuLock:
    """Inter-process and inter-coroutine lock for GPU inference safety."""
    _async_lock: Optional[asyncio.Lock] = None

    def __init__(self, lock_path: Optional[Path] = None, timeout: float = 120.0):
        self.lock_path = lock_path or settings.GPU_LOCK_FILE
        self.timeout = timeout
        self.file_lock = FileLock(str(self.lock_path), timeout=self.timeout) if FileLock else None

    @classmethod
    def _get_async_lock(cls) -> asyncio.Lock:
        if cls._async_lock is None:
            cls._async_lock = asyncio.Lock()
        return cls._async_lock

    async def __aenter__(self):
        async_lock = self._get_async_lock()
        await async_lock.acquire()
        if self.file_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.file_lock.acquire)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.file_lock and self.file_lock.is_locked:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.file_lock.release)
        finally:
            async_lock = self._get_async_lock()
            if async_lock.locked():
                async_lock.release()

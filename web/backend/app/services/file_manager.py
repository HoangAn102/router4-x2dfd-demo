import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from ..config import settings
from ..utils.logger import logger


def ensure_upload_dir() -> Path:
    """Ensure the root upload directory exists."""
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return settings.UPLOAD_DIR


@asynccontextmanager
async def isolated_temp_workspace(prefix: str = "req") -> AsyncGenerator[Path, None]:
    """Context manager providing an isolated temporary directory with guaranteed cleanup."""
    root = ensure_upload_dir()
    unique_id = f"{prefix}_{uuid.uuid4().hex}"
    temp_dir = root / unique_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        yield temp_dir
    finally:
        # Layer 1 cleanup: regular request finally block
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary workspace: {temp_dir.name}")
        except Exception as e:
            logger.warning(f"Error during workspace cleanup for {temp_dir}: {e}")


def cleanup_stale_temp_dirs(max_age_hours: int = 1) -> int:
    """Layer 2 cleanup: Sweep and delete orphan temp directories older than max_age_hours."""
    root = ensure_upload_dir()
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0

    if not root.exists():
        return 0

    for item in root.iterdir():
        if item.is_dir():
            try:
                mtime = item.stat().st_mtime
                if (now - mtime) > max_age_seconds:
                    shutil.rmtree(item, ignore_errors=True)
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to inspect/delete stale dir {item}: {e}")

    if cleaned_count > 0:
        logger.info(f"Stale temp sweep: Removed {cleaned_count} orphan directories.")
    return cleaned_count

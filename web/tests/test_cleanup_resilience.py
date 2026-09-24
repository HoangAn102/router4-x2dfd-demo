import os
import time
from pathlib import Path
import pytest
from web.backend.app.services.file_manager import (
    cleanup_stale_temp_dirs,
    ensure_upload_dir,
    isolated_temp_workspace,
)


def test_isolated_temp_workspace_cleanup():
    import asyncio

    async def _run():
        temp_dir_captured = None
        async with isolated_temp_workspace(prefix="test_resilience") as temp_dir:
            temp_dir_captured = temp_dir
            assert temp_dir.exists()
            test_file = temp_dir / "sample.txt"
            test_file.write_text("Hello isolation")
            assert test_file.exists()
        return temp_dir_captured

    temp_dir_captured = asyncio.run(_run())
    assert temp_dir_captured is not None
    assert not temp_dir_captured.exists()


def test_stale_temp_dirs_sweep():
    root = ensure_upload_dir()
    stale_dir = root / "stale_orphan_test_dir"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale_file = stale_dir / "dummy.dat"
    stale_file.write_bytes(b"stale data")

    # Manually set modification time to 2 hours in the past
    past_time = time.time() - 7200
    os.utime(stale_dir, (past_time, past_time))

    # Run sweep with max_age_hours=1
    cleaned = cleanup_stale_temp_dirs(max_age_hours=1)
    assert cleaned >= 1
    assert not stale_dir.exists()

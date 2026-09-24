import pytest
from web.backend.app.config import settings
from web.model_worker.mock_worker import MockModelWorker
from web.model_worker.worker_factory import get_model_worker


def test_production_guard_mock_mode():
    """Verify that in mock mode, MockModelWorker is returned cleanly."""
    settings.RUN_MODE = "mock"
    worker = get_model_worker()
    assert isinstance(worker, MockModelWorker)


def test_production_guard_live_mode_refuses_mock_fallback(monkeypatch):
    """
    CRITICAL INVARIANT TEST:
    When RUN_MODE='live', if LiveModelWorker fails to initialize,
    the system MUST raise RuntimeError and NEVER silently fallback to MockModelWorker.
    """
    settings.RUN_MODE = "live"
    import web.model_worker.live_worker as live_module

    def failing_init(*args, **kwargs):
        raise ImportError("Simulated missing GPU dependencies on node")

    monkeypatch.setattr(live_module, "LiveModelWorker", failing_init)

    with pytest.raises(RuntimeError, match="Production Guard Violation"):
        get_model_worker()

    # Restore settings
    settings.RUN_MODE = "mock"

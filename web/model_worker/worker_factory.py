from web.backend.app.config import settings
from web.backend.app.utils.logger import logger
from .base import BaseModelWorker
from .mock_worker import MockModelWorker


def get_model_worker() -> BaseModelWorker:
    """Factory function returning the configured model worker with Production Guard."""
    mode = settings.RUN_MODE.lower()
    logger.info(f"Initializing Model Worker for RUN_MODE='{mode}'")

    if mode == "mock":
        return MockModelWorker()

    if mode == "live":
        try:
            from .live_worker import LiveModelWorker
            return LiveModelWorker()
        except Exception as e:
            logger.critical(
                f"Production Guard: Failed to initialize LiveModelWorker in RUN_MODE='live': {e}"
            )
            # NEVER fallback to mock in live mode!
            raise RuntimeError(
                f"Production Guard Violation: RUN_MODE='live' is configured, but LiveModelWorker "
                f"cannot be loaded: {e}. Refusing to start with mock worker."
            ) from e

    raise ValueError(f"Unknown RUN_MODE '{settings.RUN_MODE}'. Must be 'mock' or 'live'.")

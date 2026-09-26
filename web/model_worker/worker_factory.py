from typing import Optional

from web.backend.app.config import settings
from web.backend.app.utils.logger import logger

from .base import BaseModelWorker
from .mock_worker import MockModelWorker


_LIVE_WORKER: Optional[BaseModelWorker] = None
_MOCK_WORKER: Optional[BaseModelWorker] = None


def get_model_worker() -> BaseModelWorker:
    """
    One process-local worker.

    Production uses exactly one Uvicorn worker, therefore Router4,
    calibrators and the X2DFD client can safely persist across requests.
    """

    global _LIVE_WORKER
    global _MOCK_WORKER

    mode = settings.RUN_MODE.lower()

    if mode == "mock":

        if _MOCK_WORKER is None:
            logger.info("Creating singleton MockModelWorker")
            _MOCK_WORKER = MockModelWorker()

        return _MOCK_WORKER

    if mode == "live":

        if _LIVE_WORKER is None:

            logger.info("Creating singleton LiveModelWorker")

            try:
                from .live_worker import LiveModelWorker
                _LIVE_WORKER = LiveModelWorker()

            except Exception as e:

                logger.critical(
                    "LiveModelWorker initialization failed: %s",
                    e,
                )

                raise RuntimeError(
                    "RUN_MODE='live' but LiveModelWorker "
                    "cannot initialize. Mock fallback refused."
                ) from e

        return _LIVE_WORKER

    raise ValueError(
        f"Unknown RUN_MODE={settings.RUN_MODE!r}"
    )

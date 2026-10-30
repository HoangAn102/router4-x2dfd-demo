"""First request after a worker (re)start gets the model-load budget, later ones the per-image budget."""
import asyncio
import json
from unittest import mock

from web.backend.app.config import settings
from web.model_worker.x2dfd_client import X2DFDClient


class FakeProc:
    def __init__(self):
        self.returncode = None
        self.stdin = mock.Mock(drain=mock.AsyncMock())
        self.stdout = mock.Mock(readline=mock.AsyncMock(return_value=json.dumps({"ok": True}).encode() + b"\n"))


def test_first_request_uses_load_timeout(monkeypatch):
    c = X2DFDClient()
    seen = []

    async def fake_wait_for(coro, timeout):
        seen.append(timeout)
        return await coro

    async def ensure():
        if c._proc is None:
            c._proc = FakeProc()

    monkeypatch.setattr(c, "_ensure_worker", ensure)
    with mock.patch("web.model_worker.x2dfd_client.asyncio.wait_for", fake_wait_for):
        asyncio.run(c._request({"mode": "score"}))
        asyncio.run(c._request({"mode": "score"}))
    assert seen == [settings.EXPERT_INFERENCE_TIMEOUT_SEC, settings.IMAGE_INFERENCE_TIMEOUT_SEC]

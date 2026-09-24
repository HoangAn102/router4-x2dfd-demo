import pytest
from pathlib import Path
from PIL import Image
from web.model_worker.mock_worker import MockModelWorker


@pytest.mark.anyio
async def test_mock_worker_image_analysis(tmp_path):
    worker = MockModelWorker()
    img_path = tmp_path / "mock_test.jpg"
    img = Image.new("RGB", (100, 100), color="green")
    img.save(img_path, "JPEG")

    resp = await worker.analyze_image(img_path, request_id="test_req_mock_1")
    assert resp.request_id == "test_req_mock_1"
    assert resp.status == "ok"
    assert resp.media_type == "image"
    assert resp.router.selected_expert in ["blending", "diffusion", "frequency", "texture"]
    assert 0.0 <= resp.router.router_confidence <= 1.0
    assert 0.0 <= resp.expert.raw_score <= 1.0
    assert 0.0 <= resp.expert.calibrated_score <= 1.0
    assert resp.final.verdict in ["REAL", "FAKE"]
    assert 0.0 <= resp.final.fake_probability <= 1.0
    assert 0.0 <= resp.final.real_probability <= 1.0
    assert round(resp.final.fake_probability + resp.final.real_probability, 4) == 1.0
    assert resp.provenance["mode"] == "mock_development"


@pytest.mark.anyio
async def test_mock_worker_video_frame_and_aggregate(tmp_path):
    worker = MockModelWorker()
    frame_path = tmp_path / "mock_frame.jpg"
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(frame_path, "JPEG")

    frame_res = await worker.analyze_video_frame(frame_path, frame_idx=0, timestamp=0.0)
    assert frame_res.frame_index == 0
    assert frame_res.verdict in ["REAL", "FAKE"]
    assert 0.0 <= frame_res.final_frame_score <= 1.0

    final_res, err_code, err_msg = worker.aggregate_video([frame_res])
    assert final_res is None
    assert err_code == "VIDEO_AGGREGATION_UNVERIFIED"

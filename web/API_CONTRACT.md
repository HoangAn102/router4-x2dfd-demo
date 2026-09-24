# Router4 + X²-DFD Web API Contract Specification (V4)

This document specifies the exact JSON contracts, schemas, headers, error structures, and behavioral invariants for the Router4 + X²-DFD Deepfake Detection backend service.

---

## 1. Global Invariants & Conventions

1. **Score Normalization & Types**:
   - Probabilities `fake_probability` and `real_probability` are continuous 64-bit floats bounded strictly to `[0.0, 1.0]`.
   - Pairwise sum: `fake_probability + real_probability == 1.0` (tolerance: $\le 10^{-4}$).
   - `boolean` values (`True`/`False`), `NaN`, and `Infinity` are rejected as invalid (Fail-Closed).
2. **AI Explanation Rule**:
   - `explanation: Optional[str] = None`
   - Populated **only** when the vision-language model generates coherent natural language reasoning beyond the classification label token.
   - If only the label ("real" or "fake") is generated, backend sets `explanation = null`.
   - The Frontend MUST hide the explanation section when `explanation === null`.
3. **Video Fail-Closed Invariant**:
   - Because no video-level aggregation algorithm exists in the research codebase snapshot, full video results return `status: "blocked"` with `error_code: "VIDEO_AGGREGATION_UNVERIFIED"` and `final: null`.
   - Individual frame-level detections are preserved in `frames[]`.
4. **Error Handling**:
   - All client and server errors return structured JSON with `stage`, `error_code`, `message`, and `request_id`.

---

## 2. Health & Readiness Endpoints

### 2.1 Liveness Probe
`GET /api/v1/health`
- **Description**: Verifies that the FastAPI process is responsive.
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "app_name": "Router4 + X²-DFD Web API",
  "version": "1.0.0",
  "run_mode": "mock",
  "timestamp": 1727168000.123
}
```

### 2.2 Worker Readiness Probe
`GET /api/v1/ready`
- **Description**: Checks readiness of active AI worker artifacts (checkpoints, weights, Python wrapper scripts).
- **Response `200 OK`**:
```json
{
  "ready": true,
  "mode": "mock",
  "components": {
    "mock_engine": {
      "name": "Mock Engine",
      "ready": true,
      "details": "Mock worker active for local development/testing."
    }
  }
}
```

---

## 3. Image Analysis (Synchronous)

`POST /api/v1/analyze/image`
- **Content-Type**: `multipart/form-data`
- **Form Param**: `file` (Binary Image: JPEG, PNG, WEBP; max 20MB, max $4096\times 4096$)

### Response `200 OK`:
```json
{
  "request_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
  "status": "ok",
  "media_type": "image",
  "router": {
    "selected_expert": "blending",
    "selected_alias": "Blending",
    "router_confidence": 0.8842,
    "margin": 0.6342
  },
  "expert": {
    "name": "blending",
    "raw_score": 0.8123,
    "calibrated_score": 0.8415,
    "score_direction": "fake_probability"
  },
  "final": {
    "verdict": "FAKE",
    "fake_probability": 0.8415,
    "real_probability": 0.1585,
    "continuous_score": 0.8415,
    "decision_threshold": 0.5,
    "explanation": "Visual cues show boundary inconsistency consistent with Blending artifacts."
  },
  "timing_ms": {
    "routing": 24.5,
    "expert_inference": 48.2,
    "x2dfd_inference": 112.4,
    "total": 185.1
  },
  "provenance": {
    "mode": "live_server",
    "router_checkpoint": "best.pt",
    "lora_adapter": "llava-v1.5-7b-lora"
  }
}
```

---

## 4. Video Analysis (Asynchronous Job & Polling)

### 4.1 Initiate Video Job
`POST /api/v1/analyze/video`
- **Content-Type**: `multipart/form-data`
- **Form Param**: `file` (Binary Video: MP4, AVI, MOV, MKV; max 100MB, duration $\le 60\text{s}$)
- **Response `202 Accepted`**:
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "processing",
  "poll_url": "/api/v1/analyze/video/jobs/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "message": "Video accepted for asynchronous processing. Please poll poll_url for progress."
}
```

### 4.2 Poll Video Status
`GET /api/v1/analyze/video/jobs/{job_id}`

#### While Processing:
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "processing",
  "stage": "analyzing_frames",
  "current_frame": 14,
  "total_frames": 32,
  "result": null,
  "error": null
}
```

#### When Aggregation is Blocked (Source Research Invariant):
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "blocked",
  "stage": "blocked_aggregation_unverified",
  "current_frame": 32,
  "total_frames": 32,
  "result": {
    "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "status": "blocked",
    "media_type": "video",
    "video": {
      "duration_seconds": 12.4,
      "fps": 30.0,
      "total_frames_extracted": 32,
      "frames_analyzed": 32,
      "sampling_protocol": "DeepfakeBench_uniform"
    },
    "final": null,
    "error_code": "VIDEO_AGGREGATION_UNVERIFIED",
    "message": "Video aggregation algorithm is unverified from repository research source. Frame-level forensic results are preserved.",
    "frames": [
      {
        "frame_index": 0,
        "timestamp_seconds": 0.0,
        "selected_expert": "frequency",
        "expert_score": 0.82,
        "final_frame_score": 0.85,
        "verdict": "FAKE",
        "thumbnail_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
      }
    ],
    "timing_ms": {
      "total": 8450.2
    },
    "provenance": {
      "mode": "live_server"
    }
  },
  "error": null
}
```

---

## 5. Error Contract

When any validation or internal failure occurs, an HTTP $4xx$ or $5xx$ status code is returned with:

```json
{
  "detail": {
    "code": "IMAGE_TOO_LARGE",
    "stage": "media_validation",
    "message": "Uploaded image size (25.40MB) exceeds maximum allowed limit (20.00MB).",
    "request_id": "c13e551c-bf1b-4eb7-a7eb-fc9df1234567",
    "details": null
  }
}
```

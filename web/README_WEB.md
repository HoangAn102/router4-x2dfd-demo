# Router4 + X²-DFD Web Application Architecture

Welcome to the Router4 + X²-DFD deepfake detection web application platform.

This system integrates a hierarchical mixture of experts:
1. **Router4 Classifier**: EfficientNet-B0 assigning input media to 1 of 4 forensic experts (`blending`, `diffusion`, `frequency`, `texture`).
2. **Forensic Expert Models**: Specialized detectors generating raw forensic scores.
3. **Teacher Calibrators**: Isotonic / logistic regression calibrators scaling raw detector scores into consistent probability space $P(\text{Fake})$.
4. **X²-DFD (LLaVA + LoRA)**: Vision-Language Multimodal Model with Weighted Forensic Scoring (WFS) prompts outputting final continuous fake/real scores and reasoning explanations.
5. **Video Engine**: Asynchronous uniform 32-frame extraction with fail-closed aggregation gate preserving per-frame forensics.

---

## Directory Structure

```
web/
├── backend/
│   ├── app/
│   │   ├── config.py             # Multi-tier timeouts, limits, run mode
│   │   ├── main.py               # FastAPI application entrypoint with lifespan
│   │   ├── routers/
│   │   │   ├── health.py         # /health and /ready probes
│   │   │   ├── analyze_image.py  # Synchronous image analysis endpoint
│   │   │   └── analyze_video.py  # Asynchronous video analysis & polling
│   │   ├── schemas/              # Pydantic contract schemas (common, image, video)
│   │   ├── services/
│   │   │   ├── file_manager.py   # Dual-layer temp cleanup & isolation
│   │   │   ├── job_manager.py    # Thread-safe in-memory async job engine
│   │   │   ├── media_validator.py# Magic bytes, dimension, and duration checks
│   │   │   ├── video_extractor.py# 32-frame uniform extractor + Base64 thumbnail
│   │   │   └── video_aggregator.py# Fail-closed aggregation gate (BLOCKED status)
│   │   └── utils/
│   │       ├── gpu_lock.py       # Inter-process GPU filelock
│   │       ├── logger.py         # Structured logging
│   │       └── subprocess_runner.py # Safe async subprocess runner
│   ├── requirements.txt
│   └── .env.example
├── model_worker/
│   ├── base.py                   # Abstract worker base class
│   ├── mock_worker.py            # Deterministic mock worker for local dev
│   ├── live_worker.py            # Live GPU coordinator
│   ├── router_client.py          # EfficientNet-B0 router
│   ├── expert_client.py          # Expert dispatch & calibrator integration
│   ├── x2dfd_client.py           # WFS prompt builder & LLaVA LoRA client
│   ├── x2dfd_worker_cli.py       # Subprocess CLI bridge for x2python
│   └── worker_factory.py         # Production guard factory
├── tests/                        # 41 unit and contract tests (pytest)
├── SOURCE_AUDIT.md               # Empirical research source audit findings
├── VERIFICATION_MATRIX.md        # Readiness matrix for research components
├── API_CONTRACT.md               # Full API JSON specifications
└── RUNBOOK.md                    # Server deployment & operations guide
```

---

## Quickstart (Development Mode)

1. **Install Python dependencies**:
   ```bash
   pip install -r web/backend/requirements.txt
   ```

2. **Run Backend with Mock Worker**:
   ```bash
   cd web/backend
   uvicorn app.main:app --reload --port 8000
   ```

3. **Run Full Test Suite**:
   ```bash
   python -m pytest web/tests -v
   ```

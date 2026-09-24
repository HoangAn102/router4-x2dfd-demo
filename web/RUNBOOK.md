# Operational Runbook: Router4 + X²-DFD Web Service

This runbook guides operators and developers on configuring, running, monitoring, and debugging the Router4 + X²-DFD detection service.

---

## 1. Operating Modes

The service operates in one of two strictly isolated modes set via `RUN_MODE` in `web/backend/.env`:

| Mode | Environment | Model Execution | Intended Use |
| :--- | :--- | :--- | :--- |
| `mock` | Local Dev / Laptop | Deterministic SHA256 simulation | Frontend UI design, contract testing, CI/CD |
| `live` | GPU Linux Server | PyTorch EfficientNet + Conda subprocs + LLaVA | Production inference on real images/videos |

> [!CAUTION]
> **Production Guard Active**: When `RUN_MODE=live`, the service will NEVER fallback to mock mode if weights or environments are missing. It will immediately fail-closed and throw an error to prevent scientific misattribution.

---

## 2. Server Environment Setup (`live` Mode)

### 2.1 Hardware Requirements
- **GPU**: NVIDIA GPU with $\ge 24\text{GB}$ VRAM (e.g. RTX 3090, RTX 4090, A100).
- **RAM**: $\ge 32\text{GB}$ System RAM.
- **Disk**: $\ge 50\text{GB}$ NVMe SSD space.

### 2.2 Conda Runtime Environments
The server requires the following conda environments created as verified in Sprint 0:

1. `x2python`: Contains PyTorch, transformers, LLaVA dependencies.
2. `freqpython`: ResNet-50 DFFreq inference environment.
3. `texpython`: Gram-Net ResNet-18 inference environment.

Verify wrappers using:
```bash
/home/user/miniconda3/envs/x2dfd/bin/python --version
/home/user/miniconda3/envs/dffreq/bin/python --version
/home/user/miniconda3/envs/gramnet/bin/python --version
```

### 2.3 Checkpoint Paths (`web/backend/.env`)
```bash
RUN_MODE=live
ROUTER_CHECKPOINT=/path/to/weights/router4/best.pt
CALIBRATORS_PATH=/path/to/weights/calibrators/calibrators_FINAL_CLEAN.joblib
ROUTER4_LORA_DIR=/path/to/weights/x2dfd_lora/router4_lora
BASE_LLAVA_DIR=/path/to/weights/base/llava-v1.5-7b

X2PYTHON_BIN=/home/user/miniconda3/envs/x2dfd/bin/python
FREQPYTHON_BIN=/home/user/miniconda3/envs/dffreq/bin/python
TEXPYTHON_BIN=/home/user/miniconda3/envs/gramnet/bin/python
```

---

## 3. Starting the Backend Server

Always run Uvicorn with **strictly 1 worker** to prevent multiple processes from contending for GPU memory:

```bash
cd web/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Readiness Check
Confirm that the live models are loaded and ready:
```bash
curl http://127.0.0.1:8000/api/v1/ready
```

---

## 4. Troubleshooting & OOM Recovery

### 4.1 Subprocess Timeout (`SubprocessTimeoutError`)
- Subprocesses that exceed the configured timeouts (60s for full image, 45s per frame) are automatically killed with `SIGKILL` on Linux.
- Check GPU utilization with `nvidia-smi`.

### 4.2 PyTorch CUDA Out-Of-Memory (OOM)
- Ensure no lingering Python processes are holding GPU VRAM:
  ```bash
  fuser -v /dev/nvidia*
  kill -9 <PID>
  ```
- The backend uses `GpuLock` (filelock on `GPU_LOCK_FILE`) to enforce single-concurrency execution on the GPU.

### 4.3 Stale Temporary Directories
- In normal operation, all uploaded files and extracted video frames are cleaned up in `finally` blocks.
- If the server crashed abruptly, run the cleanup sweep:
  ```python
  from web.backend.app.services.file_manager import cleanup_stale_temp_dirs
  cleanup_stale_temp_dirs(max_age_seconds=600)
  ```
  This is also run automatically during FastAPI startup lifespan.

---

## 5. Automated Test Suite Execution

Run all unit and integration tests:
```bash
python -m pytest web/tests -v
```
All 41 tests must pass before deployment.

# Router4 X2DFD Demo

Snapshot of source code, expert modules, and runtime context for the Router4-X2DFD deepfake detection demo.

## Overview

This repository contains the snapshot of all components required to run and evaluate the Router4-X2DFD multi-expert framework:

- **X2DFD**: Multimodal explainable deepfake detection pipeline using LLaVA-v1.5 and specialized forensic LoRA adapters.
- **Frequency Expert (DFFreq)**: Frequency-domain artifact analysis for face manipulation detection.
- **Texture Expert (Global Texture Enhancement)**: High-frequency texture and boundary artifact detection.
- **Router & MoE Integration**: Dynamic expert routing and ensemble cache generation for cross-dataset evaluation.
- **Runtime Wrappers & Benchmark**: Clean environment scripts and benchmark evaluation setups.

## Directory Structure

```
├── runtime_context/
│   ├── environment/             # Python environment freeze files for all experts
│   ├── artifact_manifest.tsv    # Manifest of model weights, checkpoints, and paths
│   └── source_roots.tsv         # Original server paths and references
├── source_snapshot/
│   ├── projects/
│   │   ├── X2DFD/               # X2DFD training, evaluation, and inference code
│   │   ├── DFFreq-main/         # Frequency-based detection expert
│   │   ├── Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild/ # Texture expert
│   │   └── x2dfd_router_integration/ # MoE cache and integration configs
│   ├── final_clean_integration/ # Benchmark configs and PEFT compatibility
│   ├── runtime_wrappers/        # Execution wrappers for different environments
│   └── server_wrappers/         # Server-side inference and verification runners
└── README.md
```

## Deployed configuration and evaluation (31/10/2026, tag `final-interim`)

- Live web = lineage **(b) parity v2**: Router v2 + isotonic calibrator v2 + LoRA `router4_x2dfd_FINAL_CLEAN`, LLaVA prompt token-identical to SFT (conv `v1`), teacher-forced real/fake read-out, `DECISION_THRESHOLD = 0.74908721` (chosen on DEV only).
- The full-test-split evaluation (R6) was pre-registered in `docs/PREREG_R6.md`. GATE R6 (V3 vs interim) was **not met**, so the V3 lineage is not deployed. Metrics, leakage check, independent recomputation, deploy smoke tests and limitations are in the server report `eval_runs/final_audit/REPORT.md`.
- Tests: `python -m pytest -q web/tests web/backend/tests`.

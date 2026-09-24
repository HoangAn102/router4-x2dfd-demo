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

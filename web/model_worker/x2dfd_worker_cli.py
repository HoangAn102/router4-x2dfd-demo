#!/usr/bin/env python3
"""
CLI wrapper script for X²-DFD LLaVA + LoRA inference.
Invoked via SubprocessRunner using the x2python runtime.
Outputs a clean JSON string to stdout containing real_score, fake_score, and answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="X2DFD Worker CLI for Web Server")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Full WFS prompt text")
    parser.add_argument("--lora-dir", required=True, help="Path to LoRA weights directory")
    parser.add_argument("--base-model", required=True, help="Path to Base LLaVA model directory")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max tokens to generate")
    args = parser.parse_args()

    # Locate and add X2DFD source directory to sys.path
    repo_root = Path(__file__).resolve().parent.parent.parent
    x2dfd_dir = repo_root / "source_snapshot" / "projects" / "X2DFD"
    if str(x2dfd_dir) not in sys.path:
        sys.path.insert(0, str(x2dfd_dir))

    try:
        from utils.lora_inference import single_image_infer_with_scores  # type: ignore

        result = single_image_infer_with_scores(
            image_path=args.image,
            question=args.prompt,
            model_path=args.lora_dir,
            model_base=args.base_model,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
        )

        output = {
            "real_score": result.get("real_score"),
            "fake_score": result.get("fake_score"),
            "answer": result.get("answer", ""),
        }
        print(json.dumps(output))
        sys.exit(0)

    except Exception as e:
        sys.stderr.write(f"X2DFD inference failed: {type(e).__name__}: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

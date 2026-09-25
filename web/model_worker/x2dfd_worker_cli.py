#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--prompt",
        required=True,
    )

    parser.add_argument(
        "--lora-dir",
        required=True,
    )

    parser.add_argument(
        "--base-model",
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # SAFEVISION_WORKER_ABSOLUTE_MEDIA_PATH_V1
    # Never depend on subprocess cwd for uploaded media.
    # --------------------------------------------------------
    image_path = (
        Path(args.image)
        .expanduser()
        .resolve(strict=True)
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image missing: {image_path}"
        )

    x2root = Path(
        os.environ.get(
            "X2DFD_PROJECT_ROOT",
            "/home/aiotlab/hoangan/projects/X2DFD",
        )
    )

    if not x2root.exists():
        raise RuntimeError(
            f"X2DFD root missing: {x2root}"
        )

    sys.path.insert(
        0,
        str(x2root),
    )

    try:
        from utils.lora_inference import (
            single_image_infer_with_scores,
        )

        result = (
            single_image_infer_with_scores(
                image_path=str(image_path),
                question=args.prompt,
                model_path=args.lora_dir,
                model_base=args.base_model,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
            )
        )

        payload = {
            "real_score":
                result.get("real_score"),

            "fake_score":
                result.get("fake_score"),

            "answer":
                result.get("answer", ""),
        }

        # Last stdout line is machine-readable JSON.
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
        )

    except Exception as e:
        sys.stderr.write(
            "X2DFD inference failed: "
            f"{type(e).__name__}: {e}\n"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

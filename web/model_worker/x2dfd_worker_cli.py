#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback

from pathlib import Path


DEFAULT_X2ROOT = "/home/aiotlab/hoangan/projects/X2DFD"


def _load_infer_function():

    x2root = Path(
        os.environ.get(
            "X2DFD_PROJECT_ROOT",
            DEFAULT_X2ROOT,
        )
    ).resolve()

    if not x2root.exists():
        raise RuntimeError(
            f"X2DFD root missing: {x2root}"
        )

    if str(x2root) not in sys.path:
        sys.path.insert(0, str(x2root))

    from utils.lora_inference import (
        single_image_infer_with_scores,
    )

    return single_image_infer_with_scores


def _run_one(
    infer_fn,
    *,
    image,
    prompt,
    lora_dir,
    base_model,
    max_new_tokens,
):

    image_path = (
        Path(image)
        .expanduser()
        .resolve(strict=True)
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image missing: {image_path}"
        )

    # Keep stdout exclusively for JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):

        result = infer_fn(
            image_path=str(image_path),
            question=prompt,
            model_path=lora_dir,
            model_base=base_model,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )

    return {
        "ok": True,
        "real_score": result.get("real_score"),
        "fake_score": result.get("fake_score"),
        "answer": result.get("answer", ""),
    }


def server_mode(args):

    infer_fn = _load_infer_function()

    protocol_out = sys.stdout

    print(
        "[X2DFD_SERVER] READY",
        file=sys.stderr,
        flush=True,
    )

    for raw in sys.stdin:

        raw = raw.strip()

        if not raw:
            continue

        try:
            req = json.loads(raw)

            response = _run_one(
                infer_fn,

                image=req["image"],
                prompt=req["prompt"],

                lora_dir=(
                    req.get("lora_dir")
                    or args.lora_dir
                ),

                base_model=(
                    req.get("base_model")
                    or args.base_model
                ),

                max_new_tokens=int(
                    req.get(
                        "max_new_tokens",
                        args.max_new_tokens,
                    )
                ),
            )

        except Exception as e:

            traceback.print_exc(
                file=sys.stderr
            )

            response = {
                "ok": False,
                "error_type": type(e).__name__,
                "error": str(e),
            }

        protocol_out.write(
            json.dumps(
                response,
                ensure_ascii=False,
            ) + "\n"
        )

        protocol_out.flush()


def oneshot_mode(args):

    infer_fn = _load_infer_function()

    try:
        result = _run_one(
            infer_fn,
            image=args.image,
            prompt=args.prompt,
            lora_dir=args.lora_dir,
            base_model=args.base_model,
            max_new_tokens=args.max_new_tokens,
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
            )
        )

    except Exception as e:

        print(
            f"X2DFD inference failed: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )

        raise SystemExit(1)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--server",
        action="store_true",
    )

    parser.add_argument("--image")
    parser.add_argument("--prompt")

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
        default=96,
    )

    args = parser.parse_args()

    if args.server:
        server_mode(args)
        return

    if not args.image or not args.prompt:
        parser.error(
            "--image and --prompt required "
            "outside --server mode"
        )

    oneshot_mode(args)


if __name__ == "__main__":
    main()

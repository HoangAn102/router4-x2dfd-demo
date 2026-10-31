"""R6 (full), system (a): the X2DFD author's questions for every frame, built by the author's own runner code.

eval.infer.runner (multi-expert config eval/configs/infer_config.yaml) = _build_conversations_multi (Blending
SwinV2-B + Diffusion ours-sync scores, author's formatting) -> lora_infer_conversation_items ->
single_image_infer_with_scores per image. This script runs the first half unchanged and stores the questions;
r6_batched.py --system a scores them (verified against single_image_infer_with_scores and against the runner's own
PHASE 4 output on the overlapping frames).

    x2python tools/v3/r6_author_questions.py --frames frames_full.csv --out a_questions.csv
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

X2 = Path("/home/aiotlab/hoangan/projects/X2DFD")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="eval/configs/infer_config.yaml")
    ap.add_argument("--chunk", type=int, default=5000)
    a = ap.parse_args()
    frames, out = a.frames.resolve(), a.out.resolve()
    os.chdir(X2)  # the config's weight paths are relative to the X2DFD root, as when running the runner
    sys.path.insert(0, str(X2))
    from eval.infer import runner as R

    cfg = R._load_yaml(a.config)
    paths = pd.read_csv(frames).image_path.drop_duplicates().tolist()
    done = set(pd.read_csv(out).image_path) if out.exists() else set()
    todo = [p for p in paths if p not in done]
    for k in range(0, len(todo), a.chunk):
        part = todo[k:k + a.chunk]
        items = R._build_conversations_multi({"Description": "/", "images": [{"image_path": p} for p in part]},
                                             experts_cfg=cfg["weak_supplies"])
        assert [it["image"] for it in items] == [os.path.abspath(p) for p in part]
        pd.DataFrame(dict(image_path=part, question=[it["conversations"][0]["value"] for it in items])).to_csv(
            out, mode="a", header=not out.exists(), index=False)
        print(f"{k + len(part)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()

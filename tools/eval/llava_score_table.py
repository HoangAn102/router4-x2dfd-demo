"""Exhaustive score table of the LoRA under the train-parity prompt.

After the parity fix (web/model_worker/x2dfd_worker_cli.py) the LLaVA input is fully determined by
(selected alias, calibrated score rounded to 3 decimals): the SFT prompts carry no <image> token. So the
whole X2DFD stage is a function of 4 x 1001 prompts. This scores every one of them with the live worker
functions (same model loader, conv template, logit read-out) and writes
    alias, score_3dp, fake_prob, real_prob, label_step, answer

    x2python tools/eval/llava_score_table.py --out DIR [--step 1]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "web" / "model_worker"))

import x2dfd_worker_cli as W  # noqa: E402

LORA = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/05_lora/router4_x2dfd_FINAL_CLEAN"
BASE = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"
EXPERTS = ["blending", "diffusion", "frequency", "texture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step", type=int, default=1, help="score step in thousandths")
    ap.add_argument("--lora", default=LORA)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "llava_score_table.csv"
    done = set()
    if path.exists():
        done = {(r["alias"], r["score_3dp"]) for r in csv.DictReader(open(path))}
    new = not path.exists()
    t0 = time.time()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["alias", "score_3dp", "fake_prob", "real_prob", "label_step", "answer"])
        if new:
            w.writeheader()
        for e in EXPERTS:
            for k in range(0, 1001, a.step):
                s = f"{k / 1000:.3f}"
                if (e, s) in done:
                    continue
                q = f"\nIs this image real or fake? And the {e} score is {s}."
                r = W.score_with_train_prompt(q, model_path=a.lora, model_base=BASE, max_new_tokens=4)
                w.writerow(dict(alias=e, score_3dp=s, fake_prob=r["fake_score"], real_prob=r["real_score"],
                                label_step=r["label_step"], answer=r["answer"]))
            fh.flush()
            print(e, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

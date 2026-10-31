"""R6 (full): image-conditioned LLaVA systems on frames_full.csv (resumable, one CSV per system).

  --system V3       Router V3 + calibrator V3 + LoRA V3; "<image>\\n…" prompt, image fed as in training, through
                    the live worker functions (web/model_worker/x2dfd_worker_cli.py)
  --system bold     pre-audit live configuration: Router v2 choice, calibrator FINAL_CLEAN, LoRA FINAL_CLEAN via
                    X2DFD utils.lora_inference.single_image_infer_with_scores("<image>\\n…") (conv llava_v0 + image)
  --system V3_gray  V3 prompts with a constant mid-grey 256x256 image. The input then depends only on
                    (alias, score .3f), so all 4 x 1001 prompts are scored once and looked up (exactly equivalent).
  --system author_json   write eval.infer.runner input JSONs (X2DFD original, run separately) per dataset

    x2python tools/v3/r6_llava.py --frames frames_full.csv --system V3 --lora-v3 DIR --out results_V3.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "web/model_worker"))
sys.path.insert(0, "/home/aiotlab/hoangan/projects/X2DFD")

BASE = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"
LORA_FC = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/05_lora/router4_x2dfd_FINAL_CLEAN"
EXPERTS = ["blending", "diffusion", "frequency", "texture"]


def q(expert, cal):
    return f"<image>\nIs this image real or fake? And the {expert} score is {cal:.3f}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--system", required=True, choices=["V3", "bold", "V3_gray", "author_json"])
    ap.add_argument("--lora-v3")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", default="test,dev")
    a = ap.parse_args()
    df = pd.read_csv(a.frames)
    df = df[df.split.isin(a.splits.split(","))]

    if a.system == "author_json":
        a.out.mkdir(parents=True, exist_ok=True)
        for ds, g in df.groupby("dataset"):
            json.dump({"Description": "/", "images": [{"image_path": p} for p in g.image_path]},
                      open(a.out / f"author_in_{ds}.json", "w"))
            print(ds, len(g))
        return

    if a.system == "V3_gray":
        import x2dfd_worker_cli as W
        from PIL import Image
        grey = a.out.with_suffix(".grey.png")
        Image.new("RGB", (256, 256), (128, 128, 128)).save(grey)
        table = {}
        for e in EXPERTS:
            for k in range(1001):
                s = k / 1000
                r = W.score_with_train_prompt(q(e, s), model_path=a.lora_v3, model_base=BASE, max_new_tokens=4,
                                              image_path=str(grey))
                table[(e, f"{s:.3f}")] = r["fake_score"]
                if k % 200 == 0:
                    print(e, k, flush=True)  # heartbeat for the R6 watchdog
        pd.DataFrame([dict(alias=e, score_3dp=s, fake_prob=p) for (e, s), p in table.items()]).to_csv(
            a.out.with_suffix(".table.csv"), index=False)
        df["V3_gray_fake"] = [table[(e, f"{c:.3f}")] for e, c in zip(df.V3_expert, df.V3_cal)]
        df[["image_path", "V3_gray_fake"]].to_csv(a.out, index=False)
        return

    done = set(pd.read_csv(a.out).image_path) if a.out.exists() else set()
    new = not a.out.exists()
    if a.system == "V3":
        import x2dfd_worker_cli as W
    else:
        from utils import lora_inference as LI
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_path", f"{a.system}_fake", f"{a.system}_step"])
        if new:
            w.writeheader()
        for i, r in enumerate(df[~df.image_path.isin(done)].itertuples()):
            if a.system == "V3":
                res = W.score_with_train_prompt(q(r.V3_expert, r.V3_cal), model_path=a.lora_v3, model_base=BASE,
                                                max_new_tokens=4, image_path=r.image_path)
                w.writerow({"image_path": r.image_path, "V3_fake": res["fake_score"], "V3_step": res["label_step"]})
            else:
                res = LI.single_image_infer_with_scores(image_path=r.image_path, question=q(r.b_expert, r.bold_cal),
                                                        model_path=LORA_FC, model_base=BASE, max_new_tokens=4)
                w.writerow({"image_path": r.image_path, "bold_fake": res["fake_score"], "bold_step": ""})
            if i % 200 == 0:
                fh.flush()
                print(i, flush=True)


if __name__ == "__main__":
    main()

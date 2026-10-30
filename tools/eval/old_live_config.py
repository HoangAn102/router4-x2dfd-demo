"""The configuration the web ran before the audit (main 1e1c06e), on the PHASE 4 benchmark frames.

Router4 v2 top-1 and the selected expert's raw score are taken from frames.csv (same code, same frames);
the differences are exactly the ones the audit fixed:
  calibrator  calibrators_FINAL_CLEAN.joblib
  LLaVA call  X2DFD utils.lora_inference.single_image_infer_with_scores(question="<image>\\nIs this image real
              or fake? And the {alias} score is {cal:.3f}.") -> conv llava_v0 + <im_start><image><im_end> + image
Writes c_fake per frame to old_live.csv (resumable). Needs ~15 GB free VRAM.

    x2python tools/eval/old_live_config.py --frames frames.csv --out old_live.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import joblib

sys.path.insert(0, "/home/aiotlab/hoangan/projects/X2DFD")
from utils import lora_inference as LI  # noqa: E402

LORA = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/05_lora/router4_x2dfd_FINAL_CLEAN"
BASE = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"
CAL = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/01_teacher/calibrators_FINAL_CLEAN.joblib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cal = joblib.load(CAL)["experts"]
    done = {r["path"] for r in csv.DictReader(open(a.out))} if a.out.exists() else set()
    new = not a.out.exists()
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "c_cal", "c_fake", "c_answer"])
        if new:
            w.writeheader()
        for i, r in enumerate(csv.DictReader(open(a.frames))):
            if r["path"] in done:
                continue
            c = float(cal[r["b_expert"]].predict([float(r["b_raw"])])[0])
            q = f"<image>\nIs this image real or fake? And the {r['b_expert'].lower()} score is {c:.3f}."
            res = LI.single_image_infer_with_scores(image_path=r["path"], question=q, model_path=LORA,
                                                    model_base=BASE, max_new_tokens=4)
            w.writerow(dict(path=r["path"], c_cal=c, c_fake=res["fake_score"], c_answer=res["answer"]))
            if i % 100 == 0:
                fh.flush()
                print(i, flush=True)


if __name__ == "__main__":
    main()

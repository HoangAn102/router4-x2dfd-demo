"""TRAIN_DIFFUSION_PLAN D2.2: teacher for Router4 v3 = recipe of projects/router_v2/prepare_teacher_v2.py::add_teacher.

  utility_e = cal_e if label==fake else 1-cal_e ; teacher_soft = softmax(U / T), T = 0.20 (v2's fixed temperature);
  teacher_hard = argmax U.
TRAIN = train_teacher_v2.csv (its OOF cal_* kept, cal_diffusion replaced by the D1 OOF values)
      + DFF train (cal_diffusion = D1 OOF; blending/frequency/texture = unchanged v2 calibrators, which never saw DFF)
VAL   = val_teacher_v2.csv (cal_diffusion re-predicted with the D1 final calibrator) + DFF val (final calibrators).

    python tools/diffusion/build_teacher_v3.py --run DIR --dff-train DFF_train_scores.csv --dff-val DFF_val_scores.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

V2 = Path("/home/aiotlab/hoangan/outputs/router_teacher_v2")
EXPERTS = ["blending", "diffusion", "frequency", "texture"]
T = 0.20
KEEP = ["image_path", "label", "dataset", "generator", "split"] + [f"score_{e}" for e in EXPERTS]


def add_teacher(df):
    y = df["label"].to_numpy(int)
    U = np.stack([np.where(y == 1, df[f"cal_{e}"], 1.0 - df[f"cal_{e}"]) for e in EXPERTS], axis=1)
    for i, e in enumerate(EXPERTS):
        df[f"utility_{e}"] = U[:, i]
    hard = U.argmax(1)
    df["teacher_hard_idx"] = hard
    df["teacher_hard"] = [EXPERTS[i] for i in hard]
    df["teacher_best_utility"] = U[np.arange(len(df)), hard]
    z = U / T
    z -= z.max(1, keepdims=True)
    s = np.exp(z)
    s /= s.sum(1, keepdims=True)
    for i, e in enumerate(EXPERTS):
        df[f"teacher_soft_{e}"] = s[:, i]
    df["teacher_temperature"] = T
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dff-train", type=Path, required=True)
    ap.add_argument("--dff-val", type=Path, required=True)
    a = ap.parse_args()
    cal = joblib.load(a.run / "01_teacher/calibrators_v2_diffrefit.joblib")["calibrators"]
    oof = pd.read_csv(a.run / "01_teacher/diffusion_oof.csv").set_index("image_path")["cal_diffusion_oof"]

    tr_old = pd.read_csv(V2 / "train_teacher_v2.csv")
    tr_old = tr_old[KEEP + [f"cal_{e}" for e in EXPERTS]]
    tr_old["cal_diffusion"] = oof.loc[tr_old.image_path].to_numpy()
    tr_new = pd.read_csv(a.dff_train)
    tr_new["dataset"], tr_new["split"] = "DeepFakeFace", "train"
    for e in ("blending", "frequency", "texture"):
        tr_new[f"cal_{e}"] = cal[e].predict(tr_new[f"score_{e}"].to_numpy(float))
    tr_new["cal_diffusion"] = oof.loc[tr_new.image_path].to_numpy()
    train = add_teacher(pd.concat([tr_old, tr_new[tr_old.columns]], ignore_index=True))

    va_old = pd.read_csv(V2 / "val_teacher_v2.csv")
    va_old = va_old[KEEP + [f"cal_{e}" for e in EXPERTS]]
    va_old["cal_diffusion"] = cal["diffusion"].predict(va_old["score_diffusion"].to_numpy(float))
    va_new = pd.read_csv(a.dff_val)
    va_new["dataset"], va_new["split"] = "DeepFakeFace", "val"
    for e in EXPERTS:
        va_new[f"cal_{e}"] = cal[e].predict(va_new[f"score_{e}"].to_numpy(float))
    val = add_teacher(pd.concat([va_old, va_new[va_old.columns]], ignore_index=True))

    out = a.run / "01_teacher"
    train.to_csv(out / "train_teacher_v3.csv", index=False)
    val.to_csv(out / "val_teacher_v3.csv", index=False)
    rep = {s: {"rows": int(len(d)),
               "by_dataset": d["dataset"].value_counts().to_dict(),
               "hard": {ds: g["teacher_hard"].value_counts().to_dict() for ds, g in d.groupby("dataset")}}
           for s, d in (("train", train), ("val", val))}
    rep["temperature"] = T
    (out / "TEACHER_V3_SUMMARY.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

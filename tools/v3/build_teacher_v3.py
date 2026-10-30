"""ROOT_FIX R3: teacher V3, recipe 0.3 (ROUTER4_FINAL_CLEAN_RETRAIN_20260919/scripts/rebuild_clean_teacher.py).

  utility_e = clip(cal_e if fake else 1-cal_e, 1e-5, 1-1e-5) on OOF cal V3 (R2)
  T: recovered from the old (v2) teacher by MSE over 160 log-spaced candidates in [0.01, 2.0], fallback 0.10.
     rebuild_clean_teacher.py looked for columns teacher_soft0..3, which train_teacher_v2.csv does not have
     (it names them teacher_soft_<expert>), so it silently fell back to 0.10; both spellings are accepted here,
     which is what the recovery step intended.
  soft = softmax(U / T), hard = argmax U.
Also reports the share of TRAIN rows whose hard label differs from the D2 teacher (ROUTER4_DIFF_RETRAIN_20261030).

    python tools/v3/build_teacher_v3.py --v3 /home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPERTS = ["blending", "diffusion", "frequency", "texture"]
OLD_TRAIN = Path("/home/aiotlab/hoangan/outputs/router_teacher_v2/train_teacher_v2.csv")
D2_TRAIN = Path("/home/aiotlab/hoangan/outputs/ROUTER4_DIFF_RETRAIN_20261030/01_teacher/train_teacher_v3.csv")


def recover_T():
    old = pd.read_csv(OLD_TRAIN, nrows=25000, low_memory=False)
    for soft_cols in ([f"teacher_soft{i}" for i in range(4)], [f"teacher_soft_{e}" for e in EXPERTS]):
        util_cols = [f"utility_{e}" for e in EXPERTS]
        if all(c in old.columns for c in soft_cols + util_cols):
            U, S = old[util_cols].to_numpy(float), old[soft_cols].to_numpy(float)
            good = np.isfinite(U).all(1) & np.isfinite(S).all(1)
            U, S = U[good], S[good]
            best = (1e99, None)
            for t in np.logspace(np.log10(.01), np.log10(2.0), 160):
                Z = U / t
                Z -= Z.max(1, keepdims=True)
                P = np.exp(Z)
                P /= P.sum(1, keepdims=True)
                mse = float(np.mean((P - S) ** 2))
                if mse < best[0]:
                    best = (mse, float(t))
            return best[1], f"recovered_from_old_teacher({soft_cols[0]}..)", best[0]
    return 0.10, "fallback_0.10", None


def build(df, T):
    y = df.label.to_numpy(int)
    U = np.column_stack([np.clip(np.where(y == 1, df[f"cal_{e}"], 1 - df[f"cal_{e}"]), 1e-5, 1 - 1e-5)
                         for e in EXPERTS])
    for i, e in enumerate(EXPERTS):
        df[f"utility_{e}"] = U[:, i]
    hard = U.argmax(1)
    df["teacher_hard_idx"] = hard
    df["teacher_hard"] = [EXPERTS[i] for i in hard]
    df["teacher_best_expert"] = df["teacher_hard"]
    df["teacher_best_utility"] = U.max(1)
    Z = U / T
    Z -= Z.max(1, keepdims=True)
    soft = np.exp(Z)
    soft /= soft.sum(1, keepdims=True)
    for i, e in enumerate(EXPERTS):
        df[f"teacher_soft{i}"] = soft[:, i]
        df[f"teacher_soft_{e}"] = soft[:, i]
    df["teacher_temperature"] = T
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", type=Path, required=True)
    a = ap.parse_args()
    cal_dir, out = a.v3 / "01_calibrators", a.v3 / "02_teacher"
    out.mkdir(parents=True, exist_ok=True)
    T, src, mse = recover_T()
    rep = {"temperature": T, "temperature_source": src, "temperature_mse": mse}
    for s in ("train", "val"):  # test stays untouched until R6/R7
        d = build(pd.read_csv(cal_dir / f"scores_V3_{s}.csv"), T)
        d.to_csv(out / f"{s}_teacher_V3.csv", index=False)
        rep[s] = {"rows": int(len(d)),
                  "hard": {ds: g.teacher_hard.value_counts().to_dict() for ds, g in d.groupby("dataset")}}
        if s == "train":
            d2 = pd.read_csv(D2_TRAIN, usecols=["image_path", "teacher_hard"]).set_index("image_path").teacher_hard
            m = d.image_path.isin(d2.index)
            diff = d.loc[m, "teacher_hard"].to_numpy() != d2.loc[d.loc[m, "image_path"]].to_numpy()
            rep["hard_label_diff_vs_D2"] = float(diff.mean())
            rep["hard_label_diff_vs_D2_by_dataset"] = {
                ds: float(diff[(d.loc[m, "dataset"] == ds).to_numpy()].mean()) for ds in d.loc[m, "dataset"].unique()}
    (out / "R3_teacher_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

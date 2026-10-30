"""TRAIN_DIFFUSION_PLAN D1: refit ONLY the Diffusion calibrator, recipe of projects/router_v2/prepare_teacher_v2.py.

  IsotonicRegression(y_min=1e-5, y_max=1-1e-5, out_of_bounds="clip"), sample weights = equal TOTAL weight per
  dataset (domain_weights), 5-fold OOF on TRAIN (v2 folds kept for old rows; DeepFakeFace rows get
  StratifiedGroupKFold by IMDB-WIKI person id, seed 42 — v2 uses group folds whenever group metadata exists),
  final model fitted on all TRAIN.
  TRAIN = router_teacher_v2/train_teacher_v2.csv + DFF train (score_diffusion from tools/diffusion/score_experts.py).
Blending/Frequency/Texture calibrator objects are copied unchanged from the v2 bundle (verified identical output).

    python tools/diffusion/fit_diffusion_calibrator.py --dff-train DFF_train_scores.csv --out-dir DIR
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

V2_TRAIN = Path("/home/aiotlab/hoangan/outputs/router_teacher_v2/train_teacher_v2.csv")
V2_BUNDLE = Path("/home/aiotlab/hoangan/outputs/router_teacher_v2/calibrators_train_only.joblib")
N_FOLDS, SEED = 5, 42


def domain_weights(df):
    counts = df["dataset"].value_counts().to_dict()
    mapping = {d: len(df) / (len(counts) * c) for d, c in counts.items()}
    return df["dataset"].map(mapping).to_numpy(float)


def iso():
    return IsotonicRegression(y_min=1e-5, y_max=1 - 1e-5, out_of_bounds="clip")


def load_train(dff_train):
    old = pd.read_csv(V2_TRAIN, usecols=["image_path", "label", "dataset", "generator", "cal_fold",
                                         "score_blending", "score_diffusion", "score_frequency", "score_texture"])
    new = pd.read_csv(dff_train)
    new["dataset"] = "DeepFakeFace"
    new["cal_fold"] = -1
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    strata = new["generator"].astype(str) + "__y" + new["label"].astype(str)
    for fold, (_, hold) in enumerate(sgk.split(np.zeros(len(new)), strata, new["person_id"].astype(str))):
        new.loc[new.index[hold], "cal_fold"] = fold
    cols = list(old.columns) + ["person_id"]
    return pd.concat([old, new[[c for c in cols if c in new.columns]]], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dff-train", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tr = load_train(a.dff_train)
    x, y = tr["score_diffusion"].to_numpy(float), tr["label"].to_numpy(int)
    oof = np.full(len(tr), np.nan)
    for f in range(N_FOLDS):
        fit, hold = tr["cal_fold"] != f, tr["cal_fold"] == f
        m = iso().fit(x[fit], y[fit], sample_weight=domain_weights(tr[fit]))
        oof[hold.to_numpy()] = m.predict(x[hold.to_numpy()])
    assert not np.isnan(oof).any()
    final = iso().fit(x, y, sample_weight=domain_weights(tr))

    bundle = joblib.load(V2_BUNDLE)
    old_diff = bundle["calibrators"]["diffusion"]
    probe = np.concatenate([np.linspace(0, 1, 2001), x[:: max(1, len(x) // 5000)]])
    keep = {e: bundle["calibrators"][e] for e in ("blending", "frequency", "texture")}
    bundle["calibrators"] = dict(keep, diffusion=final)
    bundle["diffusion_refit"] = dict(
        recipe="prepare_teacher_v2.py isotonic y_min=1e-5 domain_weights, 5-fold OOF, final on all TRAIN",
        train=[str(V2_TRAIN), str(a.dff_train)], rows=int(len(tr)),
        rows_by_dataset=tr["dataset"].value_counts().to_dict())
    out = a.out_dir / "calibrators_v2_diffrefit.joblib"
    joblib.dump(bundle, out)
    reloaded = joblib.load(out)["calibrators"]
    same = {e: bool(np.array_equal(reloaded[e].predict(probe), keep[e].predict(probe))) for e in keep}
    assert all(same.values()), same

    tr["cal_diffusion_oof"] = oof
    tr[["image_path", "dataset", "label", "cal_fold", "score_diffusion", "cal_diffusion_oof"]].to_csv(
        a.out_dir / "diffusion_oof.csv", index=False)
    grid = np.array([1e-4, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999])
    rep = dict(out=str(out), other_calibrators_identical=same,
               old_diffusion_on_grid=dict(zip(map(str, grid), old_diff.predict(grid).round(4).tolist())),
               new_diffusion_on_grid=dict(zip(map(str, grid), final.predict(grid).round(4).tolist())),
               n_levels_old=int(len(np.unique(old_diff.predict(probe)))),
               n_levels_new=int(len(np.unique(final.predict(probe)))),
               rows_by_dataset=tr["dataset"].value_counts().to_dict())
    (a.out_dir / "D1_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

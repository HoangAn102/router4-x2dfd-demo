"""ROOT_FIX R2: calibrator V3 for all 4 experts, recipe of projects/router_v2/prepare_teacher_v2.py.

  IsotonicRegression(y_min=1e-5, y_max=1-1e-5, out_of_bounds="clip"), sample weight = equal TOTAL weight per
  dataset, 5-fold OOF on TRAIN (v2 cal_fold for the v2 rows; DeepFakeFace: StratifiedGroupKFold by IMDB-WIKI
  person id, seed 42), final model on all TRAIN, outputs clipped to [1e-5, 1-1e-5].
  TRAIN = router_teacher_v2/train_teacher_v2.csv + DFF train; GANGen texture replaced by the live re-score (R1).
Writes:
  01_calibrators/calibrators_V3.joblib     {"experts": {name: IsotonicRegression}, ...meta}
  01_calibrators/scores_V3_{train,val,test}.csv   raw + cal_<e> (OOF on train, final model on val/test)

    python tools/v3/fit_calibrators_v3.py --v3 /home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

V2 = Path("/home/aiotlab/hoangan/outputs/router_teacher_v2")
DFF = Path("/home/aiotlab/hoangan/datasets/DeepFakeFace_DFF")
EXPERTS = ["blending", "diffusion", "frequency", "texture"]
N_FOLDS, SEED, EPS = 5, 42, 1e-5
COLS = ["image_path", "label", "dataset", "generator", "split", "cal_fold", "person_id"] + \
       [f"score_{e}" for e in EXPERTS]


def domain_weights(df):
    counts = df["dataset"].value_counts().to_dict()
    return df["dataset"].map({d: len(df) / (len(counts) * c) for d, c in counts.items()}).to_numpy(float)


def iso():
    return IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")


def load(split, texture_live):
    old = pd.read_csv(V2 / f"{split}_teacher_v2.csv", low_memory=False)
    old["split"] = split
    if "cal_fold" not in old:
        old["cal_fold"] = -1
    g = old.dataset == "GANGen"
    old.loc[g, "score_texture"] = texture_live.loc[old.loc[g, "image_path"]].to_numpy()
    new = pd.read_csv(DFF / f"DFF_{split}_scores.csv")
    new["dataset"], new["split"], new["cal_fold"] = "DeepFakeFace", split, -1
    if split == "train":
        sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        strata = new["generator"].astype(str) + "__y" + new["label"].astype(str)
        for f, (_, hold) in enumerate(sgk.split(np.zeros(len(new)), strata, new["person_id"].astype(str))):
            new.loc[new.index[hold], "cal_fold"] = f
    return pd.concat([old.reindex(columns=COLS), new.reindex(columns=COLS)], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", type=Path, required=True)
    a = ap.parse_args()
    out = a.v3 / "01_calibrators"
    out.mkdir(parents=True, exist_ok=True)
    tex = pd.read_csv(a.v3 / "00_scores/gangen_texture_live.csv").drop_duplicates("image_path")
    tex = tex.set_index("image_path").texture_live
    sets = {s: load(s, tex) for s in ("train", "val", "test")}
    tr = sets["train"]
    assert (tr.cal_fold >= 0).all()
    y = tr.label.to_numpy(int)
    models, rep = {}, {}
    for e in EXPERTS:
        x = tr[f"score_{e}"].to_numpy(float)
        assert np.isfinite(x).all(), e
        oof = np.full(len(tr), np.nan)
        for f in range(N_FOLDS):
            fit, hold = (tr.cal_fold != f).to_numpy(), (tr.cal_fold == f).to_numpy()
            oof[hold] = iso().fit(x[fit], y[fit], sample_weight=domain_weights(tr[fit])).predict(x[hold])
        models[e] = iso().fit(x, y, sample_weight=domain_weights(tr))
        tr[f"cal_{e}"] = np.clip(oof, EPS, 1 - EPS)
        for s in ("val", "test"):
            sets[s][f"cal_{e}"] = np.clip(models[e].predict(sets[s][f"score_{e}"].to_numpy(float)), EPS, 1 - EPS)
        from sklearn.metrics import roc_auc_score
        rep[e] = {f"{s}_{ds}_auc": float(roc_auc_score(g.label, g[f"score_{e}"]))
                  for s in ("train", "val") for ds, g in sets[s].groupby("dataset") if g.label.nunique() == 2}
        rep[e]["floor"] = float(models[e].predict([0.0])[0])
        rep[e]["levels"] = int(len(np.unique(models[e].predict(np.linspace(0, 1, 5001)))))
    bundle = {"experts": models,
              "protocol": "prepare_teacher_v2.py recipe: isotonic y_min=1e-5, equal per-dataset weight, 5-fold OOF, "
                          "final on all TRAIN; TRAIN = v2 train + DeepFakeFace train; GANGen texture re-scored live",
              "rows_by_dataset": tr.dataset.value_counts().to_dict()}
    joblib.dump(bundle, out / "calibrators_V3.joblib")
    for s, d in sets.items():
        d.to_csv(out / f"scores_V3_{s}.csv", index=False)
    rep["rows"] = {s: d.dataset.value_counts().to_dict() for s, d in sets.items()}
    (out / "R2_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

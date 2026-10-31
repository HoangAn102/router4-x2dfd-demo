"""PREREG §5: paired video-bootstrap CIs of every GATE R6 component (V3 − interim): frame AUC on FFPP/CDF/DFDCP/DFF_test
and FPR(real) @0.5 on all four test sets. Same resampling as stats.bootstrap_auc (1,000, default_rng(0), videos).
Decision stays on point estimates (PREREG); CIs are informational.

--v3-tf: post-hoc sensitivity, V3 rows scored in batch replaced by the single-image teacher-forced read-out of the web
worker (tf_check/tf_equivalence_existing_all59788.csv, column `new`). Labelled "thăm dò (post-hoc)".

    python tools/v3/r6_gate_ci.py --dir r6_full --exclude r6_full/leak_check/exclude_frames.csv [--v3-tf]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r6_report import load  # noqa: E402

TEST = ["FFPP", "CDF", "DFDCP", "DFF_test"]


def fpr(y, s):
    return float((s[y == 0] >= 0.5).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--v3-tf", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    df = load(a.dir)
    df = df[df.split == "test"]
    if a.exclude:
        df = df[~df.image_path.isin(set(pd.read_csv(a.exclude).image_path))]
    if a.v3_tf:
        t = pd.read_csv(a.dir / "tf_check/tf_equivalence_existing_all59788.csv").set_index("image_path").new
        m = df.image_path.isin(t.index)
        df.loc[m, "V3_fake"] = df.loc[m, "image_path"].map(t).to_numpy()
    out = {"post_hoc_v3_tf": a.v3_tf}
    for ds in TEST:
        g = df[df.dataset == ds].reset_index(drop=True)
        y, v3, it = g.label.to_numpy(int), g.V3_fake.to_numpy(float), g.interim_fake.to_numpy(float)
        vids = g.video.unique()
        idx = {v: np.flatnonzero(g.video.to_numpy() == v) for v in vids}
        rng = np.random.default_rng(0)
        da, df_ = [], []
        for _ in range(a.n_boot):
            rows = np.concatenate([idx[v] for v in rng.choice(vids, size=len(vids), replace=True)])
            yb = y[rows]
            da.append(roc_auc_score(yb, v3[rows]) - roc_auc_score(yb, it[rows]))
            df_.append(fpr(yb, v3[rows]) - fpr(yb, it[rows]))
        pa = roc_auc_score(y, v3) - roc_auc_score(y, it)
        pf = fpr(y, v3) - fpr(y, it)
        ci = lambda d: np.percentile(d, [2.5, 97.5]).tolist()
        out[ds] = dict(auc_delta_pts=[100 * pa] + [100 * x for x in ci(da)],
                       fpr05_delta_pts=[100 * pf] + [100 * x for x in ci(df_)],
                       auc_ci_excludes_0=bool(np.all(np.array(ci(da)) > 0) or np.all(np.array(ci(da)) < 0)),
                       fpr_ci_excludes_0=bool(np.all(np.array(ci(df_)) > 0) or np.all(np.array(ci(df_)) < 0)),
                       v3_fpr05=fpr(y, v3), interim_fpr05=fpr(y, it))
    f = lambda d: out[d]
    out["gate"] = dict(i_ffpp=bool(f("FFPP")["auc_delta_pts"][0] >= -0.5),
                       ii_better=[d for d in ("CDF", "DFDCP", "DFF_test") if f(d)["auc_delta_pts"][0] > 0],
                       iii_fpr=bool(all(f(d)["fpr05_delta_pts"][0] <= 1 for d in TEST)))
    out["gate"]["passed"] = bool(out["gate"]["i_ffpp"] and len(out["gate"]["ii_better"]) >= 2 and out["gate"]["iii_fpr"])
    name = "R6_gate_ci_posthoc_v3tf.json" if a.v3_tf else "R6_gate_ci.json"
    (a.dir / name).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

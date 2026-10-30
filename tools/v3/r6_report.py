"""R6 (full): metrics, 95% video-bootstrap CIs (1,000), DEV-chosen thresholds, GATE R6, (b) vs (b-old).

Inputs in --dir: frames_full.csv (r6_frames.py), results_{V3,bold,a}.csv (r6_batched.py), results_V3_gray.csv
(r6_llava.py), optionally author_out_<dataset>.json (eval.infer.runner run directly). Missing systems are skipped (reported as absent).
Thresholds: for each system, t_dev = argmax(TPR - FPR) on DEV (FFPP val + DFF val, never test), then applied to test.
GATE R6 (ROOT_FIX_PLAN): V3 vs interim — FFPP frame AUC not lower by > 0.5 pt, higher on >= 2 of CDF/DFDCP/DFF_test,
FPR(real) @0.5 not higher by > 1 pt on any test set.

    python tools/v3/r6_report.py --dir DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/eval"))
import benchmark as B  # noqa: E402
import stats as S  # noqa: E402

SYSTEMS = ["a", "bold", "b", "interim", "V3", "V3_gray"]
NAMES = {"a": "(a) X2DFD gốc", "bold": "(b-old) live trước audit", "b": "(b) parity v2", "interim": "(interim) Router v3+D1",
         "V3": "(V3) lineage V3", "V3_gray": "(V3, ảnh xám)"}
TEST = ["FFPP", "CDF", "DFDCP", "DFF_test"]


def load(d):
    df = pd.read_csv(d / "frames_full.csv")
    df["a_fake"] = np.nan
    for s in ("V3", "bold", "V3_gray", "a"):
        f = d / f"results_{s}.csv"
        if f.exists():
            r = pd.read_csv(f).drop_duplicates("image_path").set_index("image_path")[f"{s}_fake"]
            df[f"{s}_fake"] = df.image_path.map(r)
    for f in d.glob("author_out_*.json"):  # eval.infer.runner outputs, if run directly
        rows = [dict(path=p) for p in df.image_path]
        B.read_author(f, rows)
        got = pd.Series([r["a_fake"] for r in rows], index=df.index).replace("", np.nan).astype(float)
        df["a_fake"] = df["a_fake"].fillna(got)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    df = load(a.dir)
    systems = [s for s in SYSTEMS if f"{s}_fake" in df and df[f"{s}_fake"].notna().any()]
    dev, test = df[df.split == "dev"], df[df.split == "test"]
    rep = {"systems": systems, "coverage": {s: {ds: float(g[f"{s}_fake"].notna().mean())
                                                for ds, g in df.groupby("dataset")} for s in systems}}
    thr = {}
    for s in systems:
        g = dev[dev[f"{s}_fake"].notna()]
        if g.label.nunique() == 2 and len(g) == len(dev):
            t = S.youden_threshold(g.label, g[f"{s}_fake"])
            thr[s] = dict(t_dev=t, dev_at_05=S.rates(g.label, g[f"{s}_fake"], 0.5),
                          dev_at_t=S.rates(g.label, g[f"{s}_fake"], t))
    rep["thresholds"] = thr

    per = {}
    for ds in TEST:
        g = test[test.dataset == ds]
        cols = [f"{s}_fake" for s in systems if g[f"{s}_fake"].notna().all()]
        ci, diffs = S.bootstrap_auc(g, cols, n=a.n_boot)
        per[ds] = {"frames": int(len(g)), "videos": int(g.video.nunique()), "auc": ci, "diff": diffs, "rates": {}}
        for c in cols:
            s = c[:-5]
            per[ds]["rates"][s] = {"@0.5": S.rates(g.label, g[c], 0.5)}
            if s in thr:
                per[ds]["rates"][s]["@t_dev"] = S.rates(g.label, g[c], thr[s]["t_dev"])
            per[ds]["rates"][s]["ties"] = S.ties(g[c])
    rep["test"] = per

    if all(f"{s}_fake" in df for s in ("V3", "interim")):
        f = lambda ds, s: per[ds]["auc"].get(f"{s}_fake", {}).get("frame", [np.nan])[0]
        ffpp = (f("FFPP", "V3") - f("FFPP", "interim")) * 100
        better = [ds for ds in ("CDF", "DFDCP", "DFF_test") if f(ds, "V3") > f(ds, "interim")]
        fpr = {ds: (per[ds]["rates"]["V3"]["@0.5"]["fpr_real"] - per[ds]["rates"]["interim"]["@0.5"]["fpr_real"]) * 100
               for ds in TEST if "V3" in per[ds]["rates"]}
        rep["gate_R6"] = dict(ffpp_delta_pts=ffpp, better_on=better, fpr_delta_pts=fpr,
                              passed=bool(ffpp >= -0.5 and len(better) >= 2 and all(v <= 1 for v in fpr.values())))
    (a.dir / "R6_report.json").write_text(json.dumps(rep, indent=1, default=float))

    lines = ["| Hệ | Tập | frame AUC [95% CI] | video AUC [95% CI] | acc/FPR/recall @0.5 | @t_dev |", "|---|---|---|---|---|---|"]
    for ds in TEST:
        for s in systems:
            c = f"{s}_fake"
            if c not in per[ds]["auc"]:
                continue
            fa, va, r = per[ds]["auc"][c]["frame"], per[ds]["auc"][c]["video"], per[ds]["rates"][s]
            r5 = r["@0.5"]
            rt = r.get("@t_dev")
            lines.append(f"| {NAMES[s]} | {ds} | {fa[0]:.3f} [{fa[1]:.3f}; {fa[2]:.3f}] | {va[0]:.3f} [{va[1]:.3f}; {va[2]:.3f}] | "
                         f"{r5['acc']:.3f}/{r5['fpr_real']:.3f}/{r5['recall_fake']:.3f} | "
                         + (f"t={rt['threshold']:.3f}: {rt['acc']:.3f}/{rt['fpr_real']:.3f}/{rt['recall_fake']:.3f}" if rt else "—") + " |")
    (a.dir / "R6_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: rep[k] for k in ("thresholds", "gate_R6") if k in rep}, indent=1, default=float))


if __name__ == "__main__":
    main()

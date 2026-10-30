"""TRAIN_DIFFUSION_PLAN D2.4 / GATE C: Router4 v2 vs v3, end-to-end (Router top-1 -> calibrated score ->
LoRA FINAL_CLEAN via the exhaustive train-parity score table).

  v2 = router_training_v2/run_effb0_3domain/best.pt + router_teacher_v2 calibrators   (live after the audit)
  v3 = ROUTER4_DIFF_RETRAIN_20261030 best.pt          + calibrators_v2_diffrefit         (D1 + D2)
Sets
  INDOMAIN_TEST  router_teacher_v2/test_teacher_v2.csv, --per-domain rows per dataset (label-balanced, seed 42),
                 stored raw expert scores (the ones the teacher/calibrators were built from)
  DFF_VAL/TEST   DeepFakeFace identity-disjoint splits, raw scores from tools/diffusion/score_experts.py
  CDF / DFDCP    frames of the PHASE 4 benchmark (frames.csv); missing raw scores computed with the live wrappers
"""
import argparse
import asyncio
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from web.model_worker.expert_client import ExpertClient  # noqa: E402
from web.model_worker.router_client import EXPERTS, RouterClient  # noqa: E402

SYS = {
    "v2": ("/home/aiotlab/hoangan/outputs/router_training_v2/run_effb0_3domain/best.pt",
           "/home/aiotlab/hoangan/outputs/router_teacher_v2/calibrators_train_only.joblib"),
    "v3": ("/home/aiotlab/hoangan/outputs/ROUTER4_DIFF_RETRAIN_20261030/02_router/run_effb0_4domain/best.pt",
           "/home/aiotlab/hoangan/outputs/ROUTER4_DIFF_RETRAIN_20261030/01_teacher/calibrators_v2_diffrefit.joblib"),
}
DFF = Path("/home/aiotlab/hoangan/datasets/DeepFakeFace_DFF")


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def load_sets(per_domain, bench):
    t = pd.read_csv("/home/aiotlab/hoangan/outputs/router_teacher_v2/test_teacher_v2.csv")
    parts = []
    for ds, g in t.groupby("dataset"):
        g = g.groupby("label", group_keys=False).apply(lambda x: x.sample(min(per_domain // 2, len(x)),
                                                                           random_state=42))
        parts.append(g.assign(set=ds + "_test", video=g.image_path.map(lambda p: str(Path(p).parent))))
    for split in ("val", "test"):
        f = DFF / f"DFF_{split}_scores.csv"
        d = pd.read_csv(f if f.exists() else DFF / f"DFF_{split}.csv")
        parts.append(d.assign(set=f"DFF_{split}", video=d.image_path))
    b = pd.read_csv(bench)
    b = b[b.dataset.isin(["CDF", "DFDCP"])].rename(columns={"path": "image_path"})
    b["set"] = b.dataset
    b["score_blending"] = b["blending_raw"]
    for e in EXPERTS:
        if e != "blending":
            b[f"score_{e}"] = np.where(b.b_expert == e, b.b_raw, np.nan)
    parts.append(b)
    keep = ["set", "image_path", "video", "label", "generator"] + [f"score_{e}" for e in EXPERTS]
    df = pd.concat([p.reindex(columns=keep) for p in parts], ignore_index=True)
    df["label"] = df["label"].astype(int)
    return df


async def fill_raw(df):
    ec = ExpertClient()
    for e in EXPERTS:
        miss = df[f"score_{e}"].isna()
        if miss.any():
            paths = [Path(p) for p in df.loc[miss, "image_path"]]
            res = await ec.compute_expert_scores_batch(e, paths)
            df.loc[miss, f"score_{e}"] = [res[str(p.resolve())][0] for p in paths]


def route(df, ckpt):
    rc = RouterClient(ckpt)
    sel, conf = [], []
    for p in df.image_path:
        e, _, c, _ = rc.route_image(Path(p))
        sel.append(e)
        conf.append(c)
    return sel, conf


def evaluate(df, name, table):
    out = {}
    for s, g in df.groupby("set"):
        y = g.label.to_numpy()
        p = g[f"{name}_fake"].to_numpy()
        pred = (p >= 0.5).astype(int)
        r = dict(n=len(g), auc=auc(y, p), acc=float((pred == y).mean()),
                 fpr_real=float(pred[y == 0].mean()), recall_fake=float(pred[y == 1].mean()),
                 routes={e: float((g[f"{name}_expert"] == e).mean()) for e in EXPERTS})
        if g.video.nunique() < len(g):
            v = g.groupby("video").agg(y=("label", "first"), p=(f"{name}_fake", "mean"))
            r["video_auc"] = auc(v.y, v.p)
        fk = g[g.label == 1]
        r["diffusion_route_on_fake"] = float((fk[f"{name}_expert"] == "diffusion").mean()) if len(fk) else None
        out[s] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--per-domain", type=int, default=4000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "gate_c_rows.csv"
    t = pd.read_csv(a.table)
    table = {(al, f"{s:.3f}"): p for al, s, p in zip(t.alias, t.score_3dp, t.fake_prob)}
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        df = load_sets(a.per_domain, a.bench)
        asyncio.run(fill_raw(df))
        for name, (ckpt, _) in SYS.items():
            df[f"{name}_expert"], df[f"{name}_conf"] = route(df, ckpt)
        df.to_csv(cache, index=False)
    rep = {}
    for name, (_, calp) in SYS.items():
        b = joblib.load(calp)["calibrators"]
        raw = df.apply(lambda r: r[f"score_{r[f'{name}_expert']}"], axis=1).to_numpy(float)
        cal = np.empty(len(df))
        for e in EXPERTS:
            m = (df[f"{name}_expert"] == e).to_numpy()
            if m.any():
                cal[m] = b[e].predict(raw[m])
        df[f"{name}_cal"] = cal
        df[f"{name}_fake"] = [table[(e, f"{c:.3f}")] for e, c in zip(df[f"{name}_expert"], cal)]
        rep[name] = evaluate(df, name, table)
    df.to_csv(cache, index=False)
    (a.out / "gate_c.json").write_text(json.dumps(rep, indent=1))
    rows = []
    for s in sorted(rep["v2"]):
        r2, r3 = rep["v2"][s], rep["v3"][s]
        rows.append([s, r2["n"], f"{r2['auc']:.4f}", f"{r3['auc']:.4f}", f"{(r3['auc'] - r2['auc']) * 100:+.2f}",
                     f"{r2.get('video_auc', float('nan')):.4f}", f"{r3.get('video_auc', float('nan')):.4f}",
                     f"{r2['acc']:.3f}", f"{r3['acc']:.3f}", f"{r2['fpr_real']:.3f}", f"{r3['fpr_real']:.3f}",
                     f"{r2['recall_fake']:.3f}", f"{r3['recall_fake']:.3f}",
                     f"{r2['diffusion_route_on_fake'] or 0:.3f}", f"{r3['diffusion_route_on_fake'] or 0:.3f}"])
    hdr = ["set", "n", "auc_v2", "auc_v3", "d_auc_pts", "vauc_v2", "vauc_v3", "acc_v2", "acc_v3", "fpr_v2",
           "fpr_v3", "rec_v2", "rec_v3", "diffroute_fake_v2", "diffroute_fake_v3"]
    with open(a.out / "gate_c_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(map(str, r)))


if __name__ == "__main__":
    main()

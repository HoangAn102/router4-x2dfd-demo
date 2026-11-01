"""E2 statistics = the old report's protocol (ROUTER4_CURRENT_FROZEN_EXTERNAL_20261010/finalize.py, sha 0752f21a…).

weighted Mann-Whitney AUC; per (dataset, level) a fresh np.random.default_rng(20261001); 2,000 replicates of
multiplicity = bincount(rng.integers(0, nv, nv)) over the sorted video keys; frame rows weighted by their video's
multiplicity, video rows by the multiplicity; single-class replicates dropped; percentile 2.5 / 97.5.

    python tools/v3/e2_stats.py --selftest   # must reproduce the old Table 4 from the old predictions
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED, REPS = 20261001, 2000
RUNS = Path("/home/aiotlab/router4_x2dfd_FULL_SOURCE_20260924_054600/replication_author_audit/runs")
OLD = RUNS / "ROUTER4_CURRENT_FROZEN_EXTERNAL_20261010"


def weighted_auc(y, s, w):
    order = np.argsort(s, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(s[order])) + 1]
    p = np.add.reduceat(w[order] * y[order], starts)
    n = np.add.reduceat(w[order] * (1 - y[order]), starts)
    if p.sum() == 0 or n.sum() == 0:
        return float("nan")
    return float(np.dot(p, np.cumsum(n) - .5 * n) / (p.sum() * n.sum()))


def prepare(g):
    """g: frames of one dataset in manifest order with columns video, label. Returns group index, labels."""
    groups = sorted(set(g.video))
    index = {v: i for i, v in enumerate(groups)}
    gid = np.array([index[v] for v in g.video])
    y = g.label.to_numpy(int)
    yv = np.zeros(len(groups), int)
    yv[gid] = y
    assert np.array_equal(y, yv[gid]), "label not constant within a video"
    return gid, y, yv, len(groups)


def compare(g, base, other, reps=REPS, seed=SEED):
    """Paired (other - base) on frame and video level. Returns list of dicts (finalize.py fields)."""
    gid, y, yv, nv = prepare(g)
    counts = np.bincount(gid)
    out = []
    for level in ("frame", "video"):
        if level == "frame":
            labels, ss = y, [g[base].to_numpy(float), g[other].to_numpy(float)]
        else:
            labels = yv
            ss = [np.bincount(gid, weights=g[c].to_numpy(float)) / counts for c in (base, other)]
        point = [weighted_auc(labels, s, np.ones(len(labels))) for s in ss]
        rng, deltas = np.random.default_rng(seed), []
        for _ in range(reps):
            mult = np.bincount(rng.integers(0, nv, nv), minlength=nv)
            w = mult[gid] if level == "frame" else mult
            a, b = [weighted_auc(labels, s, w) for s in ss]
            if math.isfinite(a) and math.isfinite(b):
                deltas.append(b - a)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out.append(dict(level=level, n=len(labels), videos=nv, base_auc=point[0], other_auc=point[1],
                        delta_pp=100 * (point[1] - point[0]), ci_low_pp=100 * float(lo), ci_high_pp=100 * float(hi),
                        ci_excludes_0=bool(lo > 0 or hi < 0), bootstrap_valid=len(deltas)))
    return out


def rates(y, s, t):
    y, p = np.asarray(y, int), (np.asarray(s, float) >= t).astype(int)
    return dict(threshold=float(t), fpr_real=float(p[y == 0].mean()), recall_fake=float(p[y == 1].mean()),
                acc=float((p == y).mean()))


def selftest():
    m = pd.read_csv(OLD / "manifest.csv").rename(columns={"official_video_key": "video"})
    m["video"] = m.video.astype(str)
    for name in ("original", "router4"):
        p = pd.read_csv(OLD / f"outputs/{name}_predictions.csv").set_index("image_path").fake_score
        m[name] = m.image_path.map(p)
    ref = pd.read_csv(OLD / "outputs/comparison.csv")
    worst = 0.0
    for ds in ("Celeb-DF-v2", "DFDC", "DFDCP"):
        for r in compare(m[m.dataset == ds], "original", "router4"):
            x = ref[(ref.dataset == ds) & (ref.level == r["level"])].iloc[0]
            d = max(abs(r["base_auc"] - x.original_auc), abs(r["other_auc"] - x.router4_auc),
                    abs(r["ci_low_pp"] - x.ci_low_pp), abs(r["ci_high_pp"] - x.ci_high_pp))
            worst = max(worst, d)
            print(ds, r["level"], f"{r['base_auc']:.6f} {r['other_auc']:.6f} {r['delta_pp']:+.3f} "
                  f"[{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}]  max|diff vs old| {d:.2e}")
    print("SELFTEST", "PASS" if worst <= 1e-3 else "FAIL", worst)
    return worst <= 1e-3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(0 if selftest() else 1)

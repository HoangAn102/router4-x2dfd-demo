"""Shared statistics for the final report: video-level bootstrap CIs, paired AUC differences, ties, thresholds."""
import numpy as np
from sklearn.metrics import roc_auc_score


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def video_table(df, col):
    v = df.groupby("video").agg(y=("label", "first"), p=(col, "mean"))
    return v.y.to_numpy(int), v.p.to_numpy(float)


def bootstrap_auc(df, cols, n=1000, seed=0):
    """Resample VIDEOS with replacement (n times). Returns {col: {frame_auc, video_auc: (point, lo, hi)}} and,
    for every pair of cols, the paired difference CI on the same resamples."""
    rng = np.random.default_rng(seed)
    vids = df.video.unique()
    idx = {v: np.flatnonzero(df.video.to_numpy() == v) for v in vids}
    y_all = df.label.to_numpy(int)
    S = {c: df[c].to_numpy(float) for c in cols}
    vy = df.groupby("video").label.first()
    VS = {c: df.groupby("video")[c].mean() for c in cols}
    point = {c: dict(frame=_auc(y_all, S[c]), video=_auc(vy.to_numpy(), VS[c].loc[vy.index].to_numpy()))
             for c in cols}
    boots = {c: {"frame": [], "video": []} for c in cols}
    for _ in range(n):
        pick = rng.choice(vids, size=len(vids), replace=True)
        rows = np.concatenate([idx[v] for v in pick])
        yb, vyb = y_all[rows], vy.loc[pick].to_numpy()
        for c in cols:
            boots[c]["frame"].append(_auc(yb, S[c][rows]))
            boots[c]["video"].append(_auc(vyb, VS[c].loc[pick].to_numpy()))
    out = {}
    for c in cols:
        out[c] = {k: (point[c][k], *np.nanpercentile(boots[c][k], [2.5, 97.5]).tolist()) for k in ("frame", "video")}
    diffs = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            for k in ("frame", "video"):
                d = np.array(boots[a][k]) - np.array(boots[b][k])
                diffs[f"{a}-{b}|{k}"] = (point[a][k] - point[b][k], *np.nanpercentile(d, [2.5, 97.5]).tolist(),
                                         float(np.nanmean(d <= 0)))
    return out, diffs


def ties(s, decimals=8):
    """Distinct score values and share of frames whose score is shared with at least one other frame."""
    r = np.round(np.asarray(s, float), decimals)
    vals, counts = np.unique(r, return_counts=True)
    cnt = dict(zip(vals, counts))
    return dict(n=len(r), distinct=int(len(vals)), tied_share=float(np.mean([cnt[x] > 1 for x in r])),
                largest_tie_share=float(counts.max() / len(r)))


def rates(y, s, t):
    y, p = np.asarray(y, int), (np.asarray(s, float) >= t).astype(int)
    return dict(threshold=float(t), acc=float((p == y).mean()), fpr_real=float(p[y == 0].mean()),
                recall_fake=float(p[y == 1].mean()), bal_acc=float((1 - p[y == 0].mean() + p[y == 1].mean()) / 2))


def youden_threshold(y, s):
    """Threshold maximizing TPR - FPR (balanced accuracy) on a DEV set."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(np.asarray(y, int), np.asarray(s, float))
    i = int(np.argmax(tpr - fpr))
    return float(min(max(thr[i], 0.0), 1.0))

"""E2 report (PREREG_E2): reproducibility of (a), old-report Table-4 tables for V3-P0 / V3-P1, before/after table,
DEV thresholds, P1 deploy rule, expert distribution, stand-alone expert AUCs, ORACLE-blending (diagnostic).

    python tools/v3/e2_report.py --dir e2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_stats as S  # noqa: E402

R6 = Path("/home/aiotlab/hoangan/eval_runs/root_fix/r6_full")
OLD = S.OLD
OLD_FFPP = S.RUNS / "ROUTER4_IMPROVEMENT_RESEARCH_V1/internal/full/router4_cal_internal_predictions.csv"
OUT_SETS = ["CDF", "DFDC", "DFDCP_val"]
TABLE_SETS = ["CDF", "DFDC", "DFDCP_val", "FFPP_dev", "DFF_test"]
LABEL = {"CDF": "Celeb-DF-v2", "DFDC": "DFDC", "DFDCP_val": "DFDCP val", "FFPP_dev": "FFPP development",
         "DFF_test": "DFF test"}
OLD_ORIG = {"CDF": (0.869587, 0.916755), "DFDC": (0.817831, 0.834027), "DFDCP_val": (0.876608, 0.895869)}
EXPERTS = ["blending", "diffusion", "frequency", "texture"]


def scores(path, col):
    return pd.read_csv(path).drop_duplicates("image_path").set_index("image_path")[col]


def load(d):
    df = pd.read_csv(d / "frames_e2.csv")
    a = pd.concat([scores(R6 / "results_a.csv", "a_fake"), scores(d / "results_a.csv", "a_fake")])
    p0 = pd.concat([scores(R6 / "results_V3.csv", "V3_fake"), scores(d / "results_P0.csv", "V3_fake")])
    df["a"] = df.image_path.map(a[~a.index.duplicated()])
    df["P0"] = df.image_path.map(p0[~p0.index.duplicated()])
    for s in ("P1", "OR"):
        ch = scores(d / f"results_{s}.csv", "V3_fake")
        df[s] = df.image_path.map(ch).where(df[f"{s}_expert"] != df.P0_expert, df.P0)
    old = pd.concat([scores(OLD / "outputs/router4_predictions.csv", "fake_score"), scores(OLD_FFPP, "fake_score")])
    df["R4old"] = df.image_path.map(old)
    df["orig_old"] = df.image_path.map(scores(OLD / "outputs/original_predictions.csv", "fake_score"))
    return df


def t_dev(dev, col):
    fpr, tpr, thr = roc_curve(dev.label.to_numpy(int), dev[col].to_numpy(float))
    return float(min(max(thr[int(np.argmax(tpr - fpr))], 0.0), 1.0))


def table4(df, other, base="a", name=None):
    rows = []
    for ds in TABLE_SETS:
        g = df[df.dataset == ds]
        if g[[base, other]].isna().any().any():
            continue
        for r in S.compare(g, base, other):
            rows.append(dict(dataset=ds, **r))
    t = pd.DataFrame(rows)
    lines = [f"| Dataset | Level | Original | {name or other} | Delta pp | Paired95%CI pp |", "|---|---|---:|---:|---:|---:|"]
    for r in t.itertuples():
        lines.append(f"|{LABEL[r.dataset]}|{r.level}|{r.base_auc:.6f}|{r.other_auc:.6f}|{r.delta_pp:+.3f}|"
                     f"[{r.ci_low_pp:+.3f}, {r.ci_high_pp:+.3f}]|")
    return t, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    d = a.dir
    df = load(d)
    systems = ["a", "P0", "P1", "OR"]
    rep = {"missing": {s: {ds: int(g[s].isna().sum()) for ds, g in df.groupby("dataset")} for s in systems}}
    assert all(v == 0 for m in rep["missing"].values() for v in m.values()), rep["missing"]

    # 1. reproducibility of (a) against the old report's Original (PREREG_E2 §4)
    repro = {}
    for ds, (f_old, v_old) in OLD_ORIG.items():
        g = df[df.dataset == ds]
        gid, y, yv, nv = S.prepare(g)
        f_new = S.weighted_auc(y, g.a.to_numpy(float), np.ones(len(y)))
        v_new = S.weighted_auc(yv, np.bincount(gid, weights=g.a.to_numpy(float)) / np.bincount(gid), np.ones(nv))
        repro[ds] = dict(frame_old=f_old, frame_new=f_new, video_old=v_old, video_new=v_new,
                         max_abs_diff=max(abs(f_new - f_old), abs(v_new - v_old)),
                         ok=bool(max(abs(f_new - f_old), abs(v_new - v_old)) <= 0.005),
                         frame_score_mean_abs_diff=float((g.a - g.orig_old).abs().mean()),
                         verdict_agree_05=float(((g.a >= .5) == (g.orig_old >= .5)).mean()))
    rep["reproducibility_a"] = repro

    # 2. thresholds on DEV (FFPP val = FFPP development + DFF val)
    dev = df[df.dev]
    thr = {"a": 0.90598983, "P0": 0.4610168}  # R6 t_dev (PREREG_E2 §3: 0.906 and 0.4610168)
    thr_check = {s: t_dev(dev, s) for s in systems}
    thr.update({s: thr_check[s] for s in ("P1", "OR")})
    rep["t_dev"], rep["t_dev_recomputed_check"] = thr, thr_check
    rates = {}
    for ds in TABLE_SETS:
        g = df[df.dataset == ds]
        rates[ds] = {s: {"@0.5": S.rates(g.label, g[s], 0.5),
                         **({"@t_dev": S.rates(g.label, g[s], thr[s])} if s in thr else {})}  # Router4 cũ: no DEV
                     for s in systems + (["R4old"] if g.R4old.notna().all() else [])}
    rep["rates"] = rates

    # 3. Table-4 style tables (Original = (a) re-run) + paired secondary comparisons
    md = []
    for s, name in (("P0", "V3-P0"), ("P1", "V3-P1"), ("R4old", "Router4 CŨ")):
        t, txt = table4(df, s, name=name)
        rep[f"table4_{s}"] = t.to_dict("records")
        md += [f"### Original vs {name}", "", txt, ""]
    for base, other in (("P0", "P1"), ("P0", "OR"), ("R4old", "P0"), ("orig_old", "a")):
        t, _ = table4(df, other, base=base)
        rep[f"paired_{other}_minus_{base}"] = t.to_dict("records")

    # 4. P1 deploy rule (PREREG_E2 §5), point estimates, frame AUC
    f = {(r["dataset"], r["level"]): r for r in rep["paired_P1_minus_P0"]}
    better = [ds for ds in OUT_SETS if f[(ds, "frame")]["delta_pp"] >= 0]
    keep = {ds: f[(ds, "frame")]["delta_pp"] for ds in ("FFPP_dev", "DFF_test")}
    rep["P1_rule"] = dict(out_of_domain_not_worse=better, in_domain_delta_pp=keep,
                          passed=bool(len(better) >= 2 and all(v >= -0.5 for v in keep.values())))

    # 5. before/after table
    ba = ["| Tập | Mức | Original | Router4 CŨ | V3-P0 | V3-P1 |", "|---|---|---:|---:|---:|---:|"]
    get = lambda s, ds, lv: {(r["dataset"], r["level"]): r for r in rep[f"table4_{s}"]}.get((ds, lv))
    for ds in TABLE_SETS:
        for lv in ("frame", "video"):
            cells = []
            for s in ("R4old", "P0", "P1"):
                r = get(s, ds, lv)
                cells.append("—" if r is None else f"{r['other_auc']:.6f} ({r['delta_pp']:+.2f} [{r['ci_low_pp']:+.2f}; "
                             f"{r['ci_high_pp']:+.2f}])")
            base = get("P0", ds, lv)
            ba.append(f"| {LABEL[ds]} | {lv} | {base['base_auc']:.6f} | " + " | ".join(cells) + " |")
    md += ["### Trước/sau sửa (Δ pp so Original chạy lại [CI ghép cặp 95%, giao thức cũ])", ""] + ba + [""]

    # 6. expert distribution, stand-alone expert AUC, low-confidence share
    dist = {}
    for ds, g in df.groupby("dataset"):
        dist[ds] = {"P0": g.P0_expert.value_counts(normalize=True).round(4).to_dict(),
                    "P1": g.P1_expert.value_counts(normalize=True).round(4).to_dict(),
                    "low_conf_share": float((g.router_conf < 0.30).mean())}
        gid, y, yv, nv = S.prepare(g)
        for e in EXPERTS:
            sc = g[f"score_{e}"].to_numpy(float)
            dist[ds][f"auc_{e}"] = dict(frame=S.weighted_auc(y, sc, np.ones(len(y))),
                                        video=S.weighted_auc(yv, np.bincount(gid, weights=sc) / np.bincount(gid),
                                                             np.ones(nv)))
    rep["experts"] = dist
    (d / "E2_report.json").write_text(json.dumps(rep, indent=1, default=float))
    (d / "E2_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps({k: rep[k] for k in ("reproducibility_a", "t_dev", "P1_rule")}, indent=1, default=float))


if __name__ == "__main__":
    main()

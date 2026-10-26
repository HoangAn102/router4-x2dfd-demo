"""Report for tools/eval/pipeline_eval.py (run with --x2dfd): per-set metrics for Haar vs Haar+YuNet.

Per set and mode: images, analysed (face found), accuracy on analysed, FPR on real, recall on fake,
AUC (X2DFD fake probability), confusion matrix. Also a per-image before/after table for the reference
images (test_imgs, must-pass) and two figures.

Usage: python tools/eval/final_report.py DIR   -> DIR/ablation.csv, report_tables.md, fig_*.png
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MODES = {"haar": "Haar only (before)", "yunet": "Haar + YuNet (now)"}
SETS = {  # report set -> real groups, fake groups
    "CelebDF (external)": (["celebdf_real"], ["celebdf_fake"]),
    "FFPP test (in-domain)": (["ffpp_real"], ["ffpp_fake"]),
    "FFHQ clean (real only)": (["ffhq_real"], []),
    "AI / diffusion": (["diffusion_pilot_real"], ["ai_set"]),
}


def metrics(d):
    """d holds the selected rows of one set (any verdict)."""
    a = d[d.verdict.isin(["REAL", "FAKE"])]
    real, fake = a[a.label == 0], a[a.label == 1]
    out = {"n": len(d), "n_real": int((d.label == 0).sum()), "n_fake": int((d.label == 1).sum()),
           "analysed": len(a), "no_face": int((d.verdict == "NO_FACE").sum()),
           "acc": float((a.verdict == np.where(a.label == 1, "FAKE", "REAL")).mean()) if len(a) else math.nan,
           "fpr": float((real.verdict == "FAKE").mean()) if len(real) else math.nan,
           "recall": float((fake.verdict == "FAKE").mean()) if len(fake) else math.nan,
           "tn": int((real.verdict == "REAL").sum()), "fp": int((real.verdict == "FAKE").sum()),
           "fn": int((fake.verdict == "REAL").sum()), "tp": int((fake.verdict == "FAKE").sum())}
    out["auc"] = (float(roc_auc_score(a.label, a.x2dfd_fake_prob.astype(float)))
                  if len(real) and len(fake) else math.nan)
    return out


def pct(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def num(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    res = {m: pd.read_csv(args.run_dir / f"results_{m}.csv") for m in MODES}

    rows = []
    for m, df in res.items():
        sel = df[df.role == "select"]
        for name, (rg, fg) in SETS.items():
            rows.append({"mode": m, "set": name, **metrics(sel[sel.group.isin(rg + fg)])})
        rows.append({"mode": m, "set": "ALL (selected)", **metrics(sel)})
    abl = pd.DataFrame(rows)
    abl.to_csv(args.run_dir / "ablation.csv", index=False)

    md = ["# Results (threshold 0.5)", "",
          "| mode | set | n (real/fake) | analysed | NO_FACE | accuracy | FPR real | recall fake | AUC | TN FP FN TP |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in abl.iterrows():
        md.append(f"| {MODES[r['mode']]} | {r['set']} | {r.n} ({r.n_real}/{r.n_fake}) | {r.analysed} | {r.no_face} | "
                  f"{pct(r.acc)} | {pct(r.fpr)} | {pct(r.recall)} | {num(r.auc)} | {r.tn} {r.fp} {r.fn} {r.tp} |")

    ref = {m: df[df.role == "reference"].set_index("path") for m, df in res.items()}
    md += ["", "# Reference images (test_imgs and must-pass; never used to choose parameters)", "",
           "| file | truth | Haar only | Haar + YuNet | detector | x2dfd (YuNet crop) | router expert (conf) |", "|---|---|---|---|---|---|---|"]
    wrong = {m: 0 for m in MODES}
    for path, r in ref["yunet"].iterrows():
        h = ref["haar"].loc[path]
        truth = "FAKE" if r.label == 1 else "REAL"
        cells = []
        for m, row in (("haar", h), ("yunet", r)):
            ok = row.verdict == truth
            wrong[m] += 0 if ok else 1
            cells.append(f"{row.verdict} ({'ok' if ok else 'SAI'})")
        md.append(f"| {Path(path).name[:36]} | {truth} | {cells[0]} | {cells[1]} | {r.detector if isinstance(r.detector, str) else '-'} | "
                  f"{num(r.x2dfd_fake_prob)} | {r.expert if isinstance(r.expert, str) else '-'} ({num(r.router_conf)}) |")
    md += ["", f"Wrong on reference images: Haar only {wrong['haar']}, Haar + YuNet {wrong['yunet']} (of {len(ref['yunet'])})."]
    (args.run_dir / "report_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = list(SETS)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    w = 0.38
    for i, (m, label) in enumerate(MODES.items()):
        vals = [abl[(abl["mode"] == m) & (abl.set == s)].acc.iloc[0] * 100 for s in sets]
        bars = ax.bar(np.arange(len(sets)) + (i - 0.5) * w, vals, w, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, "-" if math.isnan(v) else f"{v:.0f}", ha="center", fontsize=8)

    def tick(s):
        a = abl[(abl["mode"] == "haar") & (abl.set == s)].iloc[0]
        z = abl[(abl["mode"] == "yunet") & (abl.set == s)].iloc[0]
        return f"{s.replace(' (', chr(10) + '(')}\nn={a.n_real}+{a.n_fake}; analysed {a.analysed} -> {z.analysed}"

    ax.set_xticks(np.arange(len(sets)))
    ax.set_xticklabels([tick(s) for s in sets], fontsize=7)
    ax.set_ylabel("Accuracy (%) on analysed images")
    ax.set_ylim(0, 112)
    ax.set_title("Accuracy by set, original pipeline (threshold 0.5)\nGroups with very few analysed images are not comparable", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(args.run_dir / "fig_accuracy_by_group.png", dpi=150)

    d = res["yunet"]
    d = d[d.role == "select"]
    cm = np.zeros((2, 3), dtype=int)
    for t, v in zip(d.label, d.verdict):
        cm[int(t), {"REAL": 0, "FAKE": 1}.get(v, 2)] += 1
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(["REAL", "FAKE", "NO_FACE"])
    ax.set_yticks(range(2)); ax.set_yticklabels(["real (truth)", "fake (truth)"])
    for i in range(2):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_title("Haar + YuNet pipeline: confusion matrix\n(all selected images)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.run_dir / "fig_confusion_final.png", dpi=150)
    print("\nsaved ablation.csv, report_tables.md and 2 figures in", args.run_dir)


if __name__ == "__main__":
    main()

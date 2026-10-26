"""Ablation report from tools/eval/pipeline_eval.py results (run with --x2dfd).

Configurations are produced by calling the web's own decide() with gates switched off via settings:
  1 X2DFD only            2 + AI-image gate            3 + blending gate (faces only)
  4 final system          = 3 + AI gate on images where no face is found (what the endpoint returns)
Rows with no face are "not analysed" in 1-3 and use the AI gate in 4.

Usage: python tools/eval/final_report.py DIR   (DIR holds results.csv; writes ablation.csv, report_tables.md, *.png)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from web.backend.app.config import settings  # noqa: E402
from web.backend.app.schemas.image import FinalResult, Reliability  # noqa: E402
from web.backend.app.services.ai_gate import build_faceless_response  # noqa: E402
from web.backend.app.services.decision import decide  # noqa: E402

CONFIGS = ["1 X2DFD only", "2 + AI gate", "3 + blending gate", "4 final system"]
SETS = {  # report set -> (real groups, fake groups)
    "CelebDF (external)": (["celebdf_real"], ["celebdf_fake"]),
    "FFPP test (in-domain)": (["ffpp_real"], ["ffpp_fake"]),
    "FFHQ clean (real only)": (["ffhq_real"], []),
    "AI / diffusion": (["diffusion_pilot_real"], ["ai_set"]),
}


def verdicts(df, ai_on, blend_on, faceless_ai):
    bl_on = settings.BLENDING_GATE_ENABLED
    settings.BLENDING_GATE_ENABLED = blend_on
    out, score = [], []
    try:
        for _, r in df.iterrows():
            ai = float(r.ai_score) if (ai_on and not pd.isna(r.ai_score)) else None  # AI gate off == no score (fail open)
            if int(r.face) == 0:
                fake = build_faceless_response("eval", ai, {}, {}) if (faceless_ai and ai is not None) else None
                out.append("FAKE" if fake else "NO_FACE")
                score.append(ai if fake else np.nan)
                continue
            if pd.isna(r.x2dfd_fake_prob):
                out.append("UNSCORED"); score.append(np.nan); continue
            p = float(r.x2dfd_fake_prob)
            f = FinalResult(verdict="FAKE" if p >= 0.5 else "REAL", fake_probability=p, real_probability=1 - p, continuous_score=p)
            bl = float(r.blending_raw) if not pd.isna(r.blending_raw) else None
            final, _, ai_d, bl_d = decide(f, Reliability(), ai, bl)
            out.append(final.verdict)
            comps = [p]
            if ai_on and ai is not None:
                comps.append(ai)
            if blend_on and bl is not None:
                comps.append(bl)
            score.append(max(comps))
    finally:
        settings.BLENDING_GATE_ENABLED = bl_on
    return out, score


def metrics(df, v, s):
    ok = [x in ("FAKE", "REAL") for x in v]
    d = df[ok].copy()
    d["v"] = [x for x in v if x in ("FAKE", "REAL")]
    d["s"] = [x for x, o in zip(s, ok) if o]
    real, fake = d[d.label == 0], d[d.label == 1]
    res = {"n": len(df), "analysed": len(d),
           "acc": float((d.v == np.where(d.label == 1, "FAKE", "REAL")).mean()) if len(d) else math.nan,
           "fpr": float((real.v == "FAKE").mean()) if len(real) else math.nan,
           "recall": float((fake.v == "FAKE").mean()) if len(fake) else math.nan,
           "n_real": len(real), "n_fake": len(fake)}
    res["auc"] = float(roc_auc_score(d.label, d.s)) if len(real) and len(fake) and d.s.notna().all() else math.nan
    res["tn"], res["fp"] = int((real.v == "REAL").sum()), int((real.v == "FAKE").sum())
    res["fn"], res["tp"] = int((fake.v == "REAL").sum()), int((fake.v == "FAKE").sum())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    df = pd.read_csv(args.run_dir / "results.csv")
    ev = df[df.role == "select"].copy()
    ref = df[df.role == "reference"].copy()

    cfg_flags = [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]
    rows, per_cfg = [], {}
    for name, (ai_on, bl_on, fl) in zip(CONFIGS, cfg_flags):
        v, s = verdicts(ev, ai_on, bl_on, fl)
        ev[name] = v
        per_cfg[name] = (v, s)
        for set_name, (rg, fg) in SETS.items():
            m = ev.group.isin(rg + fg).values
            sub = ev[m]
            res = metrics(sub, [x for x, k in zip(v, m) if k], [x for x, k in zip(s, m) if k])
            rows.append({"config": name, "set": set_name, **res})
        res = metrics(ev, v, s)
        rows.append({"config": name, "set": "ALL (selected)", **res})
    abl = pd.DataFrame(rows)
    abl.to_csv(args.run_dir / "ablation.csv", index=False)

    md = ["# Ablation (config x set)", "", "| config | set | n (real/fake) | analysed | acc | FPR real | recall fake | AUC | TN FP FN TP |", "|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, pct=True: "-" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x * 100:.1f}%" if pct else f"{x:.3f}")
    for _, r in abl.iterrows():
        md.append(f"| {r.config} | {r.set} | {r.n} ({r.n_real}/{r.n_fake}) | {r.analysed} | {f(r.acc)} | {f(r.fpr)} | {f(r.recall)} | {f(r.auc, False)} | {r.tn} {r.fp} {r.fn} {r.tp} |")

    # reference images (test_imgs / must-pass): baseline = X2DFD only (config 1) -> final system (config 4)
    md += ["", "# Reference images (not used to choose thresholds)", "",
           "| file | group | truth | X2DFD only | final system | gate | ai_score | blending_raw | x2dfd |", "|---|---|---|---|---|---|---|---|---|"]
    v1, _ = verdicts(ref, False, False, False)
    v4, _ = verdicts(ref, True, True, True)
    for (_, r), a, b in zip(ref.iterrows(), v1, v4):
        truth = "FAKE" if r.label == 1 else "REAL"
        mark = "OK" if b == truth else "SAI"
        md.append(f"| {Path(r.path).name[:34]} | {r.group} | {truth} | {a} | {b} ({mark}) | {r.gate if isinstance(r.gate, str) else ''} | {f(r.ai_score, False)} | {f(r.blending_raw, False)} | {f(r.x2dfd_fake_prob, False)} |")
    (args.run_dir / "report_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: accuracy per set for the 4 configurations
    sets = [s for s in SETS if s != "FFHQ clean (real only)"] + ["FFHQ clean (real only)"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    w = 0.2
    for i, name in enumerate(CONFIGS):
        vals = [abl[(abl.config == name) & (abl.set == s)].acc.iloc[0] * 100 for s in sets]
        bars = ax.bar(np.arange(len(sets)) + (i - 1.5) * w, vals, w, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(sets)))
    def tick(s):
        a = abl[(abl.config == CONFIGS[0]) & (abl.set == s)].iloc[0]
        z = abl[(abl.config == CONFIGS[-1]) & (abl.set == s)].iloc[0]
        return f"{s.replace(' (', chr(10) + '(')}\nn={a.n_real}+{a.n_fake}; analysed {a.analysed} -> {z.analysed}"
    ax.set_xticklabels([tick(s) for s in sets], fontsize=7)
    ax.set_ylabel("Accuracy (%) on analysed images")
    ax.set_ylim(0, 112)
    ax.set_title("Accuracy on analysed images by set, four configurations (threshold 0.5)\nbars over very few analysed images (AI/diffusion) are not comparable", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(args.run_dir / "fig_accuracy_by_group.png", dpi=150)

    # Figure 2: confusion matrix of the final system (rows = truth, cols = verdict; NO_FACE reported separately)
    v, _ = per_cfg[CONFIGS[-1]]
    ev["final"] = v
    cm = np.zeros((2, 3), dtype=int)
    for t, x in zip(ev.label, ev.final):
        cm[int(t), {"REAL": 0, "FAKE": 1, "NO_FACE": 2}.get(x, 2)] += 1
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(["REAL", "FAKE", "NO_FACE"])
    ax.set_yticks(range(2)); ax.set_yticklabels(["real (truth)", "fake (truth)"])
    for i in range(2):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_title("Final system: confusion matrix\n(all selected images)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.run_dir / "fig_confusion_final.png", dpi=150)
    print("\nsaved ablation.csv, report_tables.md, 2 figures in", args.run_dir)


if __name__ == "__main__":
    main()

"""TRAIN_DIFFUSION_PLAN D0.3 / D2.1: raw scores of all 4 experts with the live wrappers (ExpertClient), resumable.

    python tools/diffusion/score_experts.py --manifest DFF_val.csv --out DFF_val_scores.csv [--chunk 2000]
Writes the manifest columns + score_<expert> (raw) for blending/diffusion/frequency/texture, and prints AUC per
expert (fake = positive) overall and per generator.
"""
import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from web.model_worker.expert_client import ExpertClient  # noqa: E402

EXPERTS = ["blending", "diffusion", "frequency", "texture"]


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


async def score(rows, chunk, save):
    ec = ExpertClient()
    for e in EXPERTS:
        todo = [r for r in rows if r.get(f"score_{e}") in (None, "")]
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            res = await ec.compute_expert_scores_batch(e, [Path(r["image_path"]) for r in part])
            for r in part:
                r[f"score_{e}"] = res[str(Path(r["image_path"]).resolve())][0]
            save()
            print(e, min(i + chunk, len(todo)), "/", len(todo), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.out if a.out.exists() else a.manifest)))
    fields = list(rows[0].keys()) + [f"score_{e}" for e in EXPERTS if f"score_{e}" not in rows[0]]

    def save():
        tmp = a.out.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(a.out)

    asyncio.run(score(rows, a.chunk, save))
    y = [int(r["label"]) for r in rows]
    rep = {"n": len(rows), "fake": sum(y), "real": len(y) - sum(y)}
    for e in EXPERTS:
        rep[f"auc_{e}"] = auc(y, [float(r[f"score_{e}"]) for r in rows])
        for g in sorted({r["generator"] for r in rows if r["label"] == "1"}):
            sub = [r for r in rows if r["label"] == "0" or r["generator"] == g]
            rep[f"auc_{e}_{g}"] = auc([int(r["label"]) for r in sub], [float(r[f"score_{e}"]) for r in sub])
    a.out.with_suffix(".auc.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

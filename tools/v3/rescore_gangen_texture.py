"""ROOT_FIX R1: re-score the Texture expert on every GANGen image of the v2 lineage with the live wrapper.

The stored GANGen texture scores (expert_profiling/full/GANGen/texture_scores.csv, 2026-09-04) came from a script
that no longer exists and do not match the live run_texture.py (AUDIT P2). Everything else passed parity.

    python tools/v3/rescore_gangen_texture.py --out OUT.csv [--chunk 4000]
"""
import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from web.model_worker.expert_client import ExpertClient  # noqa: E402

V2 = Path("/home/aiotlab/hoangan/outputs/router_teacher_v2")


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


async def run(paths, out, chunk):
    done = {r["image_path"] for r in csv.DictReader(open(out))} if out.exists() else set()
    todo = [p for p in paths if p not in done]
    ec = ExpertClient()
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["image_path", "texture_live"])
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            res = await ec.compute_expert_scores_batch("texture", [Path(p) for p in part])
            for p in part:
                w.writerow([p, res[str(Path(p).resolve())][0]])
            fh.flush()
            print(min(i + chunk, len(todo)), "/", len(todo), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=4000)
    a = ap.parse_args()
    frames = []
    for split in ("train", "val", "test"):
        d = pd.read_csv(V2 / f"{split}_teacher_v2.csv", usecols=["image_path", "label", "dataset", "score_texture"])
        frames.append(d[d.dataset == "GANGen"].assign(split=split))
    g = pd.concat(frames, ignore_index=True)
    asyncio.run(run(g.image_path.tolist(), a.out, a.chunk))
    live = pd.read_csv(a.out).drop_duplicates("image_path").set_index("image_path").texture_live
    g["texture_live"] = live.loc[g.image_path].to_numpy()
    rep = {s: dict(n=int(len(x)), auc_stored=auc(x.label, x.score_texture), auc_live=auc(x.label, x.texture_live),
                   median_abs_diff=float((x.texture_live - x.score_texture).abs().median()))
           for s, x in g.groupby("split")}
    a.out.with_suffix(".report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

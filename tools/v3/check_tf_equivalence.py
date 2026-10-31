"""Teacher-forced read-out (x2dfd_worker_cli.score_with_train_prompt, 31/10) vs the earlier read-outs.

  --part verify    R6 V3 verification frames: new single-image teacher forcing vs the generate-based scores
                   cached in results_V3.meta.json (old score_with_train_prompt)
  --part existing  rows already in results_V3.csv (or --src) (batched teacher forcing, greedy-prefix checked): new
                   single-image teacher forcing on a stratified random sample (--n, seed 0) or all (--n 0)
  --part table     (b) score table (FINAL_CLEAN, 4 x 1001 text-only prompts): new teacher forcing vs the
                   generate-based table phase2/score_table/llava_score_table.csv; writes the new table
Criterion (user, 31/10): verdict @0.5 identical on 100%, mean |delta| <= 1e-3.

    x2python tools/v3/check_tf_equivalence.py --part verify --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "web/model_worker"))
import x2dfd_worker_cli as W  # noqa: E402

BASE = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"
V3 = "/home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030/06_lora/router4_x2dfd_V3_full30k"
FC = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/05_lora/router4_x2dfd_FINAL_CLEAN"
R6 = Path("/home/aiotlab/hoangan/eval_runs/root_fix/r6_full")
OLD_TABLE = Path("/home/aiotlab/hoangan/eval_runs/final_audit/phase2/score_table/llava_score_table.csv")


def q(expert, cal):
    return f"<image>\nIs this image real or fake? And the {expert} score is {cal:.3f}."


def summary(old, new, on):
    d = (new - old).abs()
    return dict(n=int(len(d)), verdict_agree=int(((new >= 0.5) == (old >= 0.5)).sum()), mean_abs=float(d.mean()),
                p99_abs=float(d.quantile(0.99)), max_abs=float(d.max()), mean_signed=float((new - old).mean()),
                on_template=int(on.sum()),
                passed=bool(((new >= 0.5) == (old >= 0.5)).all() and d.mean() <= 1e-3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["verify", "existing", "table"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--src", type=Path, default=R6 / "results_V3.csv", help="existing: rows to re-check")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    frames = None if a.part == "table" else pd.read_csv(R6 / "frames_full.csv").drop_duplicates("image_path") \
        .set_index("image_path")
    if a.part == "verify":
        ref = json.loads((R6 / "results_V3.meta.json").read_text())["ref"]
        rows = pd.DataFrame(dict(image_path=list(ref), old=list(ref.values())))
    elif a.part == "existing":
        ex = pd.read_csv(a.src).rename(columns={"V3_fake": "old"})
        ex["dataset"] = ex.image_path.map(frames.dataset)
        rows = ex if a.n == 0 else ex.groupby("dataset", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(1, round(a.n * len(g) / len(ex)))), random_state=0))
    if a.part in ("verify", "existing"):
        res = []
        for i, p in enumerate(rows.image_path):
            res.append(W.score_with_train_prompt(q(frames.V3_expert[p], frames.V3_cal[p]), model_path=V3,
                                                 model_base=BASE, image_path=p))
            if i % 500 == 0:
                print(i, len(rows), flush=True)  # heartbeat for the R6 watchdog
    else:
        rows = pd.read_csv(OLD_TABLE, dtype={"score_3dp": str}).rename(columns={"fake_prob": "old"})
        res = [W.score_with_train_prompt(f"\nIs this image real or fake? And the {al} score is {s}.", model_path=FC,
                                         model_base=BASE) for al, s in zip(rows.alias, rows.score_3dp)]
        pd.DataFrame(dict(alias=rows.alias, score_3dp=rows.score_3dp, fake_prob=[r["fake_score"] for r in res],
                          real_prob=[r["real_score"] for r in res], label_step=[r["label_step"] for r in res],
                          answer=[r["answer"] for r in res], on_template=[r["on_template"] for r in res])
                     ).to_csv(a.out / "llava_score_table_tf.csv", index=False)
    rows = rows.assign(new=[r["fake_score"] for r in res], on_template=[r["on_template"] for r in res])
    rows.to_csv(a.out / f"tf_equivalence_{a.part}{a.tag}.csv", index=False)
    s = summary(rows.old, rows.new, rows.on_template)
    (a.out / f"tf_equivalence_{a.part}{a.tag}.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s))


if __name__ == "__main__":
    main()

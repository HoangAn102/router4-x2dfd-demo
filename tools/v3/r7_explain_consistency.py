"""R7 (2): how often does the explanation's first sentence state the same label as the score verdict?

Stratified sample of the R6 TEST frames (dataset x label, one frame per video, seed 0, --per frames per stratum).
Explanation = the web worker's explain_with_train_prompt (training prompt ids, greedy, --max-new-tokens, unmodified);
label = X2DFDClient.explanation_label (the web's flag); verdict = R6 score >= --threshold. Report only.

    x2python tools/v3/r7_explain_consistency.py --r6 r6_full --system V3 --lora LORA_V3 --out DIR/explain_V3.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "web/model_worker"))

BASE = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"


def sample(df, per, seed=0):
    one = df.sample(frac=1, random_state=seed).drop_duplicates("video")
    return pd.concat([g.head(per) for _, g in one.groupby(["dataset", "label"])]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r6", type=Path, required=True)
    ap.add_argument("--system", choices=["V3", "b"], required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per", type=int, default=38)
    ap.add_argument("--threshold", type=float, default=0.5)
    # greedy decoding: the first sentence (where the label is) does not depend on the generation length, so a short
    # budget gives the same label as the web's 512-token explanation; the stored text is then the opening only
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()

    import x2dfd_worker_cli as W
    from web.model_worker.x2dfd_client import X2DFDClient

    df = pd.read_csv(a.r6 / "frames_full.csv")
    df = df[df.split == "test"]
    if a.system == "V3":
        r = pd.read_csv(a.r6 / "results_V3.csv").drop_duplicates("image_path").set_index("image_path").V3_fake
        df = df.assign(fake=df.image_path.map(r), expert=df.V3_expert, cal=df.V3_cal, img="<image>\n")
    else:
        df = df.assign(fake=df.b_fake, expert=df.b_expert, cal=df.b_cal, img="\n")
    s = sample(df, a.per)
    done = set(pd.read_csv(a.out).image_path) if a.out.exists() else set()
    new = not a.out.exists()
    cols = ["dataset", "label", "image_path", "expert", "cal", "fake", "verdict", "exp_label", "consistent",
            "explanation"]
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        for i, r in enumerate(s[~s.image_path.isin(done)].itertuples()):
            q = f"{r.img}Is this image real or fake? And the {r.expert} score is {r.cal:.3f}."
            text = W.explain_with_train_prompt(q, model_path=a.lora, model_base=BASE, max_new_tokens=a.max_new_tokens,
                                               image_path=r.image_path)
            verdict = "FAKE" if r.fake >= a.threshold else "REAL"
            lab = X2DFDClient.explanation_label(text)
            w.writerow(dict(dataset=r.dataset, label=r.label, image_path=r.image_path, expert=r.expert, cal=r.cal,
                            fake=r.fake, verdict=verdict, exp_label=lab or "",
                            consistent="" if lab is None else lab == verdict, explanation=text))
            fh.flush()
            print(i, r.dataset, verdict, lab, flush=True)
    res = pd.read_csv(a.out)
    has = res[res.exp_label.notna()]
    summ = dict(n=len(res), with_label=len(has), consistent=int((has.exp_label == has.verdict).sum()),
                rate=float((has.exp_label == has.verdict).mean()) if len(has) else None,
                by_dataset={d: float((g.exp_label == g.verdict).mean()) for d, g in has.groupby("dataset")},
                explanation_matches_truth=float((has.exp_label == has.label.map({0: "REAL", 1: "FAKE"})).mean()),
                verdict_matches_truth=float((res.verdict == res.label.map({0: "REAL", 1: "FAKE"})).mean()))
    a.out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

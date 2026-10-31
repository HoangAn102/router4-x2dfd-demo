"""ROOT_FIX R5: shrink SFT V3 evenly (same fraction in every source x dataset x label stratum, seed 42) when the
measured LoRA throughput would exceed the 10-hour budget. Recipe/hyper-parameters are not changed.

    python tools/v3/subsample_sft.py --sft IN.json --out OUT.json --n 28000
"""
import argparse
import json
import random
from collections import Counter, defaultdict

import pandas as pd

TRAIN_SCORES = "/home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030/01_calibrators/scores_V3_train.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    items = json.load(open(a.sft))
    ds = pd.read_csv(TRAIN_SCORES, usecols=["image_path", "dataset"]).drop_duplicates("image_path")
    ds = ds.set_index("image_path").dataset
    strata = defaultdict(list)
    for it in items:
        label = "fake" if it["conversations"][1]["value"].startswith("This image is fake") else "real"
        strata[(ds[it["image"]], label)].append(it)
    frac = a.n / len(items)
    rng = random.Random(42)
    keep = []
    for k in sorted(strata):
        g = strata[k]
        keep += rng.sample(g, round(len(g) * frac))
    rng.shuffle(keep)
    for i, it in enumerate(keep):
        it["id"] = str(i)
    json.dump(keep, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(len(keep), {f"{k[0]}|{k[1]}": (len(v), round(len(v) * frac)) for k, v in sorted(strata.items())})
    print(Counter(it["conversations"][0]["value"].split(" the ")[1].split(" ")[0] for it in keep))


if __name__ == "__main__":
    main()

"""ROOT_FIX R4: SFT V3 in the X2DFD author's format (eval/tools/feature_annotation.py, utils/pipeline_utils.py).

  human: "<image>\\nIs this image real or fake? And the {alias} score is {score:.3f}."   (alias lowercase)
  gpt:   unchanged SFS rationale of the 24k SFT ("This image is real|fake. ..."), the old Router-WFS support clause
         removed and re-derived for the V3 alias/score with the same rule (03_router_wfs_merge.py: fake & s>=0.70 /
         real & s<=0.30); DeepFakeFace samples get the short answer "This image is real|fake." + that clause only
         (no invented visual reasoning).
  alias  = Router V3 top-1 (batched, same eval transform as RouterClient), score = OOF cal V3 of that expert.
Sources: the 24,000 images of router4_x2dfd_FINAL_CLEAN_SFT.json + --dff N DeepFakeFace-train images
(half real, a quarter SD1.5 text2img, a quarter SD inpainting; seed 42).

    python tools/v3/build_sft_v3.py --v3 /home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030 --router BEST.pt
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from web.model_worker.router_client import EXPERTS, RouterClient  # noqa: E402

OLD_SFT = Path("/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/04_sft/"
               "router4_x2dfd_FINAL_CLEAN_SFT.json")
LO, HI = 0.30, 0.70
OLD_CLAUSE = re.compile(r" Additionally, the selected \w+ expert provides (strong )?weak-feature evidence "
                        r"supporting (fake|real)\.$")


def route_batch(paths, ckpt, bs=128):
    from PIL import Image
    rc = RouterClient(ckpt)
    rc.load_model()

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as im:
                return rc._transform(im.convert("RGB"))

    out = []
    dl = torch.utils.data.DataLoader(DS(), batch_size=bs, num_workers=6)
    with torch.no_grad():
        for x in dl:
            out += torch.softmax(rc.model(x.to(rc.device)), 1).argmax(1).tolist()
    return [EXPERTS[i] for i in out]


def clause(label, alias, s):
    if label == 1 and s >= HI:
        return f" Additionally, the selected {alias} expert provides strong weak-feature evidence supporting fake."
    if label == 0 and s <= LO:
        return f" Additionally, the selected {alias} expert provides weak-feature evidence supporting real."
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", type=Path, required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--dff", type=int, default=6000)
    a = ap.parse_args()
    out = a.v3 / "05_sft"
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(a.v3 / "01_calibrators/scores_V3_train.csv").drop_duplicates("image_path").set_index(
        "image_path")
    held = set()
    for s in ("val", "test"):
        held |= set(pd.read_csv(a.v3 / f"01_calibrators/scores_V3_{s}.csv", usecols=["image_path"]).image_path)

    old = json.load(open(OLD_SFT))
    rows = []
    for it in old:
        g = it["conversations"][1]["value"]
        label = 1 if g.startswith("This image is fake") else 0
        rows.append(dict(image=it["image"], label=label, answer=OLD_CLAUSE.sub("", g), src="sft24k"))
    dff = tr[tr.dataset == "DeepFakeFace"].reset_index()
    rng = random.Random(42)
    pick = []
    for gen, n in (("real", a.dff // 2), ("sd15_text2img", a.dff // 4), ("sd_inpainting", a.dff - a.dff // 2 - a.dff // 4)):
        pool = dff[dff.generator == gen].image_path.tolist()
        pick += [(p, gen) for p in rng.sample(sorted(pool), n)]
    for p, gen in pick:
        label = 0 if gen == "real" else 1
        rows.append(dict(image=p, label=label, answer="This image is fake." if label else "This image is real.",
                         src=f"DFF_{gen}"))

    missing = [r["image"] for r in rows if r["image"] not in tr.index]
    leak = [r["image"] for r in rows if r["image"] in held]
    experts = route_batch([r["image"] for r in rows], a.router)
    items, stats = [], Counter()
    for r, e in zip(rows, experts):
        if r["image"] in tr.index:
            s = float(tr.loc[r["image"], f"cal_{e}"])  # OOF cal V3
        else:  # not a calibrator TRAIN row: final calibrator (never fitted on it)
            import joblib
            cal = joblib.load(a.v3 / "01_calibrators/calibrators_V3.joblib")["experts"]
            raise SystemExit(f"SFT image missing from TRAIN scores: {r['image']} (have calibrators {list(cal)})")
        c = clause(r["label"], e, s)
        stats[(r["src"], e, "support" if c else "none")] += 1
        items.append({"id": str(len(items)), "image": r["image"], "conversations": [
            {"from": "human", "value": f"<image>\nIs this image real or fake? And the {e} score is {s:.3f}."},
            {"from": "gpt", "value": r["answer"] + c}]})
    random.Random(20260915).shuffle(items)
    for i, it in enumerate(items):
        it["id"] = str(i)

    # checks (R4.3)
    assert all(it["conversations"][0]["value"].count("<image>") == 1 and
               it["conversations"][0]["value"].startswith("<image>\n") for it in items)
    assert all(Path(it["image"]).is_file() for it in items)
    assert not leak, leak[:3]
    assert not missing, missing[:3]
    sft = out / "router4_x2dfd_V3_SFT.json"
    json.dump(items, open(sft, "w"), ensure_ascii=False, indent=1)
    rep = dict(n=len(items), by_src=Counter(r["src"] for r in rows),
               alias_by_src={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(stats.items())},
               old_clause_removed=sum(1 for it in old if OLD_CLAUSE.search(it["conversations"][1]["value"])),
               leak_val_test=len(leak))
    (out / "R4_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()

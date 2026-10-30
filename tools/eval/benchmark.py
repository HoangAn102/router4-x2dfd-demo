"""Official benchmark of the live pipeline (FIX_PLAN PHASE 4).

Datasets (DeepfakeBench face crops already on disk -> NOT re-cropped; same frames for every system):
  FFPP    in-domain test videos of router_teacher_v2/test_calibrated_v2/FFPP_test_calibrated_v2.csv
  CDF     Celeb-DF-v2 List_of_testing_videos.txt (label 1 = real there)
  DFDCP   DFDCP dataset.json set == "test"
Per dataset: --videos videos (half real / half fake, seed 42), --frames frames evenly spaced per video.

Systems
  b  Router4 x X2DFD as the web runs it after the audit: RouterClient top-1 -> ExpertClient (selected expert,
     live calibrator) -> X2DFD LoRA with the train-parity prompt. Because that prompt depends only on
     (alias, score .3f), the LLaVA stage is read from llava_score_table.csv (tools/eval/llava_score_table.py,
     produced with the same worker functions) instead of re-running LLaVA per frame.
  a  X2DFD original (author runner + infer_config.yaml: Blending + Diffusion, LoRA [ble-diff]) on the same frames:
     --author-json writes the runner input; --author-out reads its output (run separately, needs a free GPU).

    python tools/eval/benchmark.py --out DIR --table TABLE.csv [--videos 100 --frames 8]
"""
import argparse
import asyncio
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from web.model_worker.expert_client import ExpertClient  # noqa: E402
from web.model_worker.router_client import EXPERTS, RouterClient  # noqa: E402

CROSS = Path("/home/aiotlab/hoangan/datasets/CROSSDATA_FINAL_EVAL")
FFPP_TEST = Path("/home/aiotlab/hoangan/outputs/router_teacher_v2/test_calibrated_v2/FFPP_test_calibrated_v2.csv")


def _pick_frames(files, n):
    files = sorted(files)
    if len(files) <= n:
        return files
    return [files[i] for i in np.linspace(0, len(files) - 1, n, dtype=int)]


def videos_ffpp():
    vids = defaultdict(list)
    lab = {}
    for r in csv.DictReader(open(FFPP_TEST)):
        v = str(Path(r["image_path"]).parent)
        vids[v].append(r["image_path"])
        lab[v] = int(r["label"])
    return {v: (lab[v], f) for v, f in vids.items()}


def videos_cdf():
    out = {}
    for line in open(CROSS / "Celeb-DF-v2/List_of_testing_videos.txt"):
        cdf_label, rel = line.split()
        sub, name = rel.split("/")
        d = CROSS / "Celeb-DF-v2" / sub / "frames" / Path(name).stem
        files = [str(p) for p in d.glob("*.png")]
        if files:
            out[str(d)] = (0 if cdf_label == "1" else 1, files)
    return out


def videos_dfdcp():
    meta = json.load(open(CROSS / "DFDCP/dataset.json"))
    out = {}
    for key, m in meta.items():
        if m["set"] != "test":
            continue
        top = key.split("/")[0]
        d = CROSS / "DFDCP" / top / "frames" / Path(key).stem
        files = [str(p) for p in d.glob("*.png")]
        if files:
            out[str(d)] = (1 if m["label"] == "fake" else 0, files)
    return out


def build_manifest(n_videos, n_frames, seed=42):
    rows = []
    for name, fn in (("FFPP", videos_ffpp), ("CDF", videos_cdf), ("DFDCP", videos_dfdcp)):
        vids = fn()
        rng = random.Random(seed)
        for label in (0, 1):
            pool = sorted(v for v, (lab, _) in vids.items() if lab == label)
            for v in rng.sample(pool, min(n_videos // 2, len(pool))):
                for f in _pick_frames(vids[v][1], n_frames):
                    rows.append(dict(dataset=name, video=v, label=label, path=f))
    return rows


def load_table(path):
    t = {}
    for r in csv.DictReader(open(path)):
        t[(r["alias"], r["score_3dp"])] = float(r["fake_prob"])
    return t


async def run_b(rows, table):
    rc, ec = RouterClient(), ExpertClient()
    groups = defaultdict(list)
    for r in rows:
        if r.get("b_expert"):
            continue
        e, _, conf, margin = rc.route_image(Path(r["path"]))
        r.update(b_expert=e, b_router_conf=conf, b_margin=margin)
        groups[e].append(r)
    # blending raw on every frame too (PHASE 5 policy P1 needs it); selected expert for the rest
    need = defaultdict(list)
    for r in rows:
        if r.get("blending_raw") in (None, ""):
            need["blending"].append(r)
        if r["b_expert"] != "blending" and r.get("b_raw") in (None, ""):
            need[r["b_expert"]].append(r)
    for e, rs in need.items():
        res = await ec.compute_expert_scores_batch(e, [Path(r["path"]) for r in rs])
        for r in rs:
            raw, cal = res[str(Path(r["path"]).resolve())]
            if e == "blending":
                r["blending_raw"], r["blending_cal"] = raw, cal
            if r["b_expert"] == e:
                r["b_raw"], r["b_cal"] = raw, cal
    for r in rows:
        if r["b_expert"] == "blending":
            r["b_raw"], r["b_cal"] = r["blending_raw"], r["blending_cal"]
        r["b_fake"] = table[(r["b_expert"], f"{float(r['b_cal']):.3f}")]


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def metrics(rows, col):
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds and r.get(col) not in (None, "")]
        if not rs:
            continue
        y = np.array([int(r["label"]) for r in rs])
        s = np.array([float(r[col]) for r in rs])
        pred = (s >= 0.5).astype(int)
        vids = defaultdict(list)
        vlab = {}
        for r in rs:
            vids[r["video"]].append(float(r[col]))
            vlab[r["video"]] = int(r["label"])
        vy = [vlab[v] for v in vids]
        vs = [float(np.mean(vids[v])) for v in vids]
        out[ds] = dict(
            frames=len(rs), videos=len(vids),
            frame_auc=auc(y, s), video_auc_mean=auc(vy, vs),
            acc=float((pred == y).mean()),
            fpr_real=float(pred[y == 0].mean()), recall_fake=float(pred[y == 1].mean()),
            video_acc=float(((np.array(vs) >= 0.5).astype(int) == np.array(vy)).mean()),
            tn=int(((pred == 0) & (y == 0)).sum()), fp=int(((pred == 1) & (y == 0)).sum()),
            fn=int(((pred == 0) & (y == 1)).sum()), tp=int(((pred == 1) & (y == 1)).sum()),
        )
    return out


def router_hist(rows):
    h = {}
    for ds in sorted({r["dataset"] for r in rows}):
        for lab in (0, 1):
            rs = [r for r in rows if r["dataset"] == ds and int(r["label"]) == lab and r.get("b_expert")]
            h[f"{ds}_{'fake' if lab else 'real'}"] = {e: sum(r["b_expert"] == e for r in rs) for e in EXPERTS}
    return h


def read_author(path, rows):
    """eval.infer.runner output: list of conversation items with absolute image path + fake score turn."""
    by = {}
    for it in json.load(open(path)):
        img = it.get("image")
        fake = None
        for t in it.get("conversations", []):
            v = str(t.get("value", ""))
            if t.get("from") in ("fake score", "fake_score"):
                fake = float(v)
        if fake is None and "fake_score" in it:
            fake = float(it["fake_score"])
        if img and fake is not None:
            by[str(Path(img).resolve())] = fake
    for r in rows:
        r["a_fake"] = by.get(str(Path(r["path"]).resolve()), "")


def plots(rows, out, systems):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    dsets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, len(dsets), figsize=(5 * len(dsets), 4.2))
    for ax, ds in zip(np.atleast_1d(axes), dsets):
        for col, name in systems:
            rs = [r for r in rows if r["dataset"] == ds and r.get(col) not in (None, "")]
            if len({int(r["label"]) for r in rs}) < 2:
                continue
            fpr, tpr, _ = roc_curve([int(r["label"]) for r in rs], [float(r[col]) for r in rs])
            ax.plot(fpr, tpr, label=f"{name} (AUC {auc([int(r['label']) for r in rs], [float(r[col]) for r in rs]):.3f})")
        ax.plot([0, 1], [0, 1], ":", color="grey")
        ax.set_title(f"{ds} — frame ROC")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_frame.png", dpi=130)

    h = router_hist(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    keys = list(h)
    bottom = np.zeros(len(keys))
    for e in EXPERTS:
        v = np.array([h[k][e] for k in keys], float)
        ax.bar(keys, v, bottom=bottom, label=e)
        bottom += v
    ax.set_ylabel("frames")
    ax.set_title("Router4 top-1 expert per dataset/label")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout()
    fig.savefig(out / "router_hist.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4))
    for ds in dsets:
        rs = [r for r in rows if r["dataset"] == ds]
        for lab, ls in ((0, "-"), (1, "--")):
            s = [float(r["b_fake"]) for r in rs if int(r["label"]) == lab]
            ax.hist(s, bins=40, range=(0, 1), histtype="step", linestyle=ls,
                    label=f"{ds} {'fake' if lab else 'real'}")
    ax.set_xlabel("Router4 x X2DFD fake_prob")
    ax.set_title("Score distribution (b)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "score_hist_b.png", dpi=130)


FIELDS = ["dataset", "video", "label", "path", "b_expert", "b_router_conf", "b_margin", "b_raw", "b_cal", "b_fake",
          "blending_raw", "blending_cal", "a_fake"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--videos", type=int, default=100)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N frames (speed probe)")
    ap.add_argument("--author-json", type=Path, help="write eval.infer.runner input JSON for system a and exit")
    ap.add_argument("--author-out", type=Path, help="read eval.infer.runner output for system a")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res_path = a.out / "frames.csv"
    if res_path.exists():
        rows = list(csv.DictReader(open(res_path)))
    else:
        rows = build_manifest(a.videos, a.frames)
        with open(a.out / "manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["dataset", "video", "label", "path"])
            w.writeheader()
            w.writerows(rows)
    if a.limit:
        rows = rows[:a.limit]
    if a.author_json:
        json.dump({"Description": "/", "images": [{"image_path": r["path"]} for r in rows]},
                  open(a.author_json, "w"))
        print("wrote", a.author_json, len(rows))
        return
    asyncio.run(run_b(rows, load_table(a.table)))
    if a.author_out:
        read_author(a.author_out, rows)
    with open(res_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    systems = [("b_fake", "Router4 x X2DFD (b)")]
    if any(r.get("a_fake") not in (None, "") for r in rows):
        systems.insert(0, ("a_fake", "X2DFD original (a)"))
    summary = {name: metrics(rows, col) for col, name in systems}
    summary["router_hist"] = router_hist(rows)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    plots(rows, a.out, systems)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

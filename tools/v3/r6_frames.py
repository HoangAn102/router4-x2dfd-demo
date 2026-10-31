"""R6 (full): frame table for the whole TEST splits + DEV, raw expert scores, routing and table-based systems.

TEST : FFPP test (700 videos), Celeb-DF-v2 test list (518), DFDCP test (videos with frames on disk), DFF test
DEV  : FFPP val (val_teacher_v2) + DFF val          (thresholds are chosen on DEV only)
Raw scores: stored v2 raws for FFPP test/val (passed parity, |diff| <= 2e-4), DFF from score_experts.py,
CDF/DFDCP computed with the live wrappers (resumable, chunked).
Systems filled here (no image LLaVA needed):
  b        Router v2 + calibrator v2 + LoRA FINAL_CLEAN score table (teacher-forced read-out)
  interim  Router v3 (D2) + calibrator D1 + score table
  plus the inputs of the image systems: V3_expert/V3_cal (calibrator V3), bold_cal (calibrator FINAL_CLEAN on the
  Router v2 choice, the pre-audit live configuration)

    python tools/v3/r6_frames.py --out DIR
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools/eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark as B  # noqa: E402
from build_sft_v3 import route_batch  # noqa: E402
from eval_r6 import SYS, TABLE  # noqa: E402
from web.model_worker.expert_client import ExpertClient  # noqa: E402

EXPERTS = ["blending", "diffusion", "frequency", "texture"]
V2 = Path("/home/aiotlab/hoangan/outputs/router_teacher_v2")
DFF = Path("/home/aiotlab/hoangan/datasets/DeepFakeFace_DFF")
CAL_FINAL_CLEAN = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/01_teacher/calibrators_FINAL_CLEAN.joblib"
SC = [f"score_{e}" for e in EXPERTS]
TABLE_TF = Path("/home/aiotlab/hoangan/eval_runs/root_fix/r6_full/tf_check/llava_score_table_tf.csv")  # check_tf_equivalence


def build():
    parts = []
    t = pd.read_csv(B.FFPP_TEST)
    t["video"] = t.image_path.map(lambda p: str(Path(p).parent))
    parts.append(t.assign(split="test", dataset="FFPP")[["split", "dataset", "video", "label", "image_path"] + SC])
    for name, fn in (("CDF", B.videos_cdf), ("DFDCP", B.videos_dfdcp)):
        rows = [dict(split="test", dataset=name, video=v, label=lab, image_path=f)
                for v, (lab, files) in fn().items() for f in sorted(files)]
        parts.append(pd.DataFrame(rows))
    for split, tag in (("test", "test"), ("val", "dev")):
        d = pd.read_csv(DFF / f"DFF_{split}_scores.csv")
        parts.append(d.assign(split=tag, dataset=f"DFF_{split}", video=d.image_path)[
            ["split", "dataset", "video", "label", "image_path"] + SC])
    v = pd.read_csv(V2 / "val_teacher_v2.csv", low_memory=False)
    v = v[v.dataset == "FFPP"]
    v["video"] = v.image_path.map(lambda p: str(Path(p).parent))
    parts.append(v.assign(split="dev", dataset="FFPP_val")[["split", "dataset", "video", "label", "image_path"] + SC])
    df = pd.concat(parts, ignore_index=True)
    df["label"] = df.label.astype(int)
    return df


async def fill_raw(df, save, chunk=4000):
    ec = ExpertClient()
    for e in EXPERTS:
        while df[f"score_{e}"].isna().any():
            idx = df.index[df[f"score_{e}"].isna()][:chunk]
            paths = [Path(p) for p in df.loc[idx, "image_path"]]
            res = await ec.compute_expert_scores_batch(e, paths)
            df.loc[idx, f"score_{e}"] = [res[str(p.resolve())][0] for p in paths]
            save()
            print(e, int(df[f"score_{e}"].isna().sum()), "left", flush=True)


def cal_selected(df, expert_col, cal):
    out = np.empty(len(df))
    for e in EXPERTS:
        m = (df[expert_col] == e).to_numpy()
        if m.any():
            out[m] = cal[e].predict(df.loc[m, f"score_{e}"].to_numpy(float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "frames_full.csv"
    df = pd.read_csv(path) if path.exists() else build()

    def save():
        df.to_csv(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").replace(path)

    save()
    print(df.groupby(["split", "dataset"]).agg(frames=("label", "size"), videos=("video", "nunique")), flush=True)
    asyncio.run(fill_raw(df, save))
    for name, (ckpt, calp, key) in SYS.items():
        if f"{name}_expert" not in df or df[f"{name}_expert"].isna().any():
            df[f"{name}_expert"] = route_batch(df.image_path.tolist(), ckpt)
            save()
        df[f"{name}_cal"] = cal_selected(df, f"{name}_expert", joblib.load(calp)[key])
    df["bold_cal"] = cal_selected(df, "b_expert", joblib.load(CAL_FINAL_CLEAN)["experts"])
    # (b)/(interim) LLaVA stage = score table of the web worker's read-out: teacher-forced since 31/10
    # (TABLE_TF); the generate-based table (TABLE) is kept as *_fake_gen for the equivalence record.
    for col, table_csv in (("_fake", TABLE_TF), ("_fake_gen", TABLE)):  # not `path`: save() writes to `path`
        t = pd.read_csv(table_csv)
        tab = {(al, f"{s:.3f}"): p for al, s, p in zip(t.alias, t.score_3dp, t.fake_prob)}
        for name in ("b", "interim"):
            df[f"{name}{col}"] = [tab[(e, f"{c:.3f}")] for e, c in zip(df[f"{name}_expert"], df[f"{name}_cal"])]
    save()
    rep = {f"{s}|{d}": dict(frames=int(len(g)), videos=int(g.video.nunique()),
                            routes={n: g[f"{n}_expert"].value_counts().to_dict() for n in SYS})
           for (s, d), g in df.groupby(["split", "dataset"])}
    (a.out / "frames_full_summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

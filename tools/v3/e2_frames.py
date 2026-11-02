"""E2 (PREREG_E2): frame table on the old report's manifests + R6 DFF/DEV rows, and the LLaVA job files.

  build   rows = old-report manifest (CDF, DFDC, DFDCP val) + FFPP development manifest + R6 DFF_test/DFF_val;
          raw expert scores copied from R6 frames_full.csv where the frame is the same file (CDF/DFDCP/FFPP/DFF)
  raw     fill missing raw scores (DFDC) with the live expert wrappers (resumable, chunked)
  route   Router V3 softmax for every row (top-1 expert + max prob); checks the top-1 against R6 V3_expert
  jobs    P0 / P1 (max prob < 0.30 -> blending) / ORACLE (all blending) experts + calibrated scores (calibrator V3),
          and the r6_batched.py input files of the frames whose LLaVA score is not known yet

    x2python tools/v3/e2_frames.py {build,raw,route,jobs} --out DIR
"""
import argparse
import asyncio
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RUNS = Path("/home/aiotlab/router4_x2dfd_FULL_SOURCE_20260924_054600/replication_author_audit/runs")
MANIFEST = RUNS / "ROUTER4_CURRENT_FROZEN_EXTERNAL_20261010/manifest.csv"
FFPP_DEV = RUNS / "ROUTER4_IMPROVEMENT_RESEARCH_V1/internal_ffpp_val_manifest.csv"
R6 = Path("/home/aiotlab/hoangan/eval_runs/root_fix/r6_full")
V3 = Path("/home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030")
ROUTER_V3 = V3 / "03_router/run_effb0_4domain_V3/best.pt"
CAL_V3 = V3 / "01_calibrators/calibrators_V3.joblib"
EXPERTS = ["blending", "diffusion", "frequency", "texture"]
SC = [f"score_{e}" for e in EXPERTS]
NAMES = {"Celeb-DF-v2": "CDF", "DFDC": "DFDC", "DFDCP": "DFDCP_val", "FFPP_INTERNAL_VAL": "FFPP_dev"}
LOW_CONF = 0.30  # PREREG_E2 §2 (config.ROUTER_LOW_CONFIDENCE)


def build(out):
    m = pd.concat([pd.read_csv(MANIFEST), pd.read_csv(FFPP_DEV)], ignore_index=True)
    m = m.assign(dataset=m.dataset.map(NAMES), video=m.official_video_key.astype(str), role="e2")
    r6 = pd.read_csv(R6 / "frames_full.csv")
    excl = set(pd.read_csv(R6 / "leak_check/exclude_frames.csv").image_path)
    dff = r6[r6.dataset.isin(["DFF_test", "DFF_val"]) & ~r6.image_path.isin(excl)]
    dff = dff.assign(role="r6", frame_order=0)[["dataset", "video", "label", "image_path", "role", "frame_order"]]
    df = pd.concat([m[["dataset", "video", "label", "image_path", "role", "frame_order"]], dff], ignore_index=True)
    df["dev"] = df.dataset.isin(["FFPP_dev", "DFF_val"])  # DEV = FFPP val (= FFPP development) + DFF val
    raw = r6.drop_duplicates("image_path").set_index("image_path")[SC]
    for c in SC:
        df[c] = df.image_path.map(raw[c])
    df["split"] = "test"  # r6_batched.py filter; every row here is scored
    df.to_csv(out / "frames_e2.csv", index=False)
    print(df.groupby("dataset").agg(frames=("label", "size"), videos=("video", "nunique"),
                                    raw_missing=("score_blending", lambda s: int(s.isna().sum()))))


async def fill_raw(df, save, chunk=4000):
    from web.model_worker.expert_client import ExpertClient
    ec = ExpertClient()
    for e in EXPERTS:
        while df[f"score_{e}"].isna().any():
            idx = df.index[df[f"score_{e}"].isna()][:chunk]
            paths = [Path(p) for p in df.loc[idx, "image_path"]]
            res = await ec.compute_expert_scores_batch(e, paths)
            df.loc[idx, f"score_{e}"] = [res[str(p.resolve())][0] for p in paths]
            save()
            print(e, int(df[f"score_{e}"].isna().sum()), "left", flush=True)


def route(df, bs=128):
    import torch
    from PIL import Image
    from web.model_worker.router_client import RouterClient
    rc = RouterClient(str(ROUTER_V3))
    rc.load_model()
    paths = df.image_path.tolist()

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as im:
                return rc._transform(im.convert("RGB"))

    probs = []
    with torch.no_grad():
        for i, x in enumerate(torch.utils.data.DataLoader(DS(), batch_size=bs, num_workers=6)):
            probs.append(torch.softmax(rc.model(x.to(rc.device)), 1).float().cpu().numpy())
            if i % 100 == 0:
                print("route", i * bs, "/", len(paths), flush=True)
    p = np.concatenate(probs)
    for j, e in enumerate(EXPERTS):
        df[f"p_{e}"] = p[:, j]
    df["P0_expert"] = [EXPERTS[i] for i in p.argmax(1)]
    df["router_conf"] = p.max(1)
    r6 = pd.read_csv(R6 / "frames_full.csv", usecols=["image_path", "V3_expert"]).drop_duplicates("image_path")
    j = df.merge(r6, on="image_path")
    print("top-1 == R6 V3_expert:", int((j.P0_expert == j.V3_expert).sum()), "/", len(j), flush=True)
    return df


def jobs(df, out):
    cal = joblib.load(CAL_V3)["experts"]

    def calibrate(expert_col):
        res = np.empty(len(df))
        for e in EXPERTS:
            k = (df[expert_col] == e).to_numpy()
            if k.any():
                res[k] = cal[e].predict(df.loc[k, f"score_{e}"].to_numpy(float))
        return res

    df["P1_expert"] = np.where(df.router_conf < LOW_CONF, "blending", df.P0_expert)
    df["OR_expert"] = "blending"
    for s in ("P0", "P1", "OR"):
        df[f"{s}_cal"] = calibrate(f"{s}_expert")
    (out / "jobs").mkdir(exist_ok=True)
    known = pd.read_csv(R6 / "results_V3.csv").drop_duplicates("image_path").set_index("image_path").V3_fake
    df["P0_known"] = df.image_path.isin(known.index)
    r6f = pd.read_csv(R6 / "frames_full.csv", usecols=["image_path", "V3_cal"]).drop_duplicates("image_path")
    chk = df.merge(r6f, on="image_path")
    print("P0 cal == R6 V3_cal (.3f):", int((chk.P0_cal.round(3) == chk.V3_cal.round(3)).sum()), "/", len(chk))
    cols = ["split", "dataset", "video", "label", "image_path", "V3_expert", "V3_cal"]
    # P0 for frames not in R6 (DFDC); P1/ORACLE only where the expert differs from P0 (same prompt => same score)
    df[~df.P0_known].assign(V3_expert=lambda d: d.P0_expert, V3_cal=lambda d: d.P0_cal)[cols].to_csv(
        out / "jobs/P0_new.csv", index=False)
    for s in ("P1", "OR"):
        d = df[df[f"{s}_expert"] != df.P0_expert]
        d.assign(V3_expert=d[f"{s}_expert"], V3_cal=d[f"{s}_cal"])[cols].to_csv(out / f"jobs/{s}_changed.csv",
                                                                                index=False)
    a_known = set(pd.read_csv(R6 / "results_a.csv").image_path)
    df[~df.image_path.isin(a_known)][cols[:5]].to_csv(out / "jobs/a_new.csv", index=False)
    df.to_csv(out / "frames_e2.csv", index=False)
    for f in sorted((out / "jobs").glob("*.csv")):
        print(f.name, len(pd.read_csv(f)))
    print(df.groupby("dataset").agg(low_conf=("router_conf", lambda s: float((s < LOW_CONF).mean())),
                                    p1_changed=("P1_expert", "size")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "raw", "route", "jobs"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "build":
        return build(a.out)
    path = a.out / "frames_e2.csv"
    df = pd.read_csv(path)

    def save():
        df.to_csv(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").replace(path)

    if a.stage == "raw":
        asyncio.run(fill_raw(df, save))
    elif a.stage == "route":
        route(df)
        save()
    else:
        jobs(df, a.out)


if __name__ == "__main__":
    main()

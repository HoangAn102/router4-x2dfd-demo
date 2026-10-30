"""ROOT_FIX R6: compare systems on the same frames (PHASE 4 benchmark frames + DeepFakeFace test).

  b        Router v2 + calibrator v2 + LoRA FINAL_CLEAN (train-parity prompt, no image)  -> score table
  interim  Router v3 (D2) + calibrator D1 + LoRA FINAL_CLEAN                            -> score table
  V3       Router V3 + calibrator V3 + LoRA V3, "<image>\\n" prompt, image fed exactly as in training,
           run through the live worker functions (web/model_worker/x2dfd_worker_cli.py)
  V3_gray  same prompts as V3 but the face replaced by a mid-grey 256x256 image (does the LoRA use the image?)
  a        X2DFD original: frames from PHASE 4 (a_fake column); DFF test via --author-out (eval.infer.runner output)
Raw expert scores are filled with the live wrappers where missing (resumable cache rows_r6.csv).

    python tools/v3/eval_r6.py --out DIR --lora-v3 DIR [--author-dff OUT.json] [--limit N]
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
sys.path.insert(0, str(REPO / "web/model_worker"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_sft_v3 import route_batch  # noqa: E402
from web.model_worker.expert_client import ExpertClient  # noqa: E402

EXPERTS = ["blending", "diffusion", "frequency", "texture"]
BENCH = Path("/home/aiotlab/hoangan/eval_runs/final_audit/phase4/frames_3way.csv")
DFF_TEST = Path("/home/aiotlab/hoangan/datasets/DeepFakeFace_DFF/DFF_test_scores.csv")
TABLE = Path("/home/aiotlab/hoangan/eval_runs/final_audit/phase2/score_table/llava_score_table.csv")
BASE = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"
V3 = Path("/home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030")
SYS = {
    "b": ("/home/aiotlab/hoangan/outputs/router_training_v2/run_effb0_3domain/best.pt",
          "/home/aiotlab/hoangan/outputs/router_teacher_v2/calibrators_train_only.joblib", "calibrators"),
    "interim": ("/home/aiotlab/hoangan/outputs/ROUTER4_DIFF_RETRAIN_20261030/02_router/run_effb0_4domain/best.pt",
                "/home/aiotlab/hoangan/outputs/ROUTER4_DIFF_RETRAIN_20261030/01_teacher/calibrators_v2_diffrefit.joblib",
                "calibrators"),
    "V3": (str(V3 / "03_router/run_effb0_4domain_V3/best.pt"), str(V3 / "01_calibrators/calibrators_V3.joblib"),
           "experts"),
}


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def load_rows():
    b = pd.read_csv(BENCH)
    b = b.rename(columns={"path": "image_path"})
    b["score_blending"] = b["blending_raw"]
    for e in EXPERTS[1:]:
        b[f"score_{e}"] = np.where(b.b_expert == e, b.b_raw, np.nan)
    d = pd.read_csv(DFF_TEST)
    d["dataset"], d["video"] = "DFF_test", d["image_path"]
    cols = ["dataset", "video", "label", "image_path", "a_fake"] + [f"score_{e}" for e in EXPERTS]
    return pd.concat([b.reindex(columns=cols), d.reindex(columns=cols)], ignore_index=True)


async def fill_raw(df):
    ec = ExpertClient()
    for e in EXPERTS:
        m = df[f"score_{e}"].isna()
        if m.any():
            paths = [Path(p) for p in df.loc[m, "image_path"]]
            res = await ec.compute_expert_scores_batch(e, paths)
            df.loc[m, f"score_{e}"] = [res[str(p.resolve())][0] for p in paths]


def calibrate(df, name):
    _, calp, key = SYS[name]
    cal = joblib.load(calp)[key]
    raw = np.array([r[f"score_{r[f'{name}_expert']}"] for _, r in df.iterrows()], float)
    out = np.empty(len(df))
    for e in EXPERTS:
        m = (df[f"{name}_expert"] == e).to_numpy()
        if m.any():
            out[m] = cal[e].predict(raw[m])
    return out


def metrics(df, col):
    res = {}
    for ds, g in df.groupby("dataset"):
        g = g[g[col].notna()]
        if g.empty:
            continue
        y, p = g.label.to_numpy(int), g[col].to_numpy(float)
        pred = (p >= 0.5).astype(int)
        r = dict(n=len(g), frame_auc=auc(y, p), acc=float((pred == y).mean()),
                 fpr_real=float(pred[y == 0].mean()), recall_fake=float(pred[y == 1].mean()))
        if g.video.nunique() < len(g):
            v = g.groupby("video").agg(y=("label", "first"), p=(col, "mean"))
            r["video_auc"] = auc(v.y, v.p)
        res[ds] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lora-v3", required=True)
    ap.add_argument("--author-dff", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "rows_r6.csv"
    df = pd.read_csv(cache) if cache.exists() else load_rows()
    if a.limit:
        df = df.groupby("dataset", group_keys=False).head(a.limit).reset_index(drop=True)
    asyncio.run(fill_raw(df))
    for name, (ckpt, _, _) in SYS.items():
        if f"{name}_expert" not in df or df[f"{name}_expert"].isna().any():
            df[f"{name}_expert"] = route_batch(df.image_path.tolist(), ckpt)
        df[f"{name}_cal"] = calibrate(df, name)
    t = pd.read_csv(TABLE)
    tab = {(al, f"{s:.3f}"): p for al, s, p in zip(t.alias, t.score_3dp, t.fake_prob)}
    for name in ("b", "interim"):
        df[f"{name}_fake"] = [tab[(e, f"{c:.3f}")] for e, c in zip(df[f"{name}_expert"], df[f"{name}_cal"])]
    df.to_csv(cache, index=False)

    import x2dfd_worker_cli as W
    from PIL import Image
    grey = a.out / "grey_256.png"
    Image.new("RGB", (256, 256), (128, 128, 128)).save(grey)
    for col, img_col in (("V3_fake", None), ("V3_gray_fake", grey)):
        if col not in df:
            df[col] = np.nan
        for i, r in df[df[col].isna()].iterrows():
            q = f"<image>\nIs this image real or fake? And the {r.V3_expert} score is {r.V3_cal:.3f}."
            res = W.score_with_train_prompt(q, model_path=a.lora_v3, model_base=BASE, max_new_tokens=4,
                                            image_path=str(img_col or r.image_path))
            df.at[i, col] = res["fake_score"]
            df.at[i, col.replace("_fake", "_step")] = res["label_step"]
            if i % 200 == 0:
                df.to_csv(cache, index=False)
                print(col, i, flush=True)
        df.to_csv(cache, index=False)
    if a.author_dff and a.author_dff.exists():
        sys.path.insert(0, str(REPO / "tools/eval"))
        import benchmark as B
        rows = [dict(path=p) for p in df.loc[df.dataset == "DFF_test", "image_path"]]
        B.read_author(a.author_dff, rows)
        df.loc[df.dataset == "DFF_test", "a_fake"] = [r["a_fake"] if r["a_fake"] != "" else np.nan for r in rows]
        df.to_csv(cache, index=False)
    rep = {name: metrics(df, f"{name}_fake") for name in ("a", "b", "interim", "V3", "V3_gray")
           if f"{name}_fake" in df}
    rep["router_hist"] = {name: {ds: g[f"{name}_expert"].value_counts().to_dict() for ds, g in df.groupby("dataset")}
                          for name in SYS}
    (a.out / "R6_summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "router_hist"}, indent=1))


if __name__ == "__main__":
    main()

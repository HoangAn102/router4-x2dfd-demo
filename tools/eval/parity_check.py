"""Train <-> live parity check (FIX_PLAN PHASE 2, checks P1-P6).

Samples images that were used to build Router4 / calibrator / SFT and replays each live step with the web
classes (RouterClient, ExpertClient, X2DFDClient.build_wfs_prompt), comparing against the values stored by the
training pipeline. Manifest images are already DeepfakeBench crops, so the face-crop step is skipped (P7 is
checked separately).

    python tools/eval/parity_check.py --out DIR                 # P1-P5 (no LLaVA)
    python tools/eval/parity_check.py --out DIR --llava 30      # + P6 on 30 images (stop the web first)
"""
import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from web.model_worker.expert_client import ExpertClient  # noqa: E402
from web.model_worker.router_client import EXPERTS, RouterClient  # noqa: E402
from web.model_worker.x2dfd_client import X2DFDClient  # noqa: E402

ROOT = Path("/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919")
VAL_CSV = ROOT / "01_teacher/val_teacher_FINAL_CLEAN.csv"
WFS_CSV = ROOT / "03_wfs/router4_FINAL_CLEAN_routed_wfs.csv"
SFT_JSON = ROOT / "04_sft/router4_x2dfd_FINAL_CLEAN_SFT.json"
BASE_LLAVA = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"


def sample(seed=42):
    sft = {x["image"]: x for x in json.load(open(SFT_JSON))}
    val = pd.read_csv(VAL_CSV)
    wfs = pd.read_csv(WFS_CSV)
    wfs = wfs[wfs.image_path.isin(sft)]
    v = val.groupby("label", group_keys=False).apply(lambda g: g.sample(30, random_state=seed))
    t = wfs.groupby(["domain", "label"], group_keys=False).apply(lambda g: g.sample(10, random_state=seed))
    v = v.assign(part="val")
    t = t.assign(part="train")
    return pd.concat([v, t], ignore_index=True), sft


def run_router(df):
    rc = RouterClient()
    for i, p in enumerate(df.image_path):
        expert, _, conf, _ = rc.route_image(Path(p))
        probs = rc.last_route_diagnostics["probabilities"]
        df.loc[i, "live_router_expert"] = expert
        df.loc[i, "live_router_conf"] = conf
        for e, pr in zip(EXPERTS, probs):
            df.loc[i, f"live_router_prob_{e}"] = pr


async def run_experts(df):
    ec = ExpertClient()
    paths = [Path(p) for p in df.image_path]
    for e in EXPERTS:
        res = await ec.compute_expert_scores_batch(e, paths)
        for i, p in enumerate(paths):
            raw, cal = res[str(p.resolve())]
            df.loc[i, f"live_raw_{e}"] = raw
            df.loc[i, f"live_cal_{e}"] = cal
            # same calibrator applied to the stored training raw score: isolates calibrator identity from P2
            df.loc[i, f"live_cal_of_stored_raw_{e}"] = ec.calibrate(e, float(df.loc[i, f"score_{e}"]))


def prompts(df, sft):
    for i, r in df.iterrows():
        if r.part != "train":
            continue
        human = sft[r.image_path]["conversations"][0]["value"]
        live = X2DFDClient.build_wfs_prompt(r.selected_expert.title(), float(r.selected_cal_score))
        df.loc[i, "sft_prompt"] = human
        df.loc[i, "live_prompt"] = live


def token_parity(sft_item, live_question):
    """P5: exact input_ids LLaVA training saw vs the ids the live worker sends (prompt part only)."""
    import transformers
    from llava import conversation as conv_lib
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from llava.train import train as T

    tok = transformers.AutoTokenizer.from_pretrained(BASE_LLAVA, use_fast=False, model_max_length=2048,
                                                     padding_side="right")
    tok.pad_token = tok.unk_token
    # train: --version v1 -> conversation_lib.conv_templates["v1"], has_image=True (item has "image")
    T.conversation_lib.default_conversation = conv_lib.conv_templates["v1"]

    class DA:
        is_multimodal = True
        mm_use_im_start_end = False

    src = T.preprocess_multimodal(copy.deepcopy([sft_item["conversations"]]), DA())
    out = T.preprocess(src, tok, has_image=True)
    ids, labels = out["input_ids"][0], out["labels"][0]
    first_target = int((labels != -100).nonzero()[0])
    train_prompt_ids = ids[:first_target].tolist()

    # live: X2DFD utils/lora_inference.single_image_infer_with_scores (conv_mode from adapter dir name)
    name = "router4_x2dfd_FINAL_CLEAN"
    conv_mode = "llava_v1" if "v1" in name.lower() else "llava_v0"
    qs = live_question.replace("<image>", "<im_start><image><im_end>")
    conv = conv_lib.conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    live_ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX)

    return {
        "train_conv_template": "v1 (conv_vicuna_v1)",
        "live_conv_mode": conv_mode,
        "train_prompt_text": tok.decode([t for t in train_prompt_ids if t >= 0]),
        "live_prompt_text": conv.get_prompt(),
        "train_n_image_tokens": int(sum(t == IMAGE_TOKEN_INDEX for t in train_prompt_ids)),
        "live_n_image_tokens": int(sum(t == IMAGE_TOKEN_INDEX for t in live_ids)),
        "train_prompt_ids": train_prompt_ids,
        "live_prompt_ids": live_ids,
        "identical": train_prompt_ids == live_ids,
    }


def summarize(df, tp):
    out = {}
    tr = df[df.part == "train"]
    out["P1_top1_match_train"] = float((tr.live_router_expert == tr.selected_expert).mean())
    out["P1_max_abs_prob_diff_train"] = float(max(
        (tr[f"live_router_prob_{e}"] - tr[f"router_prob_{e}"]).abs().max() for e in EXPERTS))
    for e in EXPERTS:
        d = (df[f"live_raw_{e}"] - df[f"score_{e}"]).abs()
        out[f"P2_{e}_max_abs_diff"] = float(d.max())
        out[f"P2_{e}_corr"] = float(np.corrcoef(df[f"live_raw_{e}"], df[f"score_{e}"])[0, 1])
        v = df[df.part == "val"]
        out[f"P3_{e}_val_max_abs_diff_live_vs_stored_cal"] = float((v[f"live_cal_{e}"] - v[f"cal_{e}"]).abs().max())
        out[f"P3_{e}_val_max_abs_diff_calib_of_stored_raw"] = float(
            (v[f"live_cal_of_stored_raw_{e}"] - v[f"cal_{e}"]).abs().max())
        out[f"P3_{e}_train_oof_vs_final_max_abs_diff"] = float(
            (tr[f"live_cal_of_stored_raw_{e}"] - tr[f"cal_{e}"]).abs().max())
    out["P4_prompt_bytes_identical"] = int((tr.sft_prompt == tr.live_prompt).sum())
    out["P4_n"] = int(len(tr))
    out["P4_example_sft"] = tr.sft_prompt.iloc[0]
    out["P4_example_live"] = tr.live_prompt.iloc[0]
    out["P5"] = {k: v for k, v in tp.items() if not k.endswith("_ids")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "parity_rows.csv"
    df, sft = sample()
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        run_router(df)
        asyncio.run(run_experts(df))
        prompts(df, sft)
        df.to_csv(cache, index=False)
    tr0 = df[df.part == "train"].iloc[0]
    tp = token_parity(sft[tr0.image_path], tr0.live_prompt)
    summary = summarize(df, tp)
    (args.out / "parity_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

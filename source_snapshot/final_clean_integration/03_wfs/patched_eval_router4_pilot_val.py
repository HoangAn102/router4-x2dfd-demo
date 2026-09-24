import os
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

BASE = Path("/home/aiotlab/hoangan")

VAL_CSV = BASE / "outputs/router_teacher_v2/val_teacher_v2.csv"

ROUTER_CKPT = (
    BASE
    / "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/02_router/final_checkpoint/best.pt"
)

LLAVA_BASE = (
    BASE
    / "projects/X2DFD/weights/base/llava-v1.5-7b"
)

LORA = (
    BASE
    / "outputs/x2dfd_router4_training/pilot_v1/"
      "lora_router4_pilot_v1_infer"
)

OUT_DIR = (
    BASE
    / "outputs/x2dfd_router4_training/pilot_v1/val_eval"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTED_CSV = OUT_DIR / "val_routed_wfs.csv"
PRED_CSV = OUT_DIR / "val_pilot_predictions.csv"

# Pilot only
N_LLAVA = 600
SEED = 20260912

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# HELPERS
# ============================================================

def canonical_expert(x):
    s = str(x).lower()

    if "blend" in s:
        return "blending"
    if "diff" in s:
        return "diffusion"
    if "freq" in s:
        return "frequency"
    if "text" in s:
        return "texture"

    raise ValueError(f"Unknown expert name: {x}")


def alias_name(x):
    return {
        "blending": "Blending",
        "diffusion": "Diffusion",
        "frequency": "Frequency",
        "texture": "Texture",
    }[x]


def normalize_label(x):
    if pd.isna(x):
        raise ValueError("NaN label")

    if isinstance(x, str):
        s = x.strip().lower()

        if s in ["real", "0", "false"]:
            return 0
        if s in ["fake", "1", "true", "deepfake"]:
            return 1

    v = int(float(x))

    if v not in [0, 1]:
        raise ValueError(f"Unsupported label: {x}")

    return v


def detect_label_col(df):
    for c in ["label", "target", "gt", "y", "class"]:
        if c in df.columns:
            return c

    raise RuntimeError(
        "Cannot detect GT label column. "
        f"Columns={list(df.columns)}"
    )


# ============================================================
# LOAD VAL
# ============================================================

print("\n========== LOAD VAL ==========")

df = pd.read_csv(VAL_CSV)

print("rows =", len(df))

if "image_path" not in df.columns:
    raise RuntimeError("VAL CSV has no image_path column")

label_col = detect_label_col(df)

print("label column =", label_col)

df["label_norm"] = df[label_col].map(normalize_label)

exists = df["image_path"].map(
    lambda p: os.path.isfile(str(p))
)

print("existing images =", int(exists.sum()), "/", len(df))

if not exists.all():
    print("missing =", int((~exists).sum()))

df = df[exists].copy().reset_index(drop=True)

# ============================================================
# LOAD ROUTER CHECKPOINT
# ============================================================

print("\n========== LOAD ROUTER ==========")

ckpt = torch.load(
    ROUTER_CKPT,
    map_location="cpu"
)

experts_raw = ckpt["experts"]
experts = [canonical_expert(x) for x in experts_raw]

print("checkpoint experts =", experts_raw)
print("canonical experts  =", experts)

num_experts = len(experts)

router = models.efficientnet_b0(weights=None)

in_features = router.classifier[1].in_features

router.classifier[1] = nn.Linear(
    in_features,
    num_experts
)

router.load_state_dict(
    ckpt["model"],
    strict=True
)

router.to(DEVICE)
router.eval()

print("Router load OK")

# ============================================================
# EXACT VAL TRANSFORM FROM SAVED ROUTER RUN
# Resize256 -> CenterCrop224 -> ImageNetNormalize
# ============================================================

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class RouterValDataset(Dataset):
    def __init__(self, frame):
        self.df = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["image_path"]

        with Image.open(path) as im:
            im = im.convert("RGB")
            x = val_transform(im)

        return x, idx


loader = DataLoader(
    RouterValDataset(df),
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# ============================================================
# ROUTE ALL VAL
# ============================================================

print("\n========== ROUTE ALL VAL ==========")

all_idx = []
all_route = []
all_conf = []
all_margin = []

with torch.inference_mode():

    for step, (images, idxs) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        with torch.cuda.amp.autocast(
            enabled=(DEVICE == "cuda")
        ):
            logits = router(images)

        probs = torch.softmax(
            logits.float(),
            dim=1
        )

        top2 = torch.topk(
            probs,
            k=min(2, num_experts),
            dim=1
        )

        route = probs.argmax(dim=1)

        conf = top2.values[:, 0]

        if num_experts >= 2:
            margin = (
                top2.values[:, 0]
                - top2.values[:, 1]
            )
        else:
            margin = conf.clone()

        all_idx.extend(
            idxs.numpy().tolist()
        )

        all_route.extend(
            route.cpu().numpy().tolist()
        )

        all_conf.extend(
            conf.cpu().numpy().tolist()
        )

        all_margin.extend(
            margin.cpu().numpy().tolist()
        )

        if step % 50 == 0:
            done = min(
                (step + 1) * 128,
                len(df)
            )

            print(
                f"router: {done}/{len(df)}"
            )

route_df = pd.DataFrame({
    "_idx": all_idx,
    "router_expert_idx": all_route,
    "router_confidence": all_conf,
    "router_margin": all_margin,
})

df = df.merge(
    route_df,
    left_index=True,
    right_on="_idx",
    how="left"
)

df.drop(
    columns=["_idx"],
    inplace=True
)

df["selected_expert"] = df[
    "router_expert_idx"
].map(
    lambda i: experts[int(i)]
)

df["selected_alias"] = df[
    "selected_expert"
].map(alias_name)

# ============================================================
# TAKE CALIBRATED SCORE OF ROUTED EXPERT
# ============================================================

needed_cal = [
    "cal_blending",
    "cal_diffusion",
    "cal_frequency",
    "cal_texture",
]

for c in needed_cal:
    if c not in df.columns:
        raise RuntimeError(
            f"Missing calibrated score column: {c}"
        )


def selected_score(row):
    c = "cal_" + row["selected_expert"]
    return float(row[c])


df["selected_cal_score"] = df.apply(
    selected_score,
    axis=1
)

df.to_csv(
    ROUTED_CSV,
    index=False
)

print("\nSaved:", ROUTED_CSV)

# ============================================================
# ROUTE DISTRIBUTION SANITY CHECK
# ============================================================

print("\n========== ROUTE DISTRIBUTION ==========")

print(
    df["selected_expert"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .to_string()
)

if "dataset" in df.columns:

    print("\n========== PER-DOMAIN ROUTES (%) ==========")

    tab = pd.crosstab(
        df["dataset"],
        df["selected_expert"],
        normalize="index"
    ) * 100

    print(
        tab.round(2).to_string()
    )

# ============================================================
# STRATIFIED SUBSET FOR LLaVA
# dataset x label x routed expert
# ============================================================

print(
    f"\n========== SELECT {N_LLAVA} FOR LoRA VAL =========="
)

group_cols = [
    "label_norm",
    "selected_expert"
]

if "dataset" in df.columns:
    group_cols.insert(0, "dataset")

groups = list(
    df.groupby(
        group_cols,
        dropna=False
    )
)

quota = max(
    1,
    math.ceil(
        N_LLAVA / max(1, len(groups))
    )
)

parts = []

for _, g in groups:
    n = min(
        quota,
        len(g)
    )

    parts.append(
        g.sample(
            n=n,
            random_state=SEED
        )
    )

sub = pd.concat(
    parts,
    ignore_index=False
)

if len(sub) > N_LLAVA:
    sub = sub.sample(
        n=N_LLAVA,
        random_state=SEED
    )

elif len(sub) < N_LLAVA:

    remaining = df.drop(
        index=sub.index,
        errors="ignore"
    )

    n_extra = min(
        N_LLAVA - len(sub),
        len(remaining)
    )

    if n_extra > 0:
        extra = remaining.sample(
            n=n_extra,
            random_state=SEED
        )

        sub = pd.concat(
            [sub, extra]
        )

sub = sub.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

print("LoRA VAL subset =", len(sub))

print("\nSubset labels:")
print(
    sub["label_norm"]
    .value_counts()
    .to_string()
)

print("\nSubset experts:")
print(
    sub["selected_expert"]
    .value_counts()
    .to_string()
)

if "dataset" in sub.columns:
    print("\nSubset datasets:")
    print(
        sub["dataset"]
        .value_counts()
        .to_string()
    )

# Free router VRAM
del router
torch.cuda.empty_cache()

# ============================================================
# LOAD ROUTER4 LoRA
# ============================================================

print("\n========== LOAD ROUTER4 LoRA ==========")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

tokenizer, model, image_processor, context_len = (
    load_pretrained_model(
        model_path=str(LORA),
        model_base=str(LLAVA_BASE),
        model_name="llava-v1.5-7b-router4-lora",
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
    )
)

model.eval()

print("LoRA load OK")

# ============================================================
# LoRA INFERENCE
# ============================================================

print("\n========== LoRA VAL INFERENCE ==========")

records = []


def parse_prediction(answer):
    s = answer.strip().lower()

    # Pilot target is exact real/fake text.
    has_real = bool(
        re.search(r"\breal\b", s)
    )

    has_fake = bool(
        re.search(r"\bfake\b", s)
    )

    if has_fake and not has_real:
        return 1

    if has_real and not has_fake:
        return 0

    return -1


for i, row in sub.iterrows():

    image_path = row["image_path"]

    alias = row["selected_alias"]

    score = float(
        row["selected_cal_score"]
    )

    question = (
        "<image>\n"
        "Is this image real or fake? "
        f"And the {alias} score is {score:.3f}."
    )

    with Image.open(image_path) as im:
        image = im.convert("RGB")

    conv = conv_templates["v1"].copy()

    conv.append_message(
        conv.roles[0],
        question
    )

    conv.append_message(
        conv.roles[1],
        None
    )

    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    image_tensor = process_images(
        [image],
        image_processor,
        model.config
    )

    if isinstance(image_tensor, list):

        image_tensor = [
            x.to(
                model.device,
                dtype=torch.float16
            )
            for x in image_tensor
        ]

    else:
        image_tensor = image_tensor.to(
            model.device,
            dtype=torch.float16
        )

    with torch.inference_mode():

        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=16,
            use_cache=True,
        )

    # IMPORTANT:
    # LLaVA generate here returns generated sequence.
    answer = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip()

    pred = parse_prediction(
        answer
    )

    records.append({
        "image_path": image_path,
        "dataset": row.get(
            "dataset",
            "UNKNOWN"
        ),
        "label": int(
            row["label_norm"]
        ),
        "router_expert_idx": int(
            row["router_expert_idx"]
        ),
        "selected_expert": row[
            "selected_expert"
        ],
        "selected_alias": alias,
        "router_confidence": float(
            row["router_confidence"]
        ),
        "router_margin": float(
            row["router_margin"]
        ),
        "selected_cal_score": score,
        "question": question,
        "answer": answer,
        "pred": pred,
    })

    if (
        (i + 1) % 25 == 0
        or i == 0
    ):
        print(
            f"{i+1}/{len(sub)} | "
            f"GT={int(row['label_norm'])} "
            f"PRED={pred} "
            f"{alias}={score:.3f} "
            f"| {answer!r}"
        )

pred_df = pd.DataFrame(
    records
)

pred_df.to_csv(
    PRED_CSV,
    index=False
)

# ============================================================
# STRICT METRICS
# Unknown output counts as incorrect
# ============================================================

y = pred_df["label"].to_numpy()
p = pred_df["pred"].to_numpy()

correct = (
    y == p
)

acc = correct.mean()

real_mask = (
    y == 0
)

fake_mask = (
    y == 1
)

real_recall = (
    ((p == 0) & real_mask).sum()
    / max(1, real_mask.sum())
)

fake_recall = (
    ((p == 1) & fake_mask).sum()
    / max(1, fake_mask.sum())
)

bacc = (
    real_recall
    + fake_recall
) / 2

unknown = int(
    (p == -1).sum()
)

tn = int(
    ((y == 0) & (p == 0)).sum()
)

fp = int(
    ((y == 0) & (p == 1)).sum()
)

fn = int(
    ((y == 1) & (p == 0)).sum()
)

tp = int(
    ((y == 1) & (p == 1)).sum()
)

print("\n")
print("=" * 60)
print("ROUTER4-X2DFD PILOT — HELD-OUT SEEN-DOMAIN VAL")
print("=" * 60)

print(f"N             = {len(pred_df)}")
print(f"ACC           = {acc*100:.2f}%")
print(f"BACC          = {bacc*100:.2f}%")
print(f"Real Recall   = {real_recall*100:.2f}%")
print(f"Fake Recall   = {fake_recall*100:.2f}%")
print(f"Unknown pred  = {unknown}")

print("\nVALID BINARY CONFUSION COUNTS")
print(f"TN={tn}  FP={fp}")
print(f"FN={fn}  TP={tp}")

print("\nPredictions:")
print(PRED_CSV)

print("\nRouted full VAL:")
print(ROUTED_CSV)

print("=" * 60)

# ============================================================
# PER-DOMAIN DIAGNOSTIC
# ============================================================

if "dataset" in pred_df.columns:

    print("\n========== PER-DOMAIN ==========")

    for domain, g in pred_df.groupby(
        "dataset"
    ):

        yy = g["label"].to_numpy()
        pp = g["pred"].to_numpy()

        rr_mask = yy == 0
        ff_mask = yy == 1

        a = (
            yy == pp
        ).mean()

        rr = (
            ((pp == 0) & rr_mask).sum()
            / max(1, rr_mask.sum())
        )

        fr = (
            ((pp == 1) & ff_mask).sum()
            / max(1, ff_mask.sum())
        )

        ba = (rr + fr) / 2

        print(
            f"{domain:20s} "
            f"N={len(g):4d} "
            f"ACC={a*100:6.2f}% "
            f"BACC={ba*100:6.2f}% "
            f"R={rr*100:6.2f}% "
            f"F={fr*100:6.2f}%"
        )

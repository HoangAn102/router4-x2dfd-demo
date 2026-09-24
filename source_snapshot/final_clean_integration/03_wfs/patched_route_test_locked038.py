import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

BASE = Path("/home/aiotlab/hoangan")

CKPT = (
    BASE /
    "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/02_router/final_checkpoint/best.pt"
)

IN_DIR = (
    BASE /
    "outputs/router_teacher_v2/test_calibrated_v2"
)

OUT_DIR = (
    BASE /
    "outputs/x2dfd_router4_training/pilot_v1/test_routed_locked038"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "FFPP":
        IN_DIR / "FFPP_test_calibrated_v2.csv",

    "GANGen":
        IN_DIR / "GANGen_test_calibrated_v2.csv",

    "StyleGAN_FFHQ":
        IN_DIR / "StyleGAN_FFHQ_test_calibrated_v2.csv",
}

THRESHOLD = 0.38

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# CHECKPOINT
# ============================================================

ckpt = torch.load(
    CKPT,
    map_location="cpu"
)

experts_raw = ckpt["experts"]

def canon(x):
    s = str(x).lower()

    if "blend" in s:
        return "blending"
    if "diff" in s:
        return "diffusion"
    if "freq" in s:
        return "frequency"
    if "text" in s:
        return "texture"

    raise ValueError(x)

experts = [
    canon(x)
    for x in experts_raw
]

print("Checkpoint experts =", experts_raw)
print("Canonical experts  =", experts)
print("Locked threshold   =", THRESHOLD)

# ============================================================
# MODEL
# ============================================================

model = models.efficientnet_b0(
    weights=None
)

in_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    in_features,
    len(experts)
)

model.load_state_dict(
    ckpt["model"],
    strict=True
)

model.to(DEVICE)
model.eval()

# Exact saved Router v2 VAL preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class DS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        path = str(
            self.df.iloc[idx]["image_path"]
        )

        with Image.open(path) as im:
            im = im.convert("RGB")
            image = transform(im)

        return image, idx


for dataset, path in FILES.items():

    print("\n" + "="*70)
    print(dataset)
    print("="*70)

    df = pd.read_csv(
        path,
        low_memory=False
    )

    print("rows =", len(df))

    # IMPORTANT:
    # label column is NEVER accessed below.

    missing = (
        ~df["image_path"]
        .astype(str)
        .map(os.path.isfile)
    )

    if missing.any():
        raise RuntimeError(
            f"{dataset}: "
            f"{int(missing.sum())} missing images"
        )

    loader = DataLoader(
        DS(df),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    indices = []
    routes = []
    confidences = []
    margins = []

    with torch.inference_mode():

        for step, (images, idx) in enumerate(loader):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            with torch.cuda.amp.autocast(
                enabled=(DEVICE == "cuda")
            ):
                logits = model(images)

            probs = torch.softmax(
                logits.float(),
                dim=1
            )

            top2 = torch.topk(
                probs,
                k=2,
                dim=1
            )

            route = probs.argmax(
                dim=1
            )

            conf = top2.values[:, 0]

            margin = (
                top2.values[:, 0]
                - top2.values[:, 1]
            )

            indices.extend(
                idx.numpy().tolist()
            )

            routes.extend(
                route.cpu()
                .numpy()
                .tolist()
            )

            confidences.extend(
                conf.cpu()
                .numpy()
                .tolist()
            )

            margins.extend(
                margin.cpu()
                .numpy()
                .tolist()
            )

            if step % 50 == 0:
                done = min(
                    (step + 1) * 128,
                    len(df)
                )

                print(
                    f"route {done}/{len(df)}"
                )

    result = pd.DataFrame({
        "_idx": indices,
        "router_expert_idx": routes,
        "router_confidence": confidences,
        "router_margin": margins,
    })

    df = df.merge(
        result,
        left_index=True,
        right_on="_idx",
        how="left"
    )

    df.drop(
        columns=["_idx"],
        inplace=True
    )

    df["router_expert"] = (
        df["router_expert_idx"]
        .map(
            lambda i: experts[int(i)]
        )
    )

    # ----------------------------
    # PURE ROUTER SCORE
    # ----------------------------

    def get_cal(row, expert):
        return float(
            row[
                "cal_" + expert
            ]
        )

    df["pure_selected_score"] = [
        get_cal(row, e)
        for (_, row), e in zip(
            df.iterrows(),
            df["router_expert"]
        )
    ]

    # ----------------------------
    # LOCKED CONFIDENCE GATE
    # ----------------------------

    fallback = (
        df["router_confidence"]
        < THRESHOLD
    )

    df["gate_fallback_blending"] = (
        fallback
    )

    df["gate_selected_expert"] = np.where(
        fallback,
        "blending",
        df["router_expert"]
    )

    df["gate_selected_score"] = [
        get_cal(row, e)
        for (_, row), e in zip(
            df.iterrows(),
            df["gate_selected_expert"]
        )
    ]

    out = (
        OUT_DIR /
        f"{dataset}_test_routed_locked038.csv"
    )

    df.to_csv(
        out,
        index=False
    )

    print("\nPURE ROUTER DISTRIBUTION:")

    print(
        df["router_expert"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )

    print("\nLOCKED-GATE DISTRIBUTION:")

    print(
        df["gate_selected_expert"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )

    print(
        "\nFallback to Blending:",
        int(fallback.sum()),
        "/",
        len(df),
        "=",
        f"{fallback.mean()*100:.2f}%"
    )

    print(
        "Router conf mean =",
        f"{df['router_confidence'].mean():.4f}"
    )

    print(
        "Router conf median =",
        f"{df['router_confidence'].median():.4f}"
    )

    print("\nSAVED:", out)

print("\nALL TEST ROUTING COMPLETE")

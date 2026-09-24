from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from torchvision import transforms
from torchvision.models import efficientnet_b0


BASE = Path("/home/aiotlab/hoangan")

SRC = (
    BASE
    / "outputs/router_teacher_v2"
    / "train_teacher_v2.csv"
)

ROUTER = (
    BASE
    / "outputs/router_training_v2"
    / "run_effb0_3domain"
    / "best.pt"
)

OUT = (
    BASE
    / "outputs/x2dfd_router4_training"
    / "wfs"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPERTS = [
    "blending",
    "diffusion",
    "frequency",
    "texture",
]

ALIASES = {
    "blending": "Blending",
    "diffusion": "Diffusion",
    "frequency": "Frequency",
    "texture": "Texture",
}


tf = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


class DS(Dataset):

    def __init__(self, df):

        self.df = df.reset_index(
            drop=True
        )


    def __len__(self):

        return len(self.df)


    def __getitem__(self, i):

        path = str(
            self.df.iloc[i][
                "image_path"
            ]
        )

        with Image.open(path) as im:

            im = im.convert("RGB")

            x = tf(im)

        return (
            x,
            i,
        )


df = pd.read_csv(
    SRC,
    low_memory=False,
)


required = []

for e in EXPERTS:

    required += [
        f"score_{e}",
        f"cal_{e}",
    ]


missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:

    raise RuntimeError(
        f"Missing expert columns: {missing}"
    )


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = efficientnet_b0(
    weights=None
)

nf = (
    model.classifier[1]
    .in_features
)

model.classifier[1] = nn.Linear(
    nf,
    len(EXPERTS),
)


ckpt = torch.load(
    ROUTER,
    map_location="cpu",
)

model.load_state_dict(
    ckpt["model"]
)

model = model.to(
    device
).eval()


loader = DataLoader(
    DS(df),

    batch_size=128,

    shuffle=False,

    num_workers=8,

    pin_memory=True,

    persistent_workers=True,
)


route_idx = np.zeros(
    len(df),
    dtype=np.int64,
)

route_conf = np.zeros(
    len(df),
    dtype=np.float32,
)

route_margin = np.zeros(
    len(df),
    dtype=np.float32,
)


print("=" * 80)
print("FROZEN ROUTER v2 -> TRAIN WFS")
print("=" * 80)

print(
    "TRAIN images =",
    len(df)
)

print(
    "Experts are NOT trained or executed."
)

print(
    "Existing expert CSV scores are reused."
)


with torch.no_grad():

    for batch_id, (
        images,
        indices,
    ) in enumerate(
        loader,
        start=1,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(
            enabled=(
                device.type
                == "cuda"
            )
        ):

            probs = torch.softmax(
                model(images),
                dim=1,
            )


        top2 = probs.topk(
            2,
            dim=1,
        )


        idx = indices.numpy()


        route_idx[idx] = (
            top2.indices[:, 0]
            .cpu()
            .numpy()
        )

        route_conf[idx] = (
            top2.values[:, 0]
            .cpu()
            .numpy()
        )

        route_margin[idx] = (
            (
                top2.values[:, 0]
                -
                top2.values[:, 1]
            )
            .cpu()
            .numpy()
        )


        if batch_id % 200 == 0:

            print(
                f"batches={batch_id} "
                f"processed≈"
                f"{min(batch_id*128, len(df))}/"
                f"{len(df)}",
                flush=True,
            )


df["router_expert_idx"] = (
    route_idx
)

df["selected_expert"] = [
    EXPERTS[i]
    for i in route_idx
]

df["selected_alias"] = [
    ALIASES[
        EXPERTS[i]
    ]
    for i in route_idx
]

df["router_confidence"] = (
    route_conf
)

df["router_margin"] = (
    route_margin
)


raw_scores = []
cal_scores = []


for _, row in df.iterrows():

    e = row[
        "selected_expert"
    ]

    raw_scores.append(
        float(
            row[
                f"score_{e}"
            ]
        )
    )

    cal_scores.append(
        float(
            row[
                f"cal_{e}"
            ]
        )
    )


df["selected_raw_score"] = (
    raw_scores
)

df["selected_cal_score"] = (
    cal_scores
)


# Primary experiment:
# heterogeneous experts use a unified calibrated P(fake).
df["selected_score"] = (
    df["selected_cal_score"]
)


df["wfs_text"] = df.apply(
    lambda r:
        (
            f"And the "
            f"{r['selected_alias']} "
            f"score is "
            f"{float(r['selected_score']):.3f}."
        ),
    axis=1,
)


OUTCSV = (
    OUT
    / "train_routed_wfs.csv"
)

df.to_csv(
    OUTCSV,
    index=False,
)


print()
print("=" * 80)
print("ROUTE DISTRIBUTION")
print("=" * 80)

for domain in sorted(
    df["dataset"].unique()
):

    print()
    print(domain)

    sub = df[
        df["dataset"]
        == domain
    ]

    dist = (
        sub[
            "selected_expert"
        ]
        .value_counts(
            normalize=True
        )
        .reindex(
            EXPERTS,
            fill_value=0,
        )
    )

    for e, v in dist.items():

        print(
            f"  {e:<10} "
            f"{v*100:6.2f}%"
        )


print()
print(
    "Saved =",
    OUTCSV
)

print()
print(
    "ROUTED TRAIN WFS READY ✅"
)

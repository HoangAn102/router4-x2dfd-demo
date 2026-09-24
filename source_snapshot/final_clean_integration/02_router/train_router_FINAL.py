from pathlib import Path
import argparse
import csv
import json
import random
import shutil
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
)

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


# ============================================================
# CONSTANTS
# ============================================================

EXPERTS = [
    "blending",
    "diffusion",
    "frequency",
    "texture",
]

DOMAINS = [
    "FFPP",
    "GANGen",
    "StyleGAN_FFHQ",
]

DOMAIN_TO_IDX = {
    d: i
    for i, d in enumerate(DOMAINS)
}


# ============================================================
# ARGS
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--train",
        default=(
            "/home/aiotlab/hoangan/"
            "outputs/router_teacher_v2/"
            "train_teacher_v2.csv"
        ),
    )

    p.add_argument(
        "--val",
        default=(
            "/home/aiotlab/hoangan/"
            "outputs/router_teacher_v2/"
            "val_teacher_v2.csv"
        ),
    )

    p.add_argument(
        "--out",
        default=(
            "/home/aiotlab/hoangan/"
            "outputs/router_training_v2/"
            "run_effb0_3domain"
        ),
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    p.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--head-lr",
        type=float,
        default=5e-4,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--freeze-epochs",
        type=int,
        default=1,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--resume",
        default="",
        help="Path to last.pt/best.pt",
    )

    p.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help=(
            "0 = same number of samples as TRAIN. "
            "Sampling remains domain-balanced."
        ),
    )

    return p.parse_args()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ============================================================
# TRANSFORMS
#
# SAME GEOMETRY/AUGMENTATION AS ROUTER V1
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


train_tf = transforms.Compose([
    transforms.Resize(256),

    transforms.RandomCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])


val_tf = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])


# ============================================================
# DATASET
# ============================================================

class RouterDataset(Dataset):

    def __init__(
        self,
        df,
        transform,
    ):

        self.df = (
            df.reset_index(
                drop=True
            )
        )

        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = str(
            row["image_path"]
        )

        with Image.open(path) as image:

            image = image.convert(
                "RGB"
            )

            image = self.transform(
                image
            )


        target = torch.tensor(
            [
                float(
                    row[
                        f"teacher_soft_{e}"
                    ]
                )
                for e in EXPERTS
            ],
            dtype=torch.float32,
        )


        utilities = torch.tensor(
            [
                float(
                    row[
                        f"utility_{e}"
                    ]
                )
                for e in EXPERTS
            ],
            dtype=torch.float32,
        )


        hard = int(
            np.argmax(
                utilities.numpy()
            )
        )


        domain_name = str(
            row["dataset"]
        )

        domain_idx = (
            DOMAIN_TO_IDX[
                domain_name
            ]
        )


        return (
            image,
            target,
            utilities,
            hard,
            domain_idx,
        )


# ============================================================
# DOMAIN-BALANCED SAMPLER
# ============================================================

def make_domain_weights(df):

    counts = (
        df["dataset"]
        .value_counts()
        .to_dict()
    )

    print()
    print(
        "TRAIN DOMAIN COUNTS:"
    )

    for d in DOMAINS:
        print(
            f"  {d:<16}",
            counts.get(d, 0),
        )


    # Each domain has equal total sampling mass.
    weights = []

    for d in df["dataset"]:

        weights.append(
            1.0 / counts[d]
        )

    return torch.tensor(
        weights,
        dtype=torch.double,
    )


# ============================================================
# LOSS
# ============================================================

def soft_target_ce(
    logits,
    target,
):

    logp = F.log_softmax(
        logits,
        dim=1,
    )

    return -(
        target
        *
        logp
    ).sum(
        dim=1
    ).mean()


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    loss_sum = 0.0
    n_total = 0

    all_routes = []
    all_hard = []
    all_util = []
    all_domain = []


    for (
        images,
        target,
        utilities,
        hard,
        domain_idx,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        target = target.to(
            device,
            non_blocking=True,
        )

        utilities = utilities.to(
            device,
            non_blocking=True,
        )


        with torch.cuda.amp.autocast(
            enabled=(
                device.type
                == "cuda"
            )
        ):

            logits = model(
                images
            )

            loss = soft_target_ce(
                logits,
                target,
            )


        bs = images.size(0)

        loss_sum += (
            float(loss.item())
            * bs
        )

        n_total += bs


        probs = torch.softmax(
            logits,
            dim=1,
        )

        routes = probs.argmax(
            dim=1
        )


        all_routes.append(
            routes.cpu()
        )

        all_hard.append(
            hard.cpu()
        )

        all_util.append(
            utilities.cpu()
        )

        all_domain.append(
            domain_idx.cpu()
        )


    routes = torch.cat(
        all_routes
    )

    hard = torch.cat(
        all_hard
    )

    utilities = torch.cat(
        all_util
    )

    domains = torch.cat(
        all_domain
    )


    selected = utilities[
        torch.arange(
            len(routes)
        ),
        routes,
    ]

    oracle = utilities.max(
        dim=1
    ).values


    top1 = (
        routes == hard
    ).float().mean().item()


    # Teacher Top-2 agreement
    # based on utility ranking.
    teacher_top2 = (
        utilities.topk(
            2,
            dim=1,
        ).indices
    )

    top2 = (
        teacher_top2
        == routes[:, None]
    ).any(
        dim=1
    ).float().mean().item()


    result = {
        "val_loss":
            loss_sum
            /
            max(1, n_total),

        "top1":
            top1,

        "top2":
            top2,

        "selected_utility":
            selected.mean().item(),

        "oracle_utility":
            oracle.mean().item(),

        "utility_gap":
            (
                oracle
                -
                selected
            ).mean().item(),
    }


    # --------------------------------------------------------
    # Per-domain VAL
    # --------------------------------------------------------

    domain_selected = []

    for d_idx, d_name in enumerate(
        DOMAINS
    ):

        mask = (
            domains
            == d_idx
        )

        n = int(
            mask.sum().item()
        )

        if n == 0:
            continue


        d_selected = (
            selected[mask]
            .mean()
            .item()
        )

        d_oracle = (
            oracle[mask]
            .mean()
            .item()
        )

        d_top1 = (
            (
                routes[mask]
                ==
                hard[mask]
            )
            .float()
            .mean()
            .item()
        )


        result[
            f"utility_{d_name}"
        ] = d_selected

        result[
            f"oracle_{d_name}"
        ] = d_oracle

        result[
            f"top1_{d_name}"
        ] = d_top1


        domain_selected.append(
            d_selected
        )


    # Equal importance for each domain.
    result[
        "macro_val_utility"
    ] = float(
        np.mean(
            domain_selected
        )
    )


    # --------------------------------------------------------
    # Overall route distribution
    # --------------------------------------------------------

    for e_idx, expert in enumerate(
        EXPERTS
    ):

        result[
            f"route_{expert}"
        ] = (
            (
                routes
                == e_idx
            )
            .float()
            .mean()
            .item()
        )


    # --------------------------------------------------------
    # Per-domain route distribution
    # --------------------------------------------------------

    for d_idx, d_name in enumerate(
        DOMAINS
    ):

        mask = (
            domains
            == d_idx
        )

        if not mask.any():
            continue

        for e_idx, expert in enumerate(
            EXPERTS
        ):

            result[
                f"route_{d_name}_{expert}"
            ] = (
                (
                    routes[mask]
                    == e_idx
                )
                .float()
                .mean()
                .item()
            )


    return result


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_metric,
    history,
    args,
    sampler_generator,
):

    torch.save(
        {
            "epoch":
                epoch,

            "model":
                model.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "scheduler":
                scheduler.state_dict(),

            "scaler":
                scaler.state_dict(),

            "best_metric":
                best_metric,

            "history":
                history,

            "args":
                vars(args),

            "experts":
                EXPERTS,

            "domains":
                DOMAINS,

            "selection_metric":
                "macro_val_utility",

            "sampler_generator_state":
                sampler_generator.get_state(),
        },
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    seed_everything(
        args.seed
    )


    out = Path(
        args.out
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )


    train_path = Path(
        args.train
    )

    val_path = Path(
        args.val
    )


    # Explicit safeguard:
    # trainer only receives TRAIN + VAL.
    print("=" * 90)
    print("ROUTER V2 — 3-DOMAIN TRAIN")
    print("=" * 90)

    print(
        "TRAIN =",
        train_path,
    )

    print(
        "VAL   =",
        val_path,
    )

    print(
        "TEST  = NOT USED"
    )


    train_df = pd.read_csv(
        train_path,
        low_memory=False,
    )

    val_df = pd.read_csv(
        val_path,
        low_memory=False,
    )


    print()
    print(
        "TRAIN N =",
        len(train_df),
    )

    print(
        train_df["dataset"]
        .value_counts()
        .sort_index()
    )


    print()
    print(
        "VAL N =",
        len(val_df),
    )

    print(
        val_df["dataset"]
        .value_counts()
        .sort_index()
    )


    overlap = (
        set(
            train_df[
                "image_path"
            ].astype(str)
        )
        &
        set(
            val_df[
                "image_path"
            ].astype(str)
        )
    )

    if overlap:

        raise RuntimeError(
            "TRAIN/VAL overlap: "
            f"{len(overlap)}"
        )


    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_ds = RouterDataset(
        train_df,
        train_tf,
    )

    val_ds = RouterDataset(
        val_df,
        val_tf,
    )


    weights = make_domain_weights(
        train_df
    )


    samples_per_epoch = (
        args.samples_per_epoch
        if args.samples_per_epoch > 0
        else len(train_ds)
    )


    sampler_generator = (
        torch.Generator()
    )

    sampler_generator.manual_seed(
        args.seed
    )


    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=sampler_generator,
    )


    train_loader = DataLoader(
        train_ds,

        batch_size=args.batch_size,

        sampler=sampler,

        num_workers=args.workers,

        pin_memory=True,

        persistent_workers=(
            args.workers > 0
        ),

        drop_last=True,
    )


    val_loader = DataLoader(
        val_ds,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.workers,

        pin_memory=True,

        persistent_workers=(
            args.workers > 0
        ),
    )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print()
    print(
        "DEVICE =",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU =",
            torch.cuda.get_device_name(
                0
            ),
        )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Loading EfficientNet-B0..."
    )


    if args.resume:

        # Full checkpoint will overwrite all weights.
        model = efficientnet_b0(
            weights=None
        )

    else:

        model = efficientnet_b0(
            weights=(
                EfficientNet_B0_Weights.DEFAULT
            )
        )


    in_features = (
        model.classifier[1]
        .in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        len(EXPERTS),
    )

    model = model.to(
        device
    )


    # --------------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    model.features.parameters(),

                "lr":
                    args.backbone_lr,
            },
            {
                "params":
                    model.classifier.parameters(),

                "lr":
                    args.head_lr,
            },
        ],

        weight_decay=args.weight_decay,
    )


    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )


    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=(
                device.type
                == "cuda"
            )
        )
    )


    start_epoch = 1
    best_metric = -1.0
    history = []


    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if args.resume:

        resume_path = Path(
            args.resume
        )

        print()
        print(
            "RESUME =",
            resume_path,
        )


        ckpt = torch.load(
            resume_path,
            map_location="cpu",
        )


        model.load_state_dict(
            ckpt["model"]
        )

        optimizer.load_state_dict(
            ckpt["optimizer"]
        )

        scheduler.load_state_dict(
            ckpt["scheduler"]
        )

        scaler.load_state_dict(
            ckpt["scaler"]
        )


        start_epoch = (
            int(
                ckpt["epoch"]
            )
            + 1
        )

        best_metric = float(
            ckpt.get(
                "best_metric",
                -1,
            )
        )

        history = ckpt.get(
            "history",
            [],
        )


        if (
            "sampler_generator_state"
            in ckpt
        ):

            sampler_generator.set_state(
                ckpt[
                    "sampler_generator_state"
                ]
            )


        print(
            "Resume from epoch",
            start_epoch,
        )

        print(
            "Best macro VAL utility =",
            best_metric,
        )


    # --------------------------------------------------------
    # Save run config
    # --------------------------------------------------------

    with (
        out /
        "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                **vars(args),

                "experts":
                    EXPERTS,

                "domains":
                    DOMAINS,

                "selection_metric":
                    "macro_val_utility",

                "test_used":
                    False,

                "sampling":
                    "inverse-domain-count "
                    "WeightedRandomSampler",

                "train_transform":
                    (
                        "Resize256-RandomCrop224-"
                        "RandomHorizontalFlip-"
                        "ImageNetNormalize"
                    ),

                "val_transform":
                    (
                        "Resize256-CenterCrop224-"
                        "ImageNetNormalize"
                    ),
            },
            f,
            indent=2,
        )


    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        t0 = time.time()


        # ----------------------------------------------------
        # Proper backbone freeze
        # ----------------------------------------------------

        freeze = (
            epoch
            <= args.freeze_epochs
        )

        for p in (
            model.features.parameters()
        ):

            p.requires_grad = (
                not freeze
            )


        model.train()

        # requires_grad=False alone does not stop
        # BatchNorm state updates.
        if freeze:

            model.features.eval()


        train_loss_sum = 0.0
        train_n = 0


        for (
            images,
            target,
            utilities,
            hard,
            domain_idx,
        ) in train_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            target = target.to(
                device,
                non_blocking=True,
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            with torch.cuda.amp.autocast(
                enabled=(
                    device.type
                    == "cuda"
                )
            ):

                logits = model(
                    images
                )

                loss = soft_target_ce(
                    logits,
                    target,
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )

            scaler.update()


            bs = images.size(0)

            train_loss_sum += (
                float(
                    loss.item()
                )
                * bs
            )

            train_n += bs


        train_loss = (
            train_loss_sum
            /
            max(
                1,
                train_n,
            )
        )


        # ----------------------------------------------------
        # VAL
        # ----------------------------------------------------

        metrics = evaluate(
            model,
            val_loader,
            device,
        )


        scheduler.step()


        epoch_row = {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            **metrics,

            "lr_backbone":
                optimizer
                .param_groups[0]["lr"],

            "lr_head":
                optimizer
                .param_groups[1]["lr"],

            "freeze_backbone":
                int(freeze),

            "seconds":
                time.time()
                - t0,
        }


        history.append(
            epoch_row
        )


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        pd.DataFrame(
            history
        ).to_csv(
            out /
            "history.csv",
            index=False,
        )


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        route_txt = " ".join([
            (
                f"{e[0].upper()}"
                f"{metrics[f'route_{e}']*100:.1f}"
            )
            for e in EXPERTS
        ])


        print()
        print(
            f"E{epoch:02d}/"
            f"{args.epochs} | "
            f"train={train_loss:.4f} | "
            f"val={metrics['val_loss']:.4f} | "
            f"top1={metrics['top1']*100:.2f}% | "
            f"top2={metrics['top2']*100:.2f}% | "
            f"selU={metrics['selected_utility']:.4f} | "
            f"oracleU={metrics['oracle_utility']:.4f} | "
            f"gap={metrics['utility_gap']:.4f} | "
            f"macroU={metrics['macro_val_utility']:.4f} | "
            f"routes[{route_txt}] | "
            f"{epoch_row['seconds']/60:.1f} min"
        )


        print(
            "  Domain utility:",
            " | ".join([
                (
                    f"{d}="
                    f"{metrics[f'utility_{d}']:.4f}"
                )
                for d in DOMAINS
            ])
        )


        print(
            "  Domain top1:",
            " | ".join([
                (
                    f"{d}="
                    f"{metrics[f'top1_{d}']*100:.2f}%"
                )
                for d in DOMAINS
            ])
        )


        print(
            "  FFPP routes:",
            " ".join([
                (
                    f"{e[0].upper()}="
                    f"{metrics[f'route_FFPP_{e}']*100:.1f}%"
                )
                for e in EXPERTS
            ])
        )


        print(
            "  GANGen routes:",
            " ".join([
                (
                    f"{e[0].upper()}="
                    f"{metrics[f'route_GANGen_{e}']*100:.1f}%"
                )
                for e in EXPERTS
            ])
        )


        print(
            "  StyleGAN routes:",
            " ".join([
                (
                    f"{e[0].upper()}="
                    f"{metrics[f'route_StyleGAN_FFHQ_{e}']*100:.1f}%"
                )
                for e in EXPERTS
            ])
        )


        # ----------------------------------------------------
        # LAST
        # ----------------------------------------------------

        save_checkpoint(
            out / "last.pt",

            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_metric,
            history,
            args,
            sampler_generator,
        )


        # ----------------------------------------------------
        # BEST — VAL ONLY
        # ----------------------------------------------------

        current = (
            metrics[
                "macro_val_utility"
            ]
        )


        if current > best_metric:

            best_metric = current


            save_checkpoint(
                out / "best.pt",

                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_metric,
                history,
                args,
                sampler_generator,
            )


            print(
                f"  BEST ✅ "
                f"epoch={epoch} "
                f"macroU={best_metric:.6f}"
            )


    print()
    print("=" * 90)
    print("TRAINING FINISHED ✅")
    print("=" * 90)

    print(
        "BEST MACRO VAL UTILITY =",
        best_metric,
    )

    print(
        "BEST =",
        out / "best.pt",
    )

    print(
        "LAST =",
        out / "last.pt",
    )

    print(
        "TEST WAS NOT USED ✅"
    )


if __name__ == "__main__":
    main()

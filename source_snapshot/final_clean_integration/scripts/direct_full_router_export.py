from pathlib import Path
import argparse
import json
import time
import re
import sys

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXPERTS = [
    "blending",
    "diffusion",
    "frequency",
    "texture",
]


def extract_state(path):
    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        obj = torch.load(
            path,
            map_location="cpu",
        )

    if isinstance(obj, dict):
        for k in [
            "model_state_dict",
            "state_dict",
            "model",
            "net",
        ]:
            v = obj.get(k)
            if isinstance(v, dict) and v:
                return v, obj

        if obj and all(
            torch.is_tensor(v)
            for v in obj.values()
        ):
            return obj, obj

    raise RuntimeError(
        f"Cannot locate state_dict in {path}"
    )


def strip_prefix(sd):
    prefixes = [
        "module.",
        "model.",
        "net.",
    ]

    out = {}

    for k, v in sd.items():
        nk = k

        changed = True
        while changed:
            changed = False

            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True

        out[nk] = v

    return out


def build_model(ckpt):
    sd, raw = extract_state(ckpt)
    sd = strip_prefix(sd)

    # Standard torchvision EfficientNet-B0 used by Router v2.
    m = efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(
        m.classifier[1].in_features,
        4,
    )

    # Common wrapper case: backbone.features / backbone.classifier
    if any(
        k.startswith("backbone.")
        for k in sd
    ):
        alt = {}

        for k, v in sd.items():
            if k.startswith("backbone."):
                alt[k[len("backbone."):]] = v
            elif k in {
                "classifier.weight",
                "classifier.bias",
            }:
                alt[
                    "classifier.1." +
                    k.split(".", 1)[1]
                ] = v
            else:
                alt[k] = v

        sd = alt

    missing, unexpected = m.load_state_dict(
        sd,
        strict=False,
    )

    # Fail if reconstruction is obviously wrong.
    important_missing = [
        x for x in missing
        if not x.startswith("classifier.0")
    ]

    if len(important_missing) > 8 or len(unexpected) > 8:
        print("STATE KEYS SAMPLE:")
        print(list(sd.keys())[:50])

        print("MISSING:")
        print(missing[:50])

        print("UNEXPECTED:")
        print(unexpected[:50])

        raise RuntimeError(
            "Router checkpoint does not match "
            "torchvision EfficientNet-B0 reconstruction."
        )

    return m


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def candidate_transforms():
    norm = transforms.Normalize(
        MEAN,
        STD,
    )

    return {
        "square224_bilinear":
            transforms.Compose([
                transforms.Resize(
                    (224, 224),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToTensor(),
                norm,
            ]),

        "square224_bicubic":
            transforms.Compose([
                transforms.Resize(
                    (224, 224),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                norm,
            ]),

        "resize256_crop224_bilinear":
            transforms.Compose([
                transforms.Resize(
                    256,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                norm,
            ]),

        "resize256_crop224_bicubic":
            transforms.Compose([
                transforms.Resize(
                    256,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                norm,
            ]),
    }


class ImgDS(Dataset):
    def __init__(
        self,
        paths,
        tfm,
    ):
        self.paths = list(paths)
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]

        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.tfm(im)

        return i, x


@torch.inference_mode()
def infer(
    model,
    paths,
    tfm,
    batch_size,
    workers,
    device,
    print_every=100,
):
    ds = ImgDS(
        paths,
        tfm,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    probs = np.zeros(
        (len(ds), 4),
        dtype=np.float32,
    )

    t0 = time.time()

    for bi, (idx, x) in enumerate(dl, 1):
        x = x.to(
            device,
            non_blocking=True,
        )

        logits = model(x)
        p = torch.softmax(
            logits,
            dim=1,
        )

        probs[
            idx.numpy()
        ] = p.cpu().numpy()

        if (
            bi % print_every == 0
            or bi == len(dl)
        ):
            done = min(
                bi * batch_size,
                len(ds),
            )

            elapsed = time.time() - t0
            ips = done / max(
                elapsed,
                1e-9,
            )

            eta = (
                len(ds) - done
            ) / max(
                ips,
                1e-9,
            )

            print(
                f"[ROUTER] "
                f"{done}/{len(ds)} "
                f"{100*done/len(ds):.2f}% | "
                f"{ips:.1f} img/s | "
                f"ETA {eta/60:.1f} min",
                flush=True,
            )

    return probs


def cached_targets(df):
    if "router_expert_idx" in df.columns:
        x = pd.to_numeric(
            df["router_expert_idx"],
            errors="coerce",
        )

        if x.notna().mean() > 0.95:
            return x.to_numpy()

    for c in [
        "selected_expert",
        "router_selected_expert",
    ]:
        if c in df.columns:
            mp = {
                e: i
                for i, e
                in enumerate(EXPERTS)
            }

            x = (
                df[c]
                .astype(str)
                .str.lower()
                .map(mp)
            )

            if x.notna().mean() > 0.95:
                return x.to_numpy()

    return None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train",
        required=True,
    )
    ap.add_argument(
        "--old-router",
        required=True,
    )
    ap.add_argument(
        "--new-router",
        required=True,
    )
    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=12,
    )

    args = ap.parse_args()

    print("="*80)
    print("DIRECT FULL ROUTER EXPORT")
    print("="*80)

    df = pd.read_csv(
        args.train,
        low_memory=False,
    )

    if "image_path" not in df.columns:
        raise RuntimeError(
            "TRAIN has no image_path"
        )

    paths = (
        df["image_path"]
        .astype(str)
        .tolist()
    )

    print(
        "TRAIN ROWS:",
        len(df),
    )

    if len(df) < 200_000:
        raise RuntimeError(
            "Unexpectedly small clean TRAIN."
        )

    # Verify files before GPU run.
    missing = [
        p for p in paths
        if not Path(p).is_file()
    ]

    if missing:
        print(
            "MISSING SAMPLE:",
            missing[:20],
        )

        raise RuntimeError(
            f"{len(missing)} images missing"
        )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA unavailable"
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


    # ========================================================
    # PREPROCESS PARITY GATE
    # ========================================================

    print()
    print("="*80)
    print("PREPROCESS PARITY GATE")
    print("="*80)

    parity_path = Path(
        r"/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/03_wfs/LEGACY_ROUTER_PARITY.csv"
    )

    if not parity_path.is_file():
        raise RuntimeError(
            "Legacy parity manifest missing: "
            + str(parity_path)
        )

    parity_df = pd.read_csv(
        parity_path,
        low_memory=False,
    )

    if (
        "image_path" not in parity_df.columns
        or "router_expert_idx" not in parity_df.columns
    ):
        raise RuntimeError(
            "Invalid parity manifest columns"
        )

    parity_df = parity_df[
        parity_df["image_path"].map(
            lambda z: Path(str(z)).is_file()
        )
    ].reset_index(drop=True)

    if len(parity_df) < 256:
        raise RuntimeError(
            "Too few parity images"
        )

    sample_paths = (
        parity_df["image_path"]
        .astype(str)
        .tolist()
    )

    sample_gt = (
        pd.to_numeric(
            parity_df["router_expert_idx"],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    print(
        "PARITY SAMPLES:",
        len(sample_paths)
    )

    old = build_model(
        args.old_router
    ).to(device)

    old.eval()

    candidates = candidate_transforms()

    parity = {}

    for name, tfm in candidates.items():

        print()
        print(
            "TEST PREPROCESS:",
            name
        )

        pp = infer(
            old,
            sample_paths,
            tfm,
            batch_size=min(
                args.batch_size,
                256,
            ),
            workers=min(
                args.workers,
                8,
            ),
            device=device,
            print_every=999999,
        )

        pred = pp.argmax(1)

        score = float(
            (pred == sample_gt).mean()
        )

        parity[name] = score

        print(
            f"PARITY {name}: "
            f"{100*score:.3f}%"
        )

    best_name = max(
        parity,
        key=parity.get,
    )

    best_parity = parity[
        best_name
    ]

    print()
    print(
        "BEST PREPROCESS:",
        best_name,
    )

    print(
        "BEST PARITY:",
        best_parity,
    )

    # Earlier verified CPU-vs-legacy Router parity was ~99.9%.
    # 97% is therefore a conservative safety floor.
    if best_parity < 0.97:

        raise RuntimeError(
            "Could not reproduce old Router closely enough. "
            f"Best parity={best_parity:.4f}. "
            "Refusing final routing."
        )

    del old
    torch.cuda.empty_cache()

    tfm = candidates[
        best_name
    ]


    # ========================================================
    # NEW FINAL ROUTER
    # ========================================================

    print()
    print("="*80)
    print("RUN NEW CLEAN ROUTER ON FULL TRAIN")
    print("="*80)

    model = build_model(
        args.new_router
    ).to(device)

    model.eval()

    probs = infer(
        model,
        paths,
        tfm,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        print_every=50,
    )

    idx = probs.argmax(1)

    sorted_p = np.sort(
        probs,
        axis=1,
    )

    confidence = sorted_p[:, -1]
    margin = (
        sorted_p[:, -1]
        - sorted_p[:, -2]
    )

    df["router_expert_idx"] = idx
    df["selected_expert"] = [
        EXPERTS[i]
        for i in idx
    ]

    df["router_confidence"] = confidence
    df["router_margin"] = margin

    for i, e in enumerate(EXPERTS):
        df[f"router_p{i}"] = probs[:, i]
        df[f"router_prob_{e}"] = probs[:, i]

    out = Path(args.out)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out,
        index=False,
    )

    report = {
        "rows": len(df),
        "old_router": args.old_router,
        "new_router": args.new_router,
        "preprocess": best_name,
        "old_router_parity": best_parity,
        "all_preprocess_parity": parity,
        "distribution":
            df["selected_expert"]
            .value_counts()
            .to_dict(),
        "mean_confidence":
            float(
                np.mean(confidence)
            ),
        "mean_margin":
            float(
                np.mean(margin)
            ),
        "output": str(out),
    }

    report_path = (
        out.parent
        / "DIRECT_ROUTER_EXPORT_AUDIT.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("="*80)
    print("DIRECT ROUTER EXPORT READY")
    print("="*80)
    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

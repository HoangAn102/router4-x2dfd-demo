import argparse
import csv
from pathlib import Path

import torch

from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import (
    Dataset,
    DataLoader
)

from networks.resnet import resnet50


parser = argparse.ArgumentParser()

parser.add_argument(
    "--out",
    required=True
)

parser.add_argument(
    "--name",
    required=True
)

args = parser.parse_args()

OUT = Path(args.out)
CSV = OUT / "frequency_scores.csv"

with (
    OUT / "manifest.csv"
).open() as f:

    rows = list(
        csv.DictReader(f)
    )

paths = [
    r["image_path"]
    for r in rows
]


saved = {}

if CSV.exists():

    with CSV.open() as f:

        for r in csv.DictReader(f):

            try:
                saved[
                    r["image_path"]
                ] = float(
                    r["score"]
                )
            except:
                pass


pending = [
    p for p in paths
    if p not in saved
]


print()
print("=" * 70)
print(
    f"FREQUENCY -> {args.name}"
)
print("=" * 70)

print("TOTAL   :", len(paths))
print("DONE    :", len(saved))
print("PENDING :", len(pending))


if pending:

    model = resnet50(
        num_classes=1
    )

    state = torch.load(
        "checkpoints/model_epoch_last.pth",
        map_location="cpu"
    )

    if (
        isinstance(state, dict)
        and "state_dict" in state
    ):
        state = state[
            "state_dict"
        ]

    state = {
        (
            k[7:]
            if k.startswith(
                "module."
            )
            else k
        ): v
        for k, v
        in state.items()
    }

    model.load_state_dict(
        state,
        strict=True
    )

    model = (
        model.cuda().eval()
    )


    tfm = transforms.Compose([
        transforms.CenterCrop(256),

        transforms.ToTensor(),

        transforms.Normalize(
            [
                0.485,
                0.456,
                0.406
            ],
            [
                0.229,
                0.224,
                0.225
            ]
        )
    ])


    class DS(Dataset):

        def __len__(self):
            return len(pending)

        def __getitem__(
            self,
            index
        ):

            path = pending[
                index
            ]

            image = Image.open(
                path
            ).convert("RGB")

            return (
                tfm(image),
                path
            )


    loader = DataLoader(
        DS(),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


    header = not CSV.exists()

    with CSV.open(
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "score"
            ]
        )

        if header:
            writer.writeheader()

        with torch.no_grad():

            for images, batch_paths \
            in tqdm(
                loader,
                desc=f"FREQ {args.name}"
            ):

                images = images.cuda(
                    non_blocking=True
                )

                scores = torch.sigmoid(
                    model(images)
                ).flatten()

                scores = (
                    scores
                    .cpu()
                    .tolist()
                )

                for path, score \
                in zip(
                    batch_paths,
                    scores
                ):

                    writer.writerow({
                        "image_path":
                            path,

                        "score":
                            float(score)
                    })

                f.flush()


unique = {}

if CSV.exists():

    with CSV.open() as f:

        for r in csv.DictReader(f):

            if r.get(
                "score",
                ""
            ) != "":

                unique[
                    r["image_path"]
                ] = r["score"]


print()
print(
    "FINAL:",
    len(unique),
    "/",
    len(paths)
)
print("CSV:", CSV)

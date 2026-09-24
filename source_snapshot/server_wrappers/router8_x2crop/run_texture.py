import argparse
import csv
import gc
import os
import sys

from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch.utils.data import (
    Dataset,
    DataLoader
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--out",
    required=True
)

parser.add_argument(
    "--work",
    required=True
)

args = parser.parse_args()

OUT = Path(args.out)
WORK = Path(args.work)

CSV = (
    OUT
    / "texture_scores.csv"
)


with (
    OUT
    / "manifest.csv"
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
print("TEXTURE -> FFPP_X2DFD32")
print("=" * 70)

print("TOTAL   :", len(paths))
print("DONE    :", len(saved))
print("PENDING :", len(pending))


if pending:

    demo = (
        WORK
        / "demo.py"
    )

    captured = []

    original = (
        nn.Module
        .load_state_dict
    )


    def hook(
        self,
        *args,
        **kwargs
    ):

        result = original(
            self,
            *args,
            **kwargs
        )

        captured.append(
            self
        )

        return result


    nn.Module.load_state_dict = (
        hook
    )


    cwd = os.getcwd()
    argv = sys.argv[:]

    namespace = {
        "_name_":
            "_main_",

        "_file_":
            str(demo),

        "_package_":
            None,
    }


    try:

        os.chdir(
            WORK
        )

        sys.path.insert(
            0,
            str(WORK)
        )

        sys.argv = [
            str(demo)
        ]

        try:

            exec(
                compile(
                    demo.read_text(),
                    str(demo),
                    "exec"
                ),
                namespace,
                namespace
            )

        except SystemExit:
            pass

        except Exception as e:

            print(
                "Demo bootstrap:",
                repr(e)
            )

    finally:

        nn.Module.load_state_dict = (
            original
        )

        os.chdir(cwd)
        sys.argv = argv


    candidates = []

    for model in captured:

        if isinstance(
            model,
            nn.Module
        ):

            candidates.append(
                model
            )


    for obj in namespace.values():

        if isinstance(
            obj,
            nn.Module
        ):

            candidates.append(
                obj
            )


    unique = []
    seen = set()

    for model in candidates:

        if id(model) not in seen:

            seen.add(
                id(model)
            )

            unique.append(
                model
            )


    if not unique:
        raise RuntimeError(
            "Cannot load Gram-Net texture model"
        )


    model = max(
        unique,
        key=lambda m:
            sum(
                p.numel()
                for p in m.parameters()
            )
    )

    model = (
        model
        .cuda()
        .eval()
    )

    gc.collect()
    torch.cuda.empty_cache()


    class DS(Dataset):

        def __len__(self):
            return len(
                pending
            )

        def __getitem__(
            self,
            index
        ):

            path = pending[
                index
            ]

            image = cv2.imread(
                path,
                cv2.IMREAD_COLOR
            )

            if image is None:
                raise RuntimeError(
                    path
                )

            image = cv2.resize(
                image,
                (512, 512),
                interpolation=
                    cv2.INTER_LINEAR
            )

            image = image.astype(
                np.float32
            )

            image = np.transpose(
                image,
                (2, 0, 1)
            )

            return (
                torch.from_numpy(
                    image
                ),
                path
            )


    loader = DataLoader(
        DS(),
        batch_size=16,
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
                desc="TEXTURE FFPP"
            ):

                images = images.cuda(
                    non_blocking=True
                )

                output = model(
                    images
                )

                if isinstance(
                    output,
                    (tuple, list)
                ):

                    output = output[0]


                # Gram-Net original:
                # class 0 = fake.
                if (
                    output.ndim == 2
                    and output.shape[1] >= 2
                ):

                    score = F.softmax(
                        output,
                        dim=1
                    )[:, 0]

                else:

                    score = torch.sigmoid(
                        output.reshape(-1)
                    )


                score = (
                    score
                    .detach()
                    .cpu()
                    .tolist()
                )


                for path, value \
                in zip(
                    batch_paths,
                    score
                ):

                    writer.writerow({
                        "image_path":
                            path,

                        "score":
                            float(value)
                    })

                f.flush()


unique_scores = {}

if CSV.exists():

    with CSV.open() as f:

        for r in csv.DictReader(f):

            if r.get(
                "score",
                ""
            ) != "":

                unique_scores[
                    r["image_path"]
                ] = r["score"]


print()
print(
    "TEXTURE FINAL:",
    len(unique_scores),
    "/",
    len(paths)
)

print("CSV:", CSV)

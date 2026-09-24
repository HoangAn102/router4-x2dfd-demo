import argparse
import csv
import gc
import sys
import types
import traceback

from pathlib import Path

import torch
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument(
    "--out",
    required=True
)

parser.add_argument(
    "--expert",
    choices=[
        "blending",
        "diffusion"
    ],
    required=True
)

args = parser.parse_args()

OUT = Path(args.out)

# ------------------------------------------------------------
# Avoid unrelated LLaVA import in utils/__init__.py
# without editing repository files.
# ------------------------------------------------------------

utils_dir = (
    Path.cwd()
    / "utils"
)

pkg = types.ModuleType(
    "utils"
)

pkg.__path__ = [
    str(utils_dir)
]

sys.modules[
    "utils"
] = pkg

from utils.model_scoring import (
    get_provider
)


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

CSV = (
    OUT
    / f"{args.expert}_scores.csv"
)


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
    args.expert.upper(),
    "-> FFPP_X2DFD32"
)
print("=" * 70)

print("TOTAL   :", len(paths))
print("DONE    :", len(saved))
print("PENDING :", len(pending))


if pending:

    gc.collect()
    torch.cuda.empty_cache()


    if args.expert == "blending":

        provider = get_provider(
            "blending",

            model_name=
                "swinv2_base_window16_256",

            weights_path=
                "weights/blending_models/"
                "best_gf.pth",

            img_size=256,

            num_class=2,

            device="cuda:0",

            batch_size=32,

            num_workers=4,

            pin_memory=True,
        )

        batch_size = 32
        chunk_size = 512

    else:

        provider = get_provider(
            "diffusion_detector",

            weights_dir="weights",

            model="ours-sync",

            device="cuda:0",

            batch_size=8,

            num_workers=4,

            pin_memory=True,

            compile_model=False,
        )

        batch_size = 8
        chunk_size = 128


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


        for start in tqdm(
            range(
                0,
                len(pending),
                chunk_size
            ),
            desc=args.expert.upper()
        ):

            chunk = pending[
                start:
                start + chunk_size
            ]

            try:

                result = (
                    provider.compute_scores(
                        chunk,

                        batch_size=
                            batch_size,

                        num_workers=4,

                        pin_memory=True,
                    )
                )

            except Exception:

                print()
                print(
                    "ERROR AT INDEX:",
                    start
                )

                traceback.print_exc()

                print(
                    "CSV progress preserved."
                )

                break


            for path in chunk:

                obj = result.get(
                    path
                )

                if (
                    obj is None
                    or obj.score is None
                ):
                    continue

                writer.writerow({
                    "image_path":
                        path,

                    "score":
                        float(obj.score)
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
    args.expert.upper(),
    "FINAL:",
    len(unique),
    "/",
    len(paths)
)

print("CSV:", CSV)

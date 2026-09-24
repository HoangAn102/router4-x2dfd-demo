
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import csv
import hashlib
import json
import sys

from PIL import Image


train_csv = Path(sys.argv[1])
test_csv = Path(sys.argv[2])
out = Path(sys.argv[3])


def train_paths(path):

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
        newline="",
    ) as f:

        r = csv.DictReader(f)
        fields = r.fieldnames or []

        col = next(
            (
                c
                for c in [
                    "image_path",
                    "image",
                    "path",
                ]
                if c in fields
            ),
            None,
        )

        if col is None:
            raise RuntimeError(
                "No train image-path column"
            )

        return sorted(
            set(
                str(row[col]).strip()
                for row in r
                if str(row[col]).strip()
            )
        )


def test_paths(path):

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
        newline="",
    ) as f:

        r = csv.DictReader(f)

        rows = []

        for row in r:

            p = str(
                row["image_path"]
            ).strip()

            if p:
                rows.append(
                    (
                        p,
                        str(
                            row.get(
                                "dataset",
                                "",
                            )
                        ),
                    )
                )

        unique = {}

        for p, ds in rows:
            unique[p] = ds

        return sorted(
            unique.items()
        )


def h_rgb(path):

    try:
        with Image.open(path) as im:

            im = im.convert("RGB")

            h = hashlib.sha256()

            h.update(
                f"{im.width}x{im.height}|RGB|".encode()
            )

            h.update(
                im.tobytes()
            )

        return path, h.hexdigest(), None

    except Exception as e:
        return path, None, repr(e)


tr_paths = train_paths(train_csv)
te_rows = test_paths(test_csv)

te_paths = [
    p
    for p, _
    in te_rows
]

te_dataset = dict(te_rows)


def compute(paths, tag):

    hash_to_paths = defaultdict(list)
    errors = []

    # deliberately conservative so GPU inference I/O is not starved
    with ThreadPoolExecutor(
        max_workers=2
    ) as ex:

        for i, result in enumerate(
            ex.map(
                h_rgb,
                paths,
            ),
            1,
        ):

            p, h, err = result

            if err is None:
                hash_to_paths[h].append(p)

            else:
                errors.append(
                    {
                        "path": p,
                        "error": err,
                    }
                )

            if i % 5000 == 0:
                print(
                    tag,
                    i,
                    "/",
                    len(paths),
                    flush=True,
                )

    return hash_to_paths, errors


print(
    "TRAIN FILES =",
    len(tr_paths),
    flush=True,
)

print(
    "TEST FILES =",
    len(te_paths),
    flush=True,
)


train_hash, train_err = compute(
    tr_paths,
    "TRAIN",
)

test_hash, test_err = compute(
    te_paths,
    "TEST",
)


common = sorted(
    set(train_hash)
    & set(test_hash)
)


cross_dataset_duplicates = []

for h, pp in test_hash.items():

    datasets = {
        te_dataset.get(p, "")
        for p in pp
    }

    if len(datasets) > 1:
        cross_dataset_duplicates.append(
            {
                "hash": h,
                "datasets": sorted(datasets),
                "paths": pp[:20],
            }
        )


result = {
    "train_files": len(tr_paths),
    "test_files": len(te_paths),

    "train_decode_errors": len(train_err),
    "test_decode_errors": len(test_err),

    "decoded_rgb_train_test_overlap":
        len(common),

    "overlap_examples": [
        {
            "hash": h,
            "train": train_hash[h][:5],
            "test": test_hash[h][:5],
        }
        for h in common[:100]
    ],

    "cross_dataset_duplicate_hashes":
        len(cross_dataset_duplicates),

    "cross_dataset_duplicate_examples":
        cross_dataset_duplicates[:100],

    "train_error_examples":
        train_err[:50],

    "test_error_examples":
        test_err[:50],
}


out.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    json.dumps(
        result,
        indent=2,
    ),
    flush=True,
)


if train_err or test_err:
    raise SystemExit(20)


if common:
    raise SystemExit(30)


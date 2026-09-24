
from pathlib import Path
from copy import deepcopy
from collections import Counter
import csv
import hashlib
import json
import re
import sys

TEMPLATE = Path(sys.argv[1]).resolve()
MANIFEST = Path(sys.argv[2]).resolve()
SMOKE    = Path(sys.argv[3]).resolve()
FULL     = Path(sys.argv[4]).resolve()
DRIVER   = Path(sys.argv[5]).resolve()

EXPECTED_DATASETS = {
    "Celeb-DF-v2",
    "DFDCP",
}

###############################################################################
# Helpers
###############################################################################

def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def norm_path(x):

    p = Path(str(x)).expanduser()

    try:
        return str(p.resolve())
    except Exception:
        return str(p)


###############################################################################
# 1. READ + STRICTLY VERIFY PROVEN X2DFD TEMPLATE
###############################################################################

template = json.loads(
    TEMPLATE.read_text(
        encoding="utf-8"
    )
)

if not isinstance(template, dict):
    raise RuntimeError(
        "PROVEN_X2DFD_TEMPLATE root is not dict"
    )

if "images" not in template:
    raise RuntimeError(
        "PROVEN template missing root['images']"
    )

images_template = template["images"]

if not isinstance(images_template, list):
    raise RuntimeError(
        "PROVEN template root['images'] is not list"
    )

if len(images_template) != 1:
    raise RuntimeError(
        f"Expected exactly one template item, got "
        f"{len(images_template)}"
    )

item_template = images_template[0]

if not isinstance(item_template, dict):
    raise RuntimeError(
        "Template image item is not dict"
    )

if "image_path" not in item_template:
    raise RuntimeError(
        "Template item missing 'image_path'"
    )

print("✅ X2DFD schema:")
print("   IMAGE LIST PATH = ('images',)")
print("   IMAGE KEY       = image_path")
print("   TEMPLATE ITEMS  = 1")

###############################################################################
# 2. READ MANIFEST
###############################################################################

with MANIFEST.open(
    newline="",
    encoding="utf-8",
    errors="strict",
) as f:
    rows = list(csv.DictReader(f))

if len(rows) != 8:
    raise RuntimeError(
        f"Expected 8 smoke rows, got {len(rows)}"
    )

cols = set(rows[0].keys())

if "image_path" not in cols:
    raise RuntimeError(
        f"manifest missing image_path; cols={sorted(cols)}"
    )

if "dataset" not in cols:
    raise RuntimeError(
        f"manifest missing dataset; cols={sorted(cols)}"
    )

###############################################################################
# 3. VERIFY EVERY ROW
###############################################################################

seen_paths = set()
datasets = Counter()

for i, row in enumerate(rows):

    ds = str(row["dataset"]).strip()
    path = norm_path(row["image_path"])

    if ds not in EXPECTED_DATASETS:
        raise RuntimeError(
            f"row {i}: unexpected dataset={ds!r}"
        )

    # Absolute hard anti-DFDC guard.
    if ds == "DFDC":
        raise RuntimeError(
            "DFDC leaked into 2-cross smoke manifest"
        )

    if "/DFDC/" in path.replace("\\", "/"):
        raise RuntimeError(
            f"DFDC path leaked into manifest: {path}"
        )

    if not Path(path).is_file():
        raise RuntimeError(
            f"row {i}: missing image: {path}"
        )

    if path in seen_paths:
        raise RuntimeError(
            f"duplicate smoke image: {path}"
        )

    seen_paths.add(path)
    datasets[ds] += 1

if set(datasets) != EXPECTED_DATASETS:
    raise RuntimeError(
        f"Expected both Celeb + DFDCP, got {dict(datasets)}"
    )

print()
print("MANIFEST DISTRIBUTION =", dict(datasets))
print("UNIQUE IMAGES         =", len(seen_paths))

###############################################################################
# 4. CREATE JSON FROM PROVEN TEMPLATE — NEVER INVENT SCHEMA
###############################################################################

def build_json(selected_rows):

    obj = deepcopy(template)

    out_items = []

    for row in selected_rows:

        item = deepcopy(item_template)

        item["image_path"] = norm_path(
            row["image_path"]
        )

        out_items.append(item)

    obj["images"] = out_items

    return obj


def write_and_verify(path, selected_rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    obj = build_json(
        selected_rows
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Independent reread.
    x = json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(x, dict):
        raise RuntimeError(
            f"{path}: written JSON root not dict"
        )

    if not isinstance(
        x.get("images"),
        list,
    ):
        raise RuntimeError(
            f"{path}: images not list"
        )

    expected = [
        norm_path(r["image_path"])
        for r in selected_rows
    ]

    actual = [
        norm_path(r["image_path"])
        for r in x["images"]
    ]

    if actual != expected:
        raise RuntimeError(
            f"{path}: written image order/content mismatch"
        )

    for p in actual:

        if not Path(p).is_file():
            raise RuntimeError(
                f"{path}: nonexistent output image {p}"
            )

    tmp.replace(path)

    print(
        f"✅ {path.name:32s} "
        f"rows={len(actual):2d} "
        f"sha256={sha256(path)[:16]}"
    )

###############################################################################
# 5. FULL 8-IMAGE INPUT
###############################################################################

write_and_verify(
    FULL,
    rows,
)

###############################################################################
# 6. MATERIALIZE EVERY *_smoke.json ACTUALLY REFERENCED BY DRIVER
#
# No assumptions about exact filename spelling.
###############################################################################

driver_text = DRIVER.read_text(
    encoding="utf-8",
    errors="strict",
)

names = sorted(
    set(
        re.findall(
            r'\$SMOKE/([A-Za-z0-9_.-]+_smoke\.json)',
            driver_text,
        )
    )
)

print()
print("DRIVER-REFERENCED SMOKE JSONS =", names)

for name in names:

    low = name.lower()

    if "dfdcp" in low:

        subset = [
            r for r in rows
            if r["dataset"] == "DFDCP"
        ]

    elif "celeb" in low:

        subset = [
            r for r in rows
            if r["dataset"] == "Celeb-DF-v2"
        ]

    elif "dfdc" in low:

        # DFDC intentionally excluded.
        subset = []

    else:

        raise RuntimeError(
            f"Unknown driver smoke JSON name: {name}. "
            "Refusing to guess dataset ownership."
        )

    write_and_verify(
        SMOKE / name,
        subset,
    )

###############################################################################
# 7. FINAL CROSS-CHECK
###############################################################################

full = json.loads(
    FULL.read_text()
)

full_paths = [
    norm_path(x["image_path"])
    for x in full["images"]
]

if len(full_paths) != 8:
    raise RuntimeError(
        "full_inputs.json != 8 images"
    )

if set(full_paths) != seen_paths:
    raise RuntimeError(
        "full_inputs.json image set != manifest"
    )

print()
print("==========================================================")
print("✅✅ INPUT MATERIALIZATION PASS ✅✅")
print("==========================================================")
print("full_inputs rows =", len(full_paths))
print("datasets         =", dict(datasets))
print("DFDC             = 0")
print("schema           = proven template")
print("==========================================================")


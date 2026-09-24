from pathlib import Path
import csv
import json

import yaml

BASE = Path("/home/aiotlab/hoangan")
X2 = BASE / "projects/X2DFD"

OUT = (
    BASE /
    "outputs/x2dfd_router_integration/full_DFDCP_DFDC"
)
OUT.mkdir(parents=True, exist_ok=True)

TEST_ROOT = X2 / "datasets/raw/data/test"

WANTED = [
    "DFDCP_real.json",
    "DFDCP_fake.json",
    "DFDC_real.json",
    "DFDC_fake.json",
]


def find_full_json(name):
    candidates = []

    for p in TEST_ROOT.rglob(name):
        if "Tiny_Test" in p.parts:
            continue

        candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy FULL {name} ngoài Tiny_Test"
        )

    # Ưu tiên path nông nhất.
    candidates.sort(
        key=lambda p: (
            len(p.parts),
            str(p),
        )
    )

    if len(candidates) > 1:
        print(f"WARNING: nhiều candidate cho {name}:")
        for x in candidates:
            print(" ", x)
        print("Chọn:", candidates[0])

    return candidates[0]


def resolve_image(json_path, description, image_path):
    p = Path(str(image_path))

    if p.is_absolute():
        return p.resolve()

    if description:
        root = Path(str(description))

        if not root.is_absolute():
            root = X2 / root

        return (root / p).resolve()

    # fallback relative to repo
    return (X2 / p).resolve()


full_inputs = []
rows = []
summary = []

for name in WANTED:
    src = find_full_json(name)

    data = json.loads(
        src.read_text(encoding="utf-8")
    )

    images = data.get("images", [])

    if not images:
        raise RuntimeError(
            f"{src} không có images"
        )

    desc = data.get("Description")

    n_missing = 0

    for item in images:
        p = resolve_image(
            src,
            desc,
            item["image_path"],
        )

        if not p.exists():
            n_missing += 1

        rows.append({
            "image_path": str(p),
            "source_json": name,
            "source_json_path": str(src),
        })

    full_inputs.append(str(src))

    summary.append({
        "json": name,
        "path": str(src),
        "N": len(images),
        "missing_files": n_missing,
    })


# Deduplicate exact image paths.
seen = set()
unique = []

for r in rows:
    p = r["image_path"]

    if p in seen:
        continue

    seen.add(p)
    unique.append(r)


with (OUT / "manifest.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "image_path",
            "source_json",
            "source_json_path",
        ],
    )

    w.writeheader()
    w.writerows(unique)


(OUT / "full_inputs.json").write_text(
    json.dumps(
        full_inputs,
        indent=2,
    ),
    encoding="utf-8",
)

(OUT / "dataset_summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 90)
print("FULL EXTERNAL INPUTS")
print("=" * 90)

for r in summary:
    print(
        f"{r['json']:<20} "
        f"N={r['N']:>8} | "
        f"missing={r['missing_files']:>5}"
    )
    print(" ", r["path"])

print()
print("UNIQUE IMAGES =", len(unique))

missing_total = sum(
    x["missing_files"]
    for x in summary
)

print("MISSING FILES =", missing_total)

if missing_total:
    raise RuntimeError(
        "Có image path bị thiếu; không chạy full."
    )

print()
print("FULL MANIFEST READY ✅")

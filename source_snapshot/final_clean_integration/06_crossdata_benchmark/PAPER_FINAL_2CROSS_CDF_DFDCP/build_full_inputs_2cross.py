from pathlib import Path
from copy import deepcopy
import csv
import json
import sys

manifest = Path(sys.argv[1]).resolve()
smoke = Path(sys.argv[2]).resolve()
out = Path(sys.argv[3]).resolve()

PATH_KEYS = (
    "image_path",
    "image",
    "path",
    "frame_path",
)

# ---------------------------------------------------------------------
# Canonical manifest image set
# ---------------------------------------------------------------------

with manifest.open(
    newline="",
    encoding="utf-8",
    errors="strict",
) as f:
    rows = list(csv.DictReader(f))

if len(rows) != 8:
    raise RuntimeError(
        f"Expected 8 smoke manifest rows, got {len(rows)}"
    )

manifest_col = next(
    (
        k for k in PATH_KEYS
        if k in rows[0]
    ),
    None,
)

if manifest_col is None:
    raise RuntimeError(
        f"No image-path column in manifest: {list(rows[0])}"
    )

def norm(x):
    p = Path(str(x)).expanduser()

    try:
        return str(p.resolve())
    except Exception:
        return str(p)

wanted = {
    norm(r[manifest_col])
    for r in rows
}

if len(wanted) != 8:
    raise RuntimeError(
        "Smoke manifest image paths are not unique"
    )

# ---------------------------------------------------------------------
# Find every list in a JSON that contains records with image paths.
# ---------------------------------------------------------------------

def locate_lists(obj, path=()):
    found = []

    if isinstance(obj, list):

        if obj and all(
            isinstance(x, dict)
            for x in obj
        ):
            key = next(
                (
                    k for k in PATH_KEYS
                    if all(
                        isinstance(x.get(k), str)
                        and x.get(k)
                        for x in obj
                    )
                ),
                None,
            )

            if key is not None:
                found.append(
                    (path, key, obj)
                )

        for i, value in enumerate(obj):
            found += locate_lists(
                value,
                path + (i,),
            )

    elif isinstance(obj, dict):

        for k, value in obj.items():
            found += locate_lists(
                value,
                path + (k,),
            )

    return found

def get_at(obj, path):
    cur = obj

    for key in path:
        cur = cur[key]

    return cur

def set_at(obj, path, value):

    if not path:
        return value

    cur = obj

    for key in path[:-1]:
        cur = cur[key]

    cur[path[-1]] = value

    return obj

# ---------------------------------------------------------------------
# Use only explicitly generated dataset smoke JSONs.
# DFDC is intentionally excluded.
# ---------------------------------------------------------------------

sources = sorted(
    p
    for p in smoke.glob("*_smoke.json")
    if (
        p.is_file()
        and "dfdc_smoke" not in p.name.lower()
    )
)

print("SMOKE JSON CANDIDATES:")

for p in sources:
    print(" ", p)

if len(sources) < 2:
    raise RuntimeError(
        "Expected at least Celeb + DFDCP smoke JSON files"
    )

accepted = []

for p in sources:

    obj = json.loads(
        p.read_text(
            encoding="utf-8"
        )
    )

    matches = []

    for path, key, items in locate_lists(obj):

        paths = {
            norm(x[key])
            for x in items
        }

        # Candidate may contain only part of the 8-image manifest,
        # but nothing outside it.
        if (
            paths
            and paths <= wanted
        ):
            matches.append(
                (
                    path,
                    key,
                    items,
                    paths,
                )
            )

    if len(matches) == 0:
        continue

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous inference list in {p}: "
            f"{[(x[0], len(x[2])) for x in matches]}"
        )

    accepted.append(
        (
            p,
            obj,
            *matches[0],
        )
    )

if len(accepted) < 2:
    raise RuntimeError(
        "Could not identify both Celeb + DFDCP smoke inference JSONs"
    )

# All accepted JSONs must use exactly the same structural list path/key.
paths = {
    (x[2], x[3])
    for x in accepted
}

if len(paths) != 1:
    raise RuntimeError(
        f"Smoke JSON schemas differ: {paths}"
    )

list_path, image_key = next(iter(paths))

print()
print("IMAGE LIST PATH =", list_path)
print("IMAGE KEY       =", image_key)

# ---------------------------------------------------------------------
# Merge items, refusing semantic conflicts.
# ---------------------------------------------------------------------

combined = {}
owner = {}

for p, obj, path, key, items, image_paths in accepted:

    print(
        f"{p.name}: {len(items)} item(s)"
    )

    for item in items:

        image = norm(item[key])

        if image not in wanted:
            raise RuntimeError(
                f"Unexpected smoke image in {p}: {image}"
            )

        if image in combined:

            # Same image appearing twice is acceptable ONLY if the JSON record
            # is byte-for-byte semantically identical.
            if combined[image] != item:
                raise RuntimeError(
                    f"Conflicting JSON records for image:\n"
                    f"{image}\n"
                    f"A={owner[image]}\n"
                    f"B={p}"
                )

            continue

        combined[image] = item
        owner[image] = str(p)

if set(combined) != wanted:

    missing = sorted(
        wanted - set(combined)
    )

    extra = sorted(
        set(combined) - wanted
    )

    raise RuntimeError(
        f"Combined smoke JSON != manifest.\n"
        f"missing={missing}\n"
        f"extra={extra}"
    )

if len(combined) != 8:
    raise RuntimeError(
        f"Expected exactly 8 combined inputs, got {len(combined)}"
    )

# Preserve manifest order.
ordered = [
    combined[
        norm(r[manifest_col])
    ]
    for r in rows
]

# Use the first proven smoke JSON as structural base.
base = deepcopy(
    accepted[0][1]
)

base = set_at(
    base,
    list_path,
    ordered,
)

out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

tmp = out.with_suffix(
    out.suffix + ".tmp"
)

tmp.write_text(
    json.dumps(
        base,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# ---------------------------------------------------------------------
# Re-read independently.
# ---------------------------------------------------------------------

verify = json.loads(
    tmp.read_text(
        encoding="utf-8"
    )
)

items = get_at(
    verify,
    list_path,
)

if len(items) != 8:
    raise RuntimeError(
        f"Written full_inputs has {len(items)} items"
    )

actual = {
    norm(x[image_key])
    for x in items
}

if actual != wanted:
    raise RuntimeError(
        "Written full_inputs does not exactly match smoke manifest"
    )

tmp.replace(out)

print()
print("✅ FULL INPUTS MATERIALIZED")
print("rows =", len(items))
print("file =", out)

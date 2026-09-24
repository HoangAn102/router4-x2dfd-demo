from pathlib import Path
from copy import deepcopy
import hashlib
import json
import os
import sys

if len(sys.argv) != 4:
    raise RuntimeError(
        "Usage: build_exact_2cross_input.py "
        "OUTPUT CELEB_JSON DFDCP_JSON"
    )

out = Path(sys.argv[1]).expanduser().resolve()

sources = [
    ("Celeb-DF-v2", Path(sys.argv[2]).expanduser().resolve()),
    ("DFDCP",       Path(sys.argv[3]).expanduser().resolve()),
]


def has_dfdc_component(path):
    return any(
        part.upper() == "DFDC"
        for part in path.parts
    )


merged = []
seen = set()
counts = {}

template = None

for expected_dataset, src in sources:

    if not src.is_file():
        raise RuntimeError(
            f"Input JSON missing: {src}"
        )

    obj = json.loads(
        src.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"{src}: root must be JSON object"
        )

    images = obj.get("images")

    if (
        not isinstance(images, list)
        or not images
    ):
        raise RuntimeError(
            f"{src}: root['images'] must be non-empty list"
        )

    if template is None:
        template = deepcopy(obj)

    counts[expected_dataset] = len(images)

    for i, item in enumerate(images):

        if not isinstance(item, dict):
            raise RuntimeError(
                f"{src}: images[{i}] is not object"
            )

        if "image_path" not in item:
            raise RuntimeError(
                f"{src}: images[{i}] lacks image_path"
            )

        p = Path(
            str(item["image_path"])
        ).expanduser().resolve()

        if not p.is_file():
            raise RuntimeError(
                f"Missing frame: {p}"
            )

        if has_dfdc_component(p):
            raise RuntimeError(
                f"DFDC frame forbidden in 2-cross benchmark: {p}"
            )

        key = str(p)

        if key in seen:
            raise RuntimeError(
                f"Duplicate frame across datasets: {p}"
            )

        seen.add(key)

        row = deepcopy(item)
        row["image_path"] = key

        merged.append(row)


if template is None:
    raise RuntimeError(
        "No input template"
    )

template["images"] = merged


# Explicit provenance written alongside the actual JSON;
# do NOT add arbitrary keys to the X2DFD JSON schema.
provenance = {
    "scope": [
        "Celeb-DF-v2",
        "DFDCP",
    ],
    "DFDC": "EXCLUDED",
    "sources": {
        ds: str(path)
        for ds, path in sources
    },
    "source_counts": counts,
    "total_images": len(merged),
    "order": "Celeb-DF-v2 then DFDCP; source order preserved",
}


out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

tmp = out.with_suffix(
    out.suffix + ".tmp"
)

tmp.write_text(
    json.dumps(
        template,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

# Independent reread.
check = json.loads(
    tmp.read_text(
        encoding="utf-8"
    )
)

check_images = check.get("images")

if (
    not isinstance(check_images, list)
    or len(check_images) != len(merged)
):
    raise RuntimeError(
        "Atomic reread failed"
    )


check_paths = [
    str(
        Path(x["image_path"])
        .expanduser()
        .resolve()
    )
    for x in check_images
]

expected_paths = [
    x["image_path"]
    for x in merged
]

if check_paths != expected_paths:
    raise RuntimeError(
        "Image order changed while serializing"
    )


os.replace(
    tmp,
    out,
)


sha = hashlib.sha256(
    out.read_bytes()
).hexdigest()

prov = out.with_suffix(
    out.suffix + ".provenance.json"
)

provenance["sha256"] = sha

prov.write_text(
    json.dumps(
        provenance,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print("2-CROSS INPUT PASS ✅")
print("Celeb-DF-v2 =", counts["Celeb-DF-v2"])
print("DFDCP       =", counts["DFDCP"])
print("DFDC        = 0")
print("TOTAL       =", len(merged))
print("JSON        =", out)
print("SHA256      =", sha)

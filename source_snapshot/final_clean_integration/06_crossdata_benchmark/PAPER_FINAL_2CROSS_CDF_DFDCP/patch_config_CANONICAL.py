from pathlib import Path
from copy import deepcopy
import json
import os
import sys
import yaml

SCHEMA_VERSION = "x2dfd-infer.inputs.images-2cross-v2"

if len(sys.argv) != 6:
    raise RuntimeError(
        "Usage: patch_config.py "
        "SRC_YAML DST_YAML WORK CELEB_JSON DFDCP_JSON"
    )

src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve()
work = Path(sys.argv[3]).expanduser().resolve()

json_paths = [
    Path(sys.argv[4]).expanduser().resolve(),
    Path(sys.argv[5]).expanduser().resolve(),
]

if not src.is_file():
    raise RuntimeError(
        f"Source config missing: {src}"
    )

# ---------------------------------------------------------------------
# Enforce exact 2-cross input contract.
# ---------------------------------------------------------------------

if (
    "CELEB" not in json_paths[0].name.upper()
):
    raise RuntimeError(
        f"First input must be Celeb-DF-v2 JSON: {json_paths[0]}"
    )

if (
    "DFDCP" not in json_paths[1].name.upper()
):
    raise RuntimeError(
        f"Second input must be DFDCP JSON: {json_paths[1]}"
    )

for p in json_paths:

    if not p.is_file():
        raise RuntimeError(
            f"Input JSON missing: {p}"
        )

    upper = p.name.upper()

    if (
        upper.startswith("DFDC_")
        or upper in {
            "DFDC.JSON",
            "DFDC_TEST.JSON",
            "DFDC_SMOKE.JSON",
        }
    ):
        raise RuntimeError(
            f"DFDC input forbidden: {p}"
        )


# ---------------------------------------------------------------------
# Read REAL X2DFD YAML.
# ---------------------------------------------------------------------

cfg = yaml.safe_load(
    src.read_text(
        encoding="utf-8",
        errors="strict",
    )
)

if not isinstance(cfg, dict):
    raise RuntimeError(
        "Config root must be mapping"
    )

infer = cfg.get("infer")

if not isinstance(infer, dict):
    raise RuntimeError(
        "Config missing mapping: infer"
    )

inputs_node = infer.get("inputs")

if not isinstance(inputs_node, dict):
    raise RuntimeError(
        "Config missing mapping: infer.inputs"
    )

if (
    "images" not in inputs_node
    or not isinstance(
        inputs_node["images"],
        list,
    )
):
    raise RuntimeError(
        "Config missing list: infer.inputs.images"
    )


# ---------------------------------------------------------------------
# Merge frozen image records.
# ---------------------------------------------------------------------

merged = []
seen = set()
source_counts = {}

for jp in json_paths:

    obj = json.loads(
        jp.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"{jp}: root must be object"
        )

    images = obj.get("images")

    if (
        not isinstance(images, list)
        or not images
    ):
        raise RuntimeError(
            f"{jp}: root.images must be non-empty list"
        )

    source_counts[
        jp.name
    ] = len(images)

    for idx, item in enumerate(images):

        if not isinstance(item, dict):
            raise RuntimeError(
                f"{jp}[{idx}] is not object"
            )

        if "image_path" not in item:
            raise RuntimeError(
                f"{jp}[{idx}] missing image_path"
            )

        raw = str(
            item["image_path"]
        ).strip()

        if not raw:
            raise RuntimeError(
                f"{jp}[{idx}] empty image_path"
            )

        p = Path(
            raw
        ).expanduser().resolve()

        if not p.is_file():
            raise RuntimeError(
                f"Image missing: {p}"
            )

        # Exact component DFDC.
        # DFDCP does NOT match this.
        if any(
            part.upper() == "DFDC"
            for part in p.parts
        ):
            raise RuntimeError(
                f"DFDC leaked into 2-cross config: {p}"
            )

        key = str(p)

        if key in seen:
            raise RuntimeError(
                f"Duplicate image in frozen inputs: {p}"
            )

        seen.add(key)

        row = deepcopy(item)
        row["image_path"] = key

        merged.append(row)

if not merged:
    raise RuntimeError(
        "Merged frozen input list empty"
    )

cfg[
    "infer"
][
    "inputs"
][
    "images"
] = merged


# ---------------------------------------------------------------------
# Router4 cache binding.
#
# This fixes the hidden bug where base_router4.yaml could still point to
# smoke/work/router_moe_cache.csv during the FULL benchmark.
# ---------------------------------------------------------------------

providers = []

for ws in (
    cfg.get(
        "weak_supplies",
        [],
    )
    or []
):
    if (
        isinstance(ws, dict)
        and ws.get("provider")
        == "router_moe_cache"
    ):
        providers.append(ws)

if len(providers) > 1:
    raise RuntimeError(
        "More than one router_moe_cache provider"
    )

if providers:

    providers[0][
        "cache_csv"
    ] = str(
        work
        / "router_moe_cache.csv"
    )


# ---------------------------------------------------------------------
# Atomic output.
# ---------------------------------------------------------------------

dst.parent.mkdir(
    parents=True,
    exist_ok=True,
)

tmp = dst.with_suffix(
    dst.suffix + ".tmp"
)

tmp.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


# ---------------------------------------------------------------------
# Independent reread.
# ---------------------------------------------------------------------

verify = yaml.safe_load(
    tmp.read_text(
        encoding="utf-8"
    )
)

out_images = (
    verify
    .get("infer", {})
    .get("inputs", {})
    .get("images")
)

if (
    not isinstance(out_images, list)
    or len(out_images) != len(merged)
):
    raise RuntimeError(
        "Written config lost infer.inputs.images"
    )

out_paths = [
    str(
        Path(
            x["image_path"]
        )
        .expanduser()
        .resolve()
    )
    for x in out_images
]

expected_paths = [
    x["image_path"]
    for x in merged
]

if out_paths != expected_paths:
    raise RuntimeError(
        "Written config changed frozen input order/content"
    )

for p in out_paths:

    if any(
        part.upper() == "DFDC"
        for part in Path(p).parts
    ):
        raise RuntimeError(
            f"DFDC appeared after serialization: {p}"
        )


if providers:

    expected_cache = str(
        work
        / "router_moe_cache.csv"
    )

    vr = [
        x
        for x in (
            verify.get(
                "weak_supplies",
                [],
            )
            or []
        )
        if (
            isinstance(x, dict)
            and x.get("provider")
            == "router_moe_cache"
        )
    ]

    if (
        len(vr) != 1
        or vr[0].get(
            "cache_csv"
        ) != expected_cache
    ):
        raise RuntimeError(
            "Router cache binding mismatch: "
            f"expected={expected_cache}, got={vr}"
        )


os.replace(
    tmp,
    dst,
)

print(
    "PATCHER_SCHEMA =",
    SCHEMA_VERSION,
)

print(
    "PATCHED_CONFIG =",
    dst,
)

print(
    "INPUTS =",
    [
        str(p)
        for p in json_paths
    ],
)

print(
    "SOURCE_COUNTS =",
    source_counts,
)

print(
    "TOTAL_IMAGES =",
    len(merged),
)

if providers:
    print(
        "ROUTER_CACHE =",
        work / "router_moe_cache.csv",
    )

print(
    "2-CROSS CONFIG PASS ✅"
)

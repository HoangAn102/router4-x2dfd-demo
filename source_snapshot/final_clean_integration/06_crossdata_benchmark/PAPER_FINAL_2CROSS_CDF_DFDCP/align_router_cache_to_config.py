from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import sys
import yaml


if len(sys.argv) not in (3, 4):
    raise RuntimeError(
        "Usage: align_router_cache_to_config.py "
        "CONFIG_YAML CACHE_CSV [AUDIT_JSON]"
    )


CONFIG = Path(sys.argv[1]).expanduser().resolve()
CACHE = Path(sys.argv[2]).expanduser().resolve()

AUDIT = (
    Path(sys.argv[3]).expanduser().resolve()
    if len(sys.argv) == 4
    else CACHE.with_suffix(".order_alignment.json")
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def norm(value):
    return str(
        Path(
            str(value)
        )
        .expanduser()
        .resolve()
    )


if not CONFIG.is_file():
    raise RuntimeError(
        f"Config missing: {CONFIG}"
    )

if not CACHE.is_file():
    raise RuntimeError(
        f"Cache missing: {CACHE}"
    )


###############################################################################
# CONFIG ORDER
###############################################################################

cfg = yaml.safe_load(
    CONFIG.read_text(
        encoding="utf-8",
        errors="strict",
    )
)

try:
    images = (
        cfg["infer"]
           ["inputs"]
           ["images"]
    )
except Exception as e:
    raise RuntimeError(
        "Config missing infer.inputs.images"
    ) from e


if (
    not isinstance(images, list)
    or not images
):
    raise RuntimeError(
        "Config image list empty/invalid"
    )


config_paths = []

for i, item in enumerate(images):

    if (
        not isinstance(item, dict)
        or "image_path" not in item
    ):
        raise RuntimeError(
            f"Invalid config image row {i}"
        )

    p = norm(
        item["image_path"]
    )

    if not Path(p).is_file():
        raise RuntimeError(
            f"Config image missing: {p}"
        )

    if any(
        x.upper() == "DFDC"
        for x in Path(p).parts
    ):
        raise RuntimeError(
            f"DFDC leaked into 2-cross config: {p}"
        )

    config_paths.append(p)


if len(set(config_paths)) != len(config_paths):
    raise RuntimeError(
        "Duplicate paths in config"
    )


###############################################################################
# CACHE
###############################################################################

with CACHE.open(
    "r",
    newline="",
    encoding="utf-8",
    errors="strict",
) as f:

    reader = csv.DictReader(f)

    fieldnames = reader.fieldnames

    rows = list(reader)


if not fieldnames:
    raise RuntimeError(
        "Router cache has no header"
    )

if not rows:
    raise RuntimeError(
        "Router cache has zero rows"
    )


path_col = next(
    (
        c
        for c in (
            "image_path",
            "path",
            "image",
        )
        if c in fieldnames
    ),
    None,
)


if path_col is None:
    raise RuntimeError(
        "Router cache has no path column. "
        f"Columns={fieldnames}"
    )


for required in (
    "selected_raw_score",
    "selected_alias",
):
    if required not in fieldnames:
        raise RuntimeError(
            f"Router cache missing required column: {required}"
        )


cache_paths = [
    norm(
        row[path_col]
    )
    for row in rows
]


if len(set(cache_paths)) != len(cache_paths):
    raise RuntimeError(
        "Duplicate image paths in router cache"
    )


###############################################################################
# SCIENTIFIC SET CHECK
###############################################################################

config_set = set(
    config_paths
)

cache_set = set(
    cache_paths
)


if config_set != cache_set:

    missing_cache = [
        p
        for p in config_paths
        if p not in cache_set
    ]

    extra_cache = [
        p
        for p in cache_paths
        if p not in config_set
    ]

    raise RuntimeError(
        "Router cache/config IMAGE SET differs.\n"
        f"config={len(config_paths)} "
        f"cache={len(cache_paths)}\n"
        f"missing_in_cache={missing_cache[:10]}\n"
        f"extra_in_cache={extra_cache[:10]}"
    )


###############################################################################
# MAP FULL ROW TO IMAGE PATH
###############################################################################

by_path = {}

for row, p in zip(
    rows,
    cache_paths,
):

    # Row remains completely unchanged.
    by_path[p] = row


ordered_rows = [
    by_path[p]
    for p in config_paths
]


###############################################################################
# HASH BEFORE
###############################################################################

sha_before = sha256(
    CACHE
)

order_before_equal = (
    cache_paths
    == config_paths
)


###############################################################################
# REORDER ONLY IF REQUIRED
###############################################################################

backup = None


if not order_before_equal:

    backup = CACHE.with_name(
        CACHE.name
        + ".BEFORE_CANONICAL_ORDER"
    )

    if not backup.exists():
        shutil.copy2(
            CACHE,
            backup,
        )


    tmp = CACHE.with_suffix(
        CACHE.suffix + ".tmp"
    )


    with tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        writer.writerows(
            ordered_rows
        )


    os.replace(
        tmp,
        CACHE,
    )


###############################################################################
# INDEPENDENT REREAD
###############################################################################

with CACHE.open(
    "r",
    newline="",
    encoding="utf-8",
    errors="strict",
) as f:

    verify_rows = list(
        csv.DictReader(f)
    )


verify_paths = [
    norm(
        row[path_col]
    )
    for row in verify_rows
]


if verify_paths != config_paths:
    raise RuntimeError(
        "Cache order still differs after alignment"
    )


if len(verify_rows) != len(rows):
    raise RuntimeError(
        "Cache row count changed during reorder"
    )


###############################################################################
# VERIFY ROW CONTENT DID NOT CHANGE
###############################################################################

def row_fingerprint(row):

    canonical = json.dumps(
        row,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


before_map = {
    norm(row[path_col]):
        row_fingerprint(row)
    for row in rows
}


after_map = {
    norm(row[path_col]):
        row_fingerprint(row)
    for row in verify_rows
}


if before_map != after_map:
    raise RuntimeError(
        "Cache ROW CONTENT changed while reordering"
    )


###############################################################################
# SCORE SANITY
###############################################################################

for i, row in enumerate(
    verify_rows
):

    try:
        score = float(
            row[
                "selected_raw_score"
            ]
        )
    except Exception as e:
        raise RuntimeError(
            f"Invalid selected_raw_score at row {i}"
        ) from e


    if not (
        score == score
        and score not in (
            float("inf"),
            float("-inf"),
        )
    ):
        raise RuntimeError(
            f"Non-finite score row {i}: {score}"
        )


    alias = str(
        row[
            "selected_alias"
        ]
    ).strip()

    if not alias:
        raise RuntimeError(
            f"Empty selected_alias row {i}"
        )


###############################################################################
# AUDIT
###############################################################################

sha_after = sha256(
    CACHE
)


report = {
    "config": str(CONFIG),
    "cache": str(CACHE),
    "rows": len(rows),
    "same_set": True,
    "order_before_equal": order_before_equal,
    "order_after_equal": True,
    "row_content_preserved": True,
    "sha256_before": sha_before,
    "sha256_after": sha_after,
    "backup": (
        str(backup)
        if backup is not None
        else None
    ),
}


AUDIT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print(
    "CACHE ALIGN PASS ✅"
)

print(
    "rows                =",
    len(rows),
)

print(
    "same image set      = YES"
)

print(
    "order before equal  =",
    order_before_equal,
)

print(
    "order after equal   = YES"
)

print(
    "row content changed = NO"
)

print(
    "cache               =",
    CACHE,
)

print(
    "audit               =",
    AUDIT,
)

from pathlib import Path
import csv
import math
import sys
import yaml

if len(sys.argv) != 6:
    raise RuntimeError(
        "Usage: validate_config_pair.py "
        "ORIG_YAML R4_YAML WORK MODE EXPECTED_COUNT"
    )

orig_path = Path(
    sys.argv[1]
).resolve()

r4_path = Path(
    sys.argv[2]
).resolve()

work = Path(
    sys.argv[3]
).resolve()

mode = sys.argv[4]
expected_count = int(
    sys.argv[5]
)

orig = yaml.safe_load(
    orig_path.read_text(
        encoding="utf-8"
    )
)

r4 = yaml.safe_load(
    r4_path.read_text(
        encoding="utf-8"
    )
)


def get_paths(cfg, name):

    try:
        images = (
            cfg["infer"]
               ["inputs"]
               ["images"]
        )
    except Exception as e:
        raise RuntimeError(
            f"{name}: missing infer.inputs.images"
        ) from e

    if (
        not isinstance(images, list)
        or not images
    ):
        raise RuntimeError(
            f"{name}: images empty/not list"
        )

    out = []

    for i, item in enumerate(images):

        if (
            not isinstance(item, dict)
            or "image_path"
            not in item
        ):
            raise RuntimeError(
                f"{name}: invalid image item {i}"
            )

        p = Path(
            str(
                item["image_path"]
            )
        ).expanduser().resolve()

        if any(
            part.upper() == "DFDC"
            for part in p.parts
        ):
            raise RuntimeError(
                f"{name}: DFDC leaked: {p}"
            )

        out.append(
            str(p)
        )

    return out


a = get_paths(
    orig,
    "Original",
)

b = get_paths(
    r4,
    "Router4",
)

if a != b:
    raise RuntimeError(
        "Original and Router4 frozen image ORDER differs"
    )

if len(set(a)) != len(a):
    raise RuntimeError(
        "Duplicate image paths in frozen config"
    )

if (
    expected_count >= 0
    and len(a) != expected_count
):
    raise RuntimeError(
        f"Expected {expected_count} images; got {len(a)}"
    )


# Fail before LLaVA if any input frame vanished.
missing = [
    p
    for p in a
    if not Path(p).is_file()
]

if missing:
    raise RuntimeError(
        f"Missing inputs={len(missing)} "
        f"first={missing[:5]}"
    )


providers = [
    x
    for x in (
        r4.get(
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

if len(providers) != 1:
    raise RuntimeError(
        "Router4 router_moe_cache "
        f"provider count={len(providers)}"
    )

expected_cache = (
    work
    / "router_moe_cache.csv"
).resolve()

actual_cache = Path(
    str(
        providers[0].get(
            "cache_csv",
            "",
        )
    )
).expanduser().resolve()

if actual_cache != expected_cache:
    raise RuntimeError(
        "Router cache path mismatch:\n"
        f"actual={actual_cache}\n"
        f"expected={expected_cache}"
    )


# Runtime modes require an actual completed cache.
if mode in {
    "smoke-runtime",
    "full-runtime",
}:

    if not expected_cache.is_file():
        raise RuntimeError(
            f"Router cache missing: {expected_cache}"
        )

    with expected_cache.open(
        newline="",
        encoding="utf-8",
        errors="strict",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if not rows:
        raise RuntimeError(
            "Router cache empty"
        )

    cols = set(
        rows[0]
    )

    path_col = next(
        (
            c
            for c in (
                "image_path",
                "path",
                "image",
            )
            if c in cols
        ),
        None,
    )

    if path_col is None:
        raise RuntimeError(
            "Router cache lacks image-path column; "
            f"cols={sorted(cols)}"
        )

    for col in (
        "selected_raw_score",
        "selected_alias",
    ):
        if col not in cols:
            raise RuntimeError(
                f"Router cache missing {col}"
            )

    cache_paths = [
        str(
            Path(
                str(r[path_col])
            )
            .expanduser()
            .resolve()
        )
        for r in rows
    ]

    if cache_paths != a:

        if set(cache_paths) != set(a):
            raise RuntimeError(
                "Router cache/config SET mismatch: "
                f"cache={len(cache_paths)}, config={len(a)}"
            )

        raise RuntimeError(
            "Router cache/config contain same images "
            "but ORDER differs"
        )


    for i, r in enumerate(rows):

        try:
            x = float(
                r["selected_raw_score"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Invalid selected_raw_score row={i}"
            ) from e

        if not math.isfinite(x):
            raise RuntimeError(
                f"Non-finite router score row={i}: {x}"
            )

        if not str(
            r["selected_alias"]
        ).strip():
            raise RuntimeError(
                f"Empty selected_alias row={i}"
            )


print(
    f"PAIR_CHECK PASS "
    f"mode={mode} "
    f"images={len(a)}"
)

print(
    "router_cache =",
    expected_cache,
)

print(
    "DFDC = 0"
)

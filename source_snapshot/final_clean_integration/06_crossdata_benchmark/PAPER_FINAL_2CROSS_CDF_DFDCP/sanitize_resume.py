from pathlib import Path
import csv
import json
import os
import sys
import time

if len(sys.argv) != 4:
    raise RuntimeError(
        "Usage: sanitize_resume.py "
        "WORK CELEB_JSON DFDCP_JSON"
    )

work = Path(
    sys.argv[1]
).resolve()

input_jsons = [
    Path(
        sys.argv[2]
    ).resolve(),
    Path(
        sys.argv[3]
    ).resolve(),
]

manifest = (
    work
    / "manifest.csv"
)


def load_input_paths(p):

    if not p.is_file():
        raise RuntimeError(
            f"Frozen input JSON missing: {p}"
        )

    obj = json.loads(
        p.read_text(
            encoding="utf-8"
        )
    )

    images = (
        obj.get("images")
        if isinstance(obj, dict)
        else None
    )

    if (
        not isinstance(images, list)
        or not images
    ):
        raise RuntimeError(
            f"{p}: root.images invalid"
        )

    out = []

    for i, item in enumerate(images):

        if (
            not isinstance(item, dict)
            or "image_path"
            not in item
        ):
            raise RuntimeError(
                f"{p}: invalid item {i}"
            )

        q = Path(
            str(
                item["image_path"]
            )
        ).expanduser().resolve()

        if any(
            part.upper() == "DFDC"
            for part in q.parts
        ):
            raise RuntimeError(
                f"DFDC in authoritative input: {q}"
            )

        out.append(
            str(q)
        )

    return out


wanted = []

for p in input_jsons:
    wanted.extend(
        load_input_paths(p)
    )

if len(set(wanted)) != len(wanted):
    raise RuntimeError(
        "Duplicate paths across authoritative "
        "Celeb/DFDCP inputs"
    )

wanted_set = set(wanted)


if not manifest.is_file():
    raise RuntimeError(
        "Manifest missing before full expert run: "
        f"{manifest}"
    )


with manifest.open(
    newline="",
    encoding="utf-8",
    errors="strict",
) as f:

    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)


if (
    not fieldnames
    or not rows
):
    raise RuntimeError(
        "Manifest has no header/rows"
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
        f"Manifest lacks path column: {fieldnames}"
    )


def norm(x):
    return str(
        Path(
            str(x)
        )
        .expanduser()
        .resolve()
    )


by_path = {}

for r in rows:

    p = norm(
        r[path_col]
    )

    if p in by_path:
        raise RuntimeError(
            f"Duplicate manifest path: {p}"
        )

    by_path[p] = r


missing = [
    p
    for p in wanted
    if p not in by_path
]

if missing:
    raise RuntimeError(
        "Manifest missing authoritative 2-cross frames: "
        f"count={len(missing)} "
        f"first={missing[:5]}"
    )


# Preserve all original row metadata,
# but keep only authoritative two-dataset rows in canonical order.
filtered = [
    by_path[p]
    for p in wanted
]


for i, r in enumerate(filtered):

    p = Path(
        norm(
            r[path_col]
        )
    )

    if any(
        part.upper() == "DFDC"
        for part in p.parts
    ):
        raise RuntimeError(
            f"DFDC remained in filtered manifest row {i}"
        )

    if "dataset" in fieldnames:

        ds = str(
            r.get(
                "dataset",
                "",
            )
        ).strip()

        if ds not in {
            "Celeb-DF-v2",
            "DFDCP",
        }:
            raise RuntimeError(
                f"Unexpected kept dataset row={i}: {ds!r}"
            )


old_order = [
    norm(
        r[path_col]
    )
    for r in rows
]

new_order = [
    norm(
        r[path_col]
    )
    for r in filtered
]

changed = (
    old_order != new_order
)


if changed:

    backup = manifest.with_name(
        manifest.name
        + ".BEFORE_2CROSS_FILTER"
    )

    if not backup.exists():
        backup.write_bytes(
            manifest.read_bytes()
        )

    tmp = manifest.with_suffix(
        ".csv.tmp"
    )

    with tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        w.writeheader()
        w.writerows(
            filtered
        )

    os.replace(
        tmp,
        manifest,
    )

    print(
        "MANIFEST_FILTERED",
        len(rows),
        "->",
        len(filtered),
    )

    print(
        "backup =",
        backup,
    )

else:

    print(
        "MANIFEST_ALREADY_CANONICAL",
        len(filtered),
    )


stamp = time.strftime(
    "%Y%m%d_%H%M%S"
)

manifest_set = wanted_set


# Existing expert score CSV:
# partial canonical subset is valid resume.
for name in (
    "blending_scores.csv",
    "diffusion_scores.csv",
    "frequency_scores.csv",
    "texture_scores.csv",
):

    p = (
        work
        / name
    )

    if (
        not p.exists()
        or p.stat().st_size == 0
    ):
        continue


    with p.open(
        newline="",
        encoding="utf-8",
        errors="strict",
    ) as f:
        erows = list(
            csv.DictReader(f)
        )


    valid = False

    if erows:

        cols = set(
            erows[0]
        )

        pc = next(
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

        if pc:

            paths = [
                norm(
                    r[pc]
                )
                for r in erows
            ]

            valid = (
                len(
                    set(paths)
                )
                == len(paths)
                and set(paths).issubset(
                    manifest_set
                )
                and all(
                    not any(
                        part.upper()
                        == "DFDC"
                        for part in Path(x).parts
                    )
                    for x in paths
                )
            )


    if not valid:

        q = p.with_name(
            p.name
            + f".STALE_{stamp}"
        )

        p.rename(q)

        print(
            "QUARANTINED",
            p,
            "->",
            q,
        )

    else:

        print(
            "RESUME_OK",
            name,
            len(erows),
            "/",
            len(wanted),
        )


# Router cache is only reusable when COMPLETE
# and EXACTLY aligned with canonical manifest order.
cache = (
    work
    / "router_moe_cache.csv"
)

if (
    cache.exists()
    and cache.stat().st_size
):

    with cache.open(
        newline="",
        encoding="utf-8",
        errors="strict",
    ) as f:
        crows = list(
            csv.DictReader(f)
        )


    valid = False

    if crows:

        cols = set(
            crows[0]
        )

        pc = next(
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

        if pc:

            paths = [
                norm(
                    r[pc]
                )
                for r in crows
            ]

            valid = (
                paths == wanted
                and len(
                    set(paths)
                )
                == len(paths)
                and "selected_raw_score"
                in cols
                and "selected_alias"
                in cols
            )


    if not valid:

        q = cache.with_name(
            cache.name
            + f".STALE_{stamp}"
        )

        cache.rename(q)

        print(
            "QUARANTINED router cache",
            cache,
            "->",
            q,
        )

    else:

        print(
            "ROUTER_CACHE_RESUME_OK",
            len(crows),
        )


print(
    "2-CROSS MANIFEST/RESUME SANITIZER PASS; "
    f"rows={len(wanted)}"
)

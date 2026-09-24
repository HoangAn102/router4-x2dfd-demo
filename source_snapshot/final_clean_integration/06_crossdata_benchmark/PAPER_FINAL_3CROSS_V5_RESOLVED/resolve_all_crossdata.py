from pathlib import Path, PurePosixPath
from collections import defaultdict, Counter
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile


BASE = Path(sys.argv[1])
REAL = Path(sys.argv[2])
RAW = Path(sys.argv[3])
DOWNLOAD = Path(sys.argv[4])
SHADOW = Path(sys.argv[5])
RECOVERY = Path(sys.argv[6])
AUDIT = Path(sys.argv[7])

for p in [SHADOW, RECOVERY, AUDIT]:
    p.mkdir(parents=True, exist_ok=True)


###############################################################################
# GENERIC HELPERS
###############################################################################

def die(msg):
    raise RuntimeError(msg)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def count_png(folder):
    folder = Path(folder)

    if not folder.is_dir():
        return 0

    return sum(
        1
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".png"
    )


def safe_symlink(source, target):
    source = Path(source).resolve()
    target = Path(target)

    if not source.is_dir():
        die(f"Symlink source is not directory: {source}")

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if target.exists() or target.is_symlink():

        if target.is_symlink() or target.is_file():
            target.unlink()

        elif target.is_dir():
            shutil.rmtree(target)

    target.symlink_to(
        source,
        target_is_directory=True,
    )


def normalized_parts(value):
    return [
        str(x).lower()
        for x in value
        if str(x)
        and str(x).lower()
        not in {
            ".",
            "..",
            "frames",
            "frame",
        }
    ]


###############################################################################
# FILE METADATA RESOLUTION
###############################################################################

def exact_files(root, filename):
    root = Path(root)

    if not root.exists():
        return []

    return sorted(
        {
            p.resolve()
            for p in root.rglob(filename)
            if p.is_file()
        }
    )


def exact_zip_members(zip_path, filename):
    zip_path = Path(zip_path)

    if not zip_path.is_file():
        return []

    result = []

    with zipfile.ZipFile(zip_path) as z:

        for info in z.infolist():

            if info.is_dir():
                continue

            if (
                PurePosixPath(info.filename).name.lower()
                != filename.lower()
            ):
                continue

            result.append(
                (
                    info.filename,
                    z.read(info.filename),
                )
            )

    return result


def choose_metadata(
    dataset,
    filename,
    extracted_root,
    raw_zip,
    download_tokens,
):

    candidates = []

    # Highest priority: extracted teacher package.
    for p in exact_files(
        extracted_root,
        filename,
    ):
        candidates.append({
            "priority": 300,
            "source": str(p),
            "data": p.read_bytes(),
        })

    # Raw ZIP.
    for member, data in exact_zip_members(
        raw_zip,
        filename,
    ):
        candidates.append({
            "priority": 200,
            "source":
                f"{raw_zip}::{member}",
            "data": data,
        })

    # Download tree.
    if DOWNLOAD.exists():

        for p in DOWNLOAD.rglob(filename):

            if not p.is_file():
                continue

            low = str(p).lower()

            if download_tokens and not any(
                token.lower() in low
                for token in download_tokens
            ):
                continue

            candidates.append({
                "priority": 100,
                "source": str(p.resolve()),
                "data": p.read_bytes(),
            })

    if not candidates:
        die(
            f"No metadata source found for "
            f"{dataset}/{filename}"
        )

    for c in candidates:
        c["sha256"] = sha256_bytes(
            c["data"]
        )

    highest = max(
        c["priority"]
        for c in candidates
    )

    top = [
        c
        for c in candidates
        if c["priority"] == highest
    ]

    # Two equally authoritative copies may exist,
    # but must contain identical bytes.
    top_hashes = {
        c["sha256"]
        for c in top
    }

    if len(top_hashes) != 1:
        die(
            f"Conflicting equally authoritative metadata "
            f"for {dataset}/{filename}: "
            f"{[c['source'] for c in top]}"
        )

    selected = top[0]

    return selected, [
        {
            "priority": c["priority"],
            "source": c["source"],
            "sha256": c["sha256"],
        }
        for c in candidates
    ]


###############################################################################
# DIRECTORY CATALOG
#
# Any directory containing direct PNG children is a candidate preprocessed
# video/frame directory. We do not assume a fixed extraction layout here.
###############################################################################

def catalog_png_dirs(root):
    root = Path(root)

    by_name = defaultdict(list)
    all_dirs = []

    if not root.exists():
        return by_name, all_dirs

    for current, dirs, files in os.walk(root):

        pngs = [
            x
            for x in files
            if x.lower().endswith(".png")
        ]

        if not pngs:
            continue

        p = Path(current).resolve()

        item = {
            "path": p,
            "count": len(pngs),
            "parts": normalized_parts(
                p.parts
            ),
        }

        by_name[
            p.name.lower()
        ].append(item)

        all_dirs.append(item)

    return by_name, all_dirs


###############################################################################
# ZIP PREPROCESSED-PNG CATALOG
#
# Important:
# We only recover PNG files ALREADY existing in ZIP.
# We never generate new frames.
###############################################################################

def catalog_zip_png_dirs(zip_path):
    zip_path = Path(zip_path)

    result = defaultdict(
        lambda: defaultdict(list)
    )

    if not zip_path.is_file():
        return result

    with zipfile.ZipFile(zip_path) as z:

        for info in z.infolist():

            if (
                info.is_dir()
                or not info.filename.lower().endswith(".png")
            ):
                continue

            parent = str(
                PurePosixPath(
                    info.filename
                ).parent
            )

            basename = (
                PurePosixPath(
                    parent
                ).name.lower()
            )

            result[
                basename
            ][
                parent
            ].append(
                info.filename
            )

    return result


def score_parts(candidate_parts, hints):
    candidate_parts = set(
        normalized_parts(
            candidate_parts
        )
    )

    score = 0

    for hint in normalized_parts(
        hints
    ):

        if hint in candidate_parts:
            score += 10

    return score


def choose_unique_candidate(
    candidates,
    hints,
):

    if not candidates:
        return None, "NONE"

    if len(candidates) == 1:
        return candidates[0], "UNIQUE"

    scored = []

    for c in candidates:

        score = score_parts(
            c["parts"],
            hints,
        )

        scored.append(
            (
                score,
                str(c["path"]),
                c,
            )
        )

    scored.sort(
        reverse=True,
        key=lambda x: (
            x[0],
            x[1],
        ),
    )

    best_score = scored[0][0]

    best = [
        x
        for x in scored
        if x[0] == best_score
    ]

    if (
        best_score > 0
        and len(best) == 1
    ):
        return (
            best[0][2],
            f"SCORED_{best_score}",
        )

    return None, (
        "AMBIGUOUS:"
        + "|".join(
            x[1]
            for x in scored[:10]
        )
    )


def choose_zip_parent(
    parent_to_members,
    hints,
):

    if not parent_to_members:
        return None, "ZIP_NONE"

    candidates = []

    for parent, members in (
        parent_to_members.items()
    ):

        candidates.append({
            "parent": parent,
            "members": members,
            "parts":
                PurePosixPath(
                    parent
                ).parts,
        })

    if len(candidates) == 1:
        return candidates[0], "ZIP_UNIQUE"

    scored = []

    for c in candidates:

        score = score_parts(
            c["parts"],
            hints,
        )

        scored.append(
            (
                score,
                c["parent"],
                c,
            )
        )

    scored.sort(
        reverse=True,
        key=lambda x: (
            x[0],
            x[1],
        ),
    )

    top_score = scored[0][0]

    top = [
        x
        for x in scored
        if x[0] == top_score
    ]

    if (
        top_score > 0
        and len(top) == 1
    ):
        return (
            top[0][2],
            f"ZIP_SCORED_{top_score}",
        )

    return None, (
        "ZIP_AMBIGUOUS:"
        + "|".join(
            x[1]
            for x in scored[:10]
        )
    )


def extract_zip_png_group(
    zip_path,
    parent,
    members,
    destination,
):

    zip_path = Path(zip_path)
    destination = Path(destination)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    names = []

    with zipfile.ZipFile(zip_path) as z:

        for member in sorted(members):

            basename = (
                PurePosixPath(
                    member
                ).name
            )

            target = (
                destination
                / basename
            )

            if target.exists():
                die(
                    f"Duplicate recovered filename "
                    f"in {destination}: {basename}"
                )

            target.write_bytes(
                z.read(member)
            )

            names.append(
                basename
            )

    if not names:
        die(
            f"ZIP group unexpectedly empty: "
            f"{zip_path}::{parent}"
        )

    return destination.resolve()


###############################################################################
# RESOLVE VIDEO DIRECTORY
###############################################################################

def resolve_video(
    dataset,
    video_id,
    hints,
    extracted_by_name,
    zip_catalog,
    raw_zip,
    recovery_subdir,
):

    key = video_id.lower()

    # 1. Already-extracted data.
    item, reason = choose_unique_candidate(
        extracted_by_name.get(
            key,
            [],
        ),
        hints,
    )

    if item is not None:

        return {
            "status": "RESOLVED",
            "source_type":
                "EXTRACTED",

            "reason":
                reason,

            "path":
                item["path"],

            "frames":
                item["count"],
        }

    extracted_reason = reason

    # 2. Already-preprocessed PNGs inside raw ZIP.
    zip_item, zip_reason = (
        choose_zip_parent(
            zip_catalog.get(
                key,
                {},
            ),
            hints,
        )
    )

    if zip_item is not None:

        destination = (
            RECOVERY
            / recovery_subdir
            / video_id
        )

        if destination.exists():
            shutil.rmtree(
                destination
            )

        recovered = extract_zip_png_group(
            raw_zip,
            zip_item["parent"],
            zip_item["members"],
            destination,
        )

        return {
            "status": "RESOLVED",
            "source_type":
                "RECOVERED_FROM_ZIP",

            "reason":
                zip_reason,

            "path":
                recovered,

            "frames":
                count_png(
                    recovered
                ),

            "zip_parent":
                zip_item[
                    "parent"
                ],
        }

    return {
        "status": "UNRESOLVED",
        "source_type": "",
        "reason":
            extracted_reason
            + " ; "
            + zip_reason,

        "path": None,
        "frames": 0,
    }


###############################################################################
# COMMON AUDIT WRITER
###############################################################################

audit_rows = []


def add_audit(
    dataset,
    metadata_key,
    video_id,
    resolution,
):

    audit_rows.append({
        "dataset":
            dataset,

        "metadata_key":
            metadata_key,

        "video_id":
            video_id,

        "status":
            resolution[
                "status"
            ],

        "source_type":
            resolution.get(
                "source_type",
                "",
            ),

        "reason":
            resolution.get(
                "reason",
                "",
            ),

        "resolved_path":
            str(
                resolution.get(
                    "path",
                    "",
                )
                or ""
            ),

        "frames":
            resolution.get(
                "frames",
                0,
            ),

        "zip_parent":
            resolution.get(
                "zip_parent",
                "",
            ),
    })


###############################################################################
# CELEB-DF-v2
###############################################################################

print()
print("=" * 80)
print("CELEB-DF-v2")
print("=" * 80)

celeb_root = (
    REAL
    / "Celeb-DF-v2"
)

celeb_meta, celeb_meta_all = (
    choose_metadata(
        "Celeb-DF-v2",
        "List_of_testing_videos.txt",
        celeb_root,
        RAW / "Celeb-DF-v2.zip",
        [
            "celeb",
            "cdf",
        ],
    )
)

celeb_text = (
    celeb_meta["data"]
    .decode(
        "utf-8",
        errors="ignore",
    )
)

celeb_catalog, _ = (
    catalog_png_dirs(
        celeb_root
    )
)

celeb_zip_catalog = (
    catalog_zip_png_dirs(
        RAW
        / "Celeb-DF-v2.zip"
    )
)

celeb_shadow = (
    SHADOW
    / "Celeb-DF-v2"
)

celeb_shadow.mkdir(
    parents=True,
    exist_ok=True,
)

(
    celeb_shadow
    / "List_of_testing_videos.txt"
).write_bytes(
    celeb_meta["data"]
)

celeb_total = 0
celeb_unresolved = []


for raw_line in celeb_text.splitlines():

    line = raw_line.strip()

    if not line:
        continue

    tokens = line.split()

    mp4 = next(
        (
            token
            for token in tokens
            if token.lower().endswith(
                ".mp4"
            )
        ),
        None,
    )

    if mp4 is None:
        continue

    rel = Path(mp4)

    category = rel.parent.name
    video_id = rel.stem

    if category not in {
        "Celeb-real",
        "Celeb-synthesis",
        "YouTube-real",
    }:
        die(
            f"Unknown Celeb official category: "
            f"{category}"
        )

    resolution = resolve_video(
        "Celeb-DF-v2",
        video_id,
        [
            category,
            video_id,
        ],
        celeb_catalog,
        celeb_zip_catalog,
        RAW
        / "Celeb-DF-v2.zip",
        Path(
            "Celeb-DF-v2"
        )
        / category,
    )

    add_audit(
        "Celeb-DF-v2",
        mp4,
        video_id,
        resolution,
    )

    celeb_total += 1

    if (
        resolution[
            "status"
        ]
        != "RESOLVED"
    ):

        celeb_unresolved.append(
            mp4
        )

        continue

    target = (
        celeb_shadow
        / category
        / "frames"
        / video_id
    )

    safe_symlink(
        resolution["path"],
        target,
    )


if celeb_total < 100:
    die(
        f"Suspicious Celeb official test count: "
        f"{celeb_total}"
    )

if celeb_unresolved:
    die(
        f"Celeb unresolved videos: "
        f"{len(celeb_unresolved)}/"
        f"{celeb_total}; "
        f"examples={celeb_unresolved[:30]}"
    )

print(
    "Celeb official videos =",
    celeb_total,
)

print("Celeb unresolved = 0")


###############################################################################
# DFDCP
###############################################################################

print()
print("=" * 80)
print("DFDCP")
print("=" * 80)

dfdcp_root = (
    REAL
    / "DFDCP"
)

dfdcp_meta, dfdcp_meta_all = (
    choose_metadata(
        "DFDCP",
        "dataset.json",
        dfdcp_root,
        RAW / "DFDCP.zip",
        [
            "dfdcp",
        ],
    )
)

try:
    dfdcp_json = json.loads(
        dfdcp_meta["data"]
        .decode(
            "utf-8",
            errors="ignore",
        )
    )

except Exception as exc:
    die(
        f"DFDCP dataset.json invalid: "
        f"{exc}"
    )


if not isinstance(
    dfdcp_json,
    dict,
):
    die(
        "DFDCP dataset.json "
        "must be dictionary"
    )


dfdcp_catalog, _ = (
    catalog_png_dirs(
        dfdcp_root
    )
)

dfdcp_zip_catalog = (
    catalog_zip_png_dirs(
        RAW
        / "DFDCP.zip"
    )
)

dfdcp_shadow = (
    SHADOW
    / "DFDCP"
)

dfdcp_shadow.mkdir(
    parents=True,
    exist_ok=True,
)

(
    dfdcp_shadow
    / "dataset.json"
).write_bytes(
    dfdcp_meta["data"]
)

dfdcp_total = 0
dfdcp_unresolved = []


for metadata_key, info in (
    dfdcp_json.items()
):

    if not isinstance(
        info,
        dict,
    ):
        continue

    split = str(
        info.get(
            "set",
            "",
        )
    ).strip().lower()

    if split != "test":
        continue

    label = str(
        info.get(
            "label",
            "",
        )
    ).strip().lower()

    if label not in {
        "real",
        "fake",
    }:
        die(
            f"DFDCP invalid label: "
            f"{metadata_key} -> {label}"
        )

    parts = Path(
        metadata_key
    ).parts

    if not parts:
        die(
            f"DFDCP invalid key: "
            f"{metadata_key}"
        )

    video_id = Path(
        parts[-1]
    ).stem

    # The existing X2DFD manifest builder
    # expects first key component / frames / video.
    index = (
        parts[0]
        if len(parts) > 1
        else "0"
    )

    hints = [
        *parts[:-1],
        video_id,
    ]

    resolution = resolve_video(
        "DFDCP",
        video_id,
        hints,
        dfdcp_catalog,
        dfdcp_zip_catalog,
        RAW
        / "DFDCP.zip",
        Path(
            "DFDCP"
        )
        / str(index),
    )

    add_audit(
        "DFDCP",
        metadata_key,
        video_id,
        resolution,
    )

    dfdcp_total += 1

    if (
        resolution[
            "status"
        ]
        != "RESOLVED"
    ):

        dfdcp_unresolved.append(
            {
                "key":
                    metadata_key,

                "reason":
                    resolution[
                        "reason"
                    ],
            }
        )

        continue

    target = (
        dfdcp_shadow
        / str(index)
        / "frames"
        / video_id
    )

    safe_symlink(
        resolution["path"],
        target,
    )


if dfdcp_total < 10:
    die(
        f"Suspicious DFDCP test count: "
        f"{dfdcp_total}"
    )


if dfdcp_unresolved:

    path = (
        AUDIT
        / "DFDCP_UNRESOLVED.json"
    )

    path.write_text(
        json.dumps(
            dfdcp_unresolved,
            indent=2,
        )
    )

    die(
        f"DFDCP is NOT fully resolved: "
        f"{len(dfdcp_unresolved)}/"
        f"{dfdcp_total}. "
        f"Report={path}"
    )


print(
    "DFDCP official test videos =",
    dfdcp_total,
)

print("DFDCP unresolved = 0")


###############################################################################
# DFDC FRAME CATALOG
###############################################################################

print()
print("=" * 80)
print("DFDC")
print("=" * 80)

dfdc_root = (
    REAL
    / "DFDC"
)

dfdc_catalog, dfdc_all_dirs = (
    catalog_png_dirs(
        dfdc_root
    )
)

if not dfdc_all_dirs:
    die(
        f"No extracted DFDC PNG directories "
        f"under {dfdc_root}"
    )


# DFDC video stems must be unique for label joining.
duplicates = {
    name: items
    for name, items
    in dfdc_catalog.items()
    if len(items) > 1
}

if duplicates:

    duplicate_report = {
        name: [
            str(x["path"])
            for x in items
        ]
        for name, items
        in list(
            duplicates.items()
        )[:100]
    }

    path = (
        AUDIT
        / "DFDC_DUPLICATE_VIDEO_IDS.json"
    )

    path.write_text(
        json.dumps(
            duplicate_report,
            indent=2,
        )
    )

    die(
        "DFDC has ambiguous duplicated "
        f"video folder names. Report={path}"
    )


dfdc_frame_ids = set(
    dfdc_catalog.keys()
)


###############################################################################
# DFDC EXPLICIT LABEL CANDIDATES
###############################################################################

def normalize_label(value):

    s = str(
        value
    ).strip().lower()

    if s in {
        "0",
        "0.0",
        "real",
        "false",
        "dfdc_real",
    }:
        return 0

    if s in {
        "1",
        "1.0",
        "fake",
        "true",
        "dfdc_fake",
    }:
        return 1

    return None


def parse_label_csv(data):

    reader = csv.DictReader(
        io.StringIO(
            data.decode(
                "utf-8-sig",
                errors="ignore",
            )
        )
    )

    fields = (
        reader.fieldnames
        or []
    )

    file_col = next(
        (
            x
            for x in fields
            if x.lower()
            in {
                "filename",
                "file",
                "video",
                "video_name",
            }
        ),
        None,
    )

    label_col = next(
        (
            x
            for x in fields
            if x.lower()
            in {
                "label",
                "class",
                "target",
            }
        ),
        None,
    )

    split_col = next(
        (
            x
            for x in fields
            if x.lower()
            in {
                "split",
                "set",
                "partition",
            }
        ),
        None,
    )

    if (
        file_col is None
        or label_col is None
    ):
        return None

    mapping = {}

    for row in reader:

        if split_col is not None:

            split = str(
                row.get(
                    split_col,
                    "",
                )
            ).strip().lower()

            if (
                split
                and split != "test"
            ):
                continue

        filename = str(
            row.get(
                file_col,
                "",
            )
        ).strip()

        if not filename:
            continue

        label = normalize_label(
            row.get(
                label_col,
                "",
            )
        )

        if label is None:
            continue

        video_id = (
            Path(
                filename
            ).stem.lower()
        )

        old = mapping.get(
            video_id
        )

        if (
            old is not None
            and old != label
        ):
            return None

        mapping[
            video_id
        ] = label

    return (
        mapping
        if mapping
        else None
    )


def parse_label_json(data):

    try:
        obj = json.loads(
            data.decode(
                "utf-8-sig",
                errors="ignore",
            )
        )
    except Exception:
        return None

    mapping = {}
    conflict = False


    def add(
        video,
        label,
        split=None,
    ):
        nonlocal conflict

        if split is not None:

            split_text = str(
                split
            ).strip().lower()

            if (
                split_text
                and split_text != "test"
            ):
                return

        normalized = (
            normalize_label(
                label
            )
        )

        if normalized is None:
            return

        video_id = (
            Path(
                str(video)
            ).stem.lower()
        )

        if not video_id:
            return

        old = mapping.get(
            video_id
        )

        if (
            old is not None
            and old != normalized
        ):
            conflict = True
            return

        mapping[
            video_id
        ] = normalized


    def walk(node):

        if isinstance(
            node,
            dict,
        ):

            # filename-keyed record
            for key, value in (
                node.items()
            ):

                if (
                    isinstance(
                        value,
                        dict,
                    )
                    and "label"
                    in value
                ):

                    add(
                        key,
                        value.get(
                            "label"
                        ),
                        value.get(
                            "split",
                            value.get(
                                "set"
                            ),
                        ),
                    )

            # explicit record object
            if "label" in node:

                filename = None

                for key in [
                    "filename",
                    "file",
                    "video",
                    "video_name",
                ]:

                    value = (
                        node.get(
                            key
                        )
                    )

                    if (
                        isinstance(
                            value,
                            str,
                        )
                        and value
                    ):
                        filename = value
                        break

                if filename:

                    add(
                        filename,
                        node.get(
                            "label"
                        ),
                        node.get(
                            "split",
                            node.get(
                                "set"
                            ),
                        ),
                    )

            for value in (
                node.values()
            ):
                walk(value)

        elif isinstance(
            node,
            list,
        ):

            for value in node:
                walk(value)


    walk(obj)

    if conflict:
        return None

    return (
        mapping
        if mapping
        else None
    )


label_candidates = []


def add_label_candidate(
    source,
    kind,
    data,
    priority,
):

    mapping = (
        parse_label_csv(data)
        if kind == "csv"
        else parse_label_json(data)
    )

    if not mapping:
        return

    matched = (
        dfdc_frame_ids
        & set(mapping)
    )

    coverage = (
        len(matched)
        / len(
            dfdc_frame_ids
        )
    )

    label_candidates.append({
        "source":
            source,

        "kind":
            kind,

        "priority":
            priority,

        "data":
            data,

        "mapping":
            mapping,

        "coverage":
            coverage,

        "matched":
            len(matched),

        "labels":
            len(mapping),

        "missing":
            sorted(
                dfdc_frame_ids
                - set(mapping)
            ),
    })


###############################################################################
# DFDC local metadata
###############################################################################

search_roots = [
    (
        dfdc_root,
        500,
    ),
    (
        DOWNLOAD,
        300,
    ),
    (
        BASE
        / "projects"
        / "X2DFD",
        200,
    ),
]

for search_root, base_priority in (
    search_roots
):

    if not search_root.exists():
        continue

    for current, dirs, files in (
        os.walk(
            search_root
        )
    ):

        for filename in files:

            p = (
                Path(current)
                / filename
            )

            low_path = str(
                p
            ).lower()

            low_name = (
                filename.lower()
            )

            try:

                size = (
                    p.stat()
                    .st_size
                )

            except Exception:
                continue

            if (
                size <= 0
                or size
                > 256
                * 1024
                * 1024
            ):
                continue

            if low_name == "labels.csv":

                add_label_candidate(
                    str(
                        p.resolve()
                    ),
                    "csv",
                    p.read_bytes(),
                    base_priority
                    + 100,
                )

            elif (
                p.suffix.lower()
                == ".json"
                and (
                    "dfdc"
                    in low_path
                    or "label"
                    in low_name
                    or "metadata"
                    in low_name
                )
            ):

                add_label_candidate(
                    str(
                        p.resolve()
                    ),
                    "json",
                    p.read_bytes(),
                    base_priority,
                )


###############################################################################
# DFDC metadata inside raw ZIP
###############################################################################

dfdc_zip = (
    RAW
    / "DFDC.zip"
)

if dfdc_zip.is_file():

    with zipfile.ZipFile(
        dfdc_zip
    ) as z:

        for info in z.infolist():

            if info.is_dir():
                continue

            if (
                info.file_size <= 0
                or info.file_size
                > 256
                * 1024
                * 1024
            ):
                continue

            low = (
                info.filename.lower()
            )

            basename = (
                PurePosixPath(
                    low
                ).name
            )

            if basename == "labels.csv":

                add_label_candidate(
                    f"{dfdc_zip}::{info.filename}",
                    "csv",
                    z.read(
                        info.filename
                    ),
                    1000,
                )

            elif (
                low.endswith(
                    ".json"
                )
                and (
                    "dfdc"
                    in low
                    or "label"
                    in basename
                    or "metadata"
                    in basename
                )
            ):

                add_label_candidate(
                    f"{dfdc_zip}::{info.filename}",
                    "json",
                    z.read(
                        info.filename
                    ),
                    900,
                )


###############################################################################
# Strict canonical-layout fallback.
#
# This is NOT generic folder guessing.
# We only accept the exact DeepfakeBench-style class tokens:
#
#     DFDC_Real
#     DFDC_Fake
#
# and only if EVERY extracted DFDC frame directory has exactly one of them.
###############################################################################

canonical_mapping = {}
canonical_valid = True


for item in dfdc_all_dirs:

    parts = [
        x.lower()
        for x in item[
            "path"
        ].parts
    ]

    has_real = (
        "dfdc_real"
        in parts
    )

    has_fake = (
        "dfdc_fake"
        in parts
    )

    if has_real == has_fake:
        canonical_valid = False
        break

    canonical_mapping[
        item["path"]
        .name.lower()
    ] = (
        0
        if has_real
        else 1
    )


if (
    canonical_valid
    and set(
        canonical_mapping
    )
    == dfdc_frame_ids
    and set(
        canonical_mapping.values()
    )
    == {
        0,
        1,
    }
):

    label_candidates.append({
        "source":
            "CANONICAL_DFDC_REAL_DFDC_FAKE_PATH_ENCODING",

        "kind":
            "canonical_layout",

        "priority":
            100,

        "data":
            b"",

        "mapping":
            canonical_mapping,

        "coverage":
            1.0,

        "matched":
            len(
                dfdc_frame_ids
            ),

        "labels":
            len(
                canonical_mapping
            ),

        "missing":
            [],
    })


###############################################################################
# Select DFDC label source
###############################################################################

label_candidates.sort(
    key=lambda x: (
        x["coverage"],
        x["priority"],
        x["matched"],
    ),
    reverse=True,
)


label_report = []

for c in label_candidates:

    label_report.append({
        "source":
            c["source"],

        "kind":
            c["kind"],

        "priority":
            c["priority"],

        "coverage":
            c["coverage"],

        "matched":
            c["matched"],

        "labels":
            c["labels"],

        "missing_examples":
            c["missing"][:30],
    })


(
    AUDIT
    / "DFDC_LABEL_CANDIDATES.json"
).write_text(
    json.dumps(
        label_report,
        indent=2,
    )
)


full_label_candidates = [
    c
    for c in label_candidates
    if c["coverage"] == 1.0
]


if not full_label_candidates:

    die(
        "No scientifically defensible "
        "DFDC label source covers 100% "
        "of extracted evaluation videos. "
        f"Audit={AUDIT/'DFDC_LABEL_CANDIDATES.json'}"
    )


best_labels = (
    full_label_candidates[0]
)


# All full-coverage sources must agree on
# every actually evaluated video.
for other in (
    full_label_candidates[1:]
):

    conflicts = [
        video_id
        for video_id
        in dfdc_frame_ids
        if (
            best_labels[
                "mapping"
            ][video_id]
            != other[
                "mapping"
            ][video_id]
        )
    ]

    if conflicts:

        die(
            "Two full-coverage DFDC "
            "label sources disagree. "
            f"A={best_labels['source']} "
            f"B={other['source']} "
            f"examples={conflicts[:30]}"
        )


###############################################################################
# Build canonical DFDC test layout
###############################################################################

dfdc_shadow_test = (
    SHADOW
    / "DFDC"
    / "test"
)

dfdc_shadow_frames = (
    dfdc_shadow_test
    / "frames"
)

dfdc_shadow_frames.mkdir(
    parents=True,
    exist_ok=True,
)


for video_id in sorted(
    dfdc_frame_ids
):

    item = (
        dfdc_catalog[
            video_id
        ][0]
    )

    resolution = {
        "status":
            "RESOLVED",

        "source_type":
            "EXTRACTED",

        "reason":
            "UNIQUE_DFDC_VIDEO_ID",

        "path":
            item["path"],

        "frames":
            item["count"],
    }

    add_audit(
        "DFDC",
        video_id,
        video_id,
        resolution,
    )

    safe_symlink(
        item["path"],
        dfdc_shadow_frames
        / video_id,
    )


dfdc_labels_csv = (
    dfdc_shadow_test
    / "labels.csv"
)


with dfdc_labels_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "label",
        ],
    )

    writer.writeheader()

    for video_id in sorted(
        dfdc_frame_ids
    ):

        writer.writerow({
            "filename":
                f"{video_id}.mp4",

            "label":
                int(
                    best_labels[
                        "mapping"
                    ][video_id]
                ),
        })


print(
    "DFDC evaluated videos =",
    len(
        dfdc_frame_ids
    )
)

print(
    "DFDC label source =",
    best_labels[
        "source"
    ]
)

print(
    "DFDC label coverage = 100%"
)


###############################################################################
# WRITE RESOLUTION AUDIT
###############################################################################

resolution_csv = (
    AUDIT
    / "VIDEO_RESOLUTION.csv"
)

fieldnames = [
    "dataset",
    "metadata_key",
    "video_id",
    "status",
    "source_type",
    "reason",
    "resolved_path",
    "frames",
    "zip_parent",
]


with resolution_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        audit_rows
    )


###############################################################################
# GLOBAL FRAME VALIDATION
###############################################################################

bad_counts = [
    r
    for r in audit_rows
    if (
        r["status"]
        != "RESOLVED"
        or int(
            r["frames"]
        )
        <= 0
    )
]


if bad_counts:

    die(
        f"Global unresolved/zero-frame "
        f"videos remain: "
        f"{len(bad_counts)}. "
        f"Audit={resolution_csv}"
    )


# Existing X2DFD/DeepfakeBench preprocessing observed
# in this project is bounded at 32 frames/video.
# We fail rather than silently downsample an unexpected layout.
over32 = [
    r
    for r in audit_rows
    if int(
        r["frames"]
    ) > 32
]


if over32:

    path = (
        AUDIT
        / "OVER_32_FRAMES.json"
    )

    path.write_text(
        json.dumps(
            over32[:500],
            indent=2,
        )
    )

    die(
        f"Videos with >32 preprocessed "
        f"PNG frames detected: "
        f"{len(over32)}. "
        f"Refusing to resample. "
        f"Audit={path}"
    )


###############################################################################
# FINAL CANONICAL LAYOUT VALIDATION
###############################################################################

# Celeb
for r in [
    x
    for x in audit_rows
    if x["dataset"]
    == "Celeb-DF-v2"
]:

    metadata_path = Path(
        r["metadata_key"]
    )

    category = (
        metadata_path
        .parent
        .name
    )

    target = (
        SHADOW
        / "Celeb-DF-v2"
        / category
        / "frames"
        / r["video_id"]
    )

    if count_png(target) <= 0:
        die(
            f"Broken canonical Celeb link: "
            f"{target}"
        )


# DFDCP metadata-to-shadow check
for metadata_key, info in (
    dfdcp_json.items()
):

    if not isinstance(
        info,
        dict,
    ):
        continue

    if str(
        info.get(
            "set",
            "",
        )
    ).lower() != "test":
        continue

    parts = Path(
        metadata_key
    ).parts

    index = (
        parts[0]
        if len(parts) > 1
        else "0"
    )

    video_id = (
        Path(
            parts[-1]
        ).stem
    )

    target = (
        SHADOW
        / "DFDCP"
        / str(index)
        / "frames"
        / video_id
    )

    if count_png(target) <= 0:
        die(
            f"Broken canonical DFDCP link: "
            f"{target}"
        )


# DFDC label/frame exact ID equality
with dfdc_labels_csv.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    rows = list(
        csv.DictReader(f)
    )


label_ids = {
    Path(
        row["filename"]
    ).stem.lower()
    for row in rows
}


shadow_frame_ids = {
    p.name.lower()
    for p in (
        dfdc_shadow_frames
    ).iterdir()
    if p.is_dir()
}


if label_ids != shadow_frame_ids:

    die(
        "DFDC canonical label/frame ID "
        "sets differ."
    )


###############################################################################
# SUMMARY
###############################################################################

dataset_counts = Counter(
    r["dataset"]
    for r in audit_rows
)

recovered_counts = Counter(
    r["dataset"]
    for r in audit_rows
    if r["source_type"]
    == "RECOVERED_FROM_ZIP"
)


frame_distributions = {}


for dataset in [
    "Celeb-DF-v2",
    "DFDCP",
    "DFDC",
]:

    counts = [
        int(
            r["frames"]
        )
        for r in audit_rows
        if r["dataset"]
        == dataset
    ]

    frame_distributions[
        dataset
    ] = {
        "videos":
            len(counts),

        "min_frames":
            min(counts)
            if counts
            else None,

        "max_frames":
            max(counts)
            if counts
            else None,

        "distribution":
            dict(
                sorted(
                    Counter(
                        counts
                    ).items()
                )
            ),
    }


summary = {
    "status":
        "PASS",

    "datasets":
        dict(
            dataset_counts
        ),

    "recovered_from_existing_zip_pngs":
        dict(
            recovered_counts
        ),

    "frame_distributions":
        frame_distributions,

    "metadata": {
        "Celeb-DF-v2": {
            "selected":
                celeb_meta[
                    "source"
                ],

            "sha256":
                celeb_meta[
                    "sha256"
                ],

            "candidates":
                celeb_meta_all,
        },

        "DFDCP": {
            "selected":
                dfdcp_meta[
                    "source"
                ],

            "sha256":
                dfdcp_meta[
                    "sha256"
                ],

            "candidates":
                dfdcp_meta_all,
        },

        "DFDC": {
            "selected":
                best_labels[
                    "source"
                ],

            "kind":
                best_labels[
                    "kind"
                ],

            "coverage":
                best_labels[
                    "coverage"
                ],
        },
    },

    "guards": {
        "middle_frame_only":
            False,

        "random_sampling":
            False,

        "frame_duplication":
            False,

        "ad_hoc_face_detection":
            False,

        "generic_real_fake_folder_guess":
            False,

        "silent_missing_video_skip":
            False,

        "ambiguous_video_mapping_allowed":
            False,

        "DFDC_label_coverage":
            1.0,

        "both_models_use_same_shadow":
            True,
    },

    "resolution_csv":
        str(
            resolution_csv
        ),
}


summary_path = (
    AUDIT
    / "PREFLIGHT_PASS.json"
)


summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


print()
print("=" * 80)
print("✅ ALL THREE DATASETS RESOLVED")
print("=" * 80)

print(
    json.dumps(
        summary,
        indent=2,
    )
)


from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile


BASE, DATA, RAW, DL, SHADOW, AUDIT = map(Path, sys.argv[1:])

SHADOW.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# BASIC HELPERS
# ============================================================================

def fail(msg):
    raise RuntimeError(msg)


def sha_bytes(x):
    return hashlib.sha256(x).hexdigest()


def sha_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def pngs(folder):
    folder = Path(folder)

    if not folder.is_dir():
        return []

    return sorted(
        p.resolve()
        for p in folder.glob("*.png")
        if p.is_file()
    )


def link_dir(src, dst):
    src = Path(src).resolve()
    dst = Path(dst)

    if not src.is_dir():
        fail(f"Symlink source missing: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink() or dst.is_file():
        dst.unlink()

    elif dst.exists():
        shutil.rmtree(dst)

    dst.symlink_to(src, target_is_directory=True)


# ============================================================================
# METADATA RESOLUTION
#
# Exact known filename only.
# No guessing labels from folder names.
# ============================================================================

def local_exact(roots, filename, tokens=()):
    out = []

    for root in roots:
        root = Path(root)

        if not root.exists():
            continue

        for p in root.rglob(filename):

            if not p.is_file():
                continue

            low = str(p).lower()

            if tokens and not any(
                token.lower() in low
                for token in tokens
            ):
                continue

            out.append(p.resolve())

    return sorted(set(out))


def zip_exact(zip_path, filename):
    out = []
    zip_path = Path(zip_path)

    if not zip_path.is_file():
        return out

    with zipfile.ZipFile(zip_path) as z:

        for info in z.infolist():

            if info.is_dir():
                continue

            if PurePosixPath(info.filename).name.lower() != filename.lower():
                continue

            out.append(
                (
                    info.filename,
                    z.read(info.filename),
                )
            )

    return out


def resolve_exact_metadata(
    dataset,
    filename,
    extracted_root,
    zip_path,
    download_tokens,
):
    # priority 1: extracted package
    hits = local_exact(
        [extracted_root],
        filename,
    )

    if hits:

        hashes = {
            sha_file(p)
            for p in hits
        }

        if len(hashes) != 1:
            fail(
                f"Conflicting extracted {dataset}/{filename}: {hits}"
            )

        p = hits[0]
        b = p.read_bytes()

        return b, {
            "source": str(p),
            "method": "EXTRACTED_PACKAGE",
            "sha256": sha_bytes(b),
        }

    # priority 2: official/raw zip
    hits = zip_exact(
        zip_path,
        filename,
    )

    if hits:

        hashes = {
            sha_bytes(b)
            for _, b in hits
        }

        if len(hashes) != 1:
            fail(
                f"Conflicting {filename} inside {zip_path}"
            )

        member, b = hits[0]

        return b, {
            "source": f"{zip_path}::{member}",
            "method": "RAW_ZIP",
            "sha256": sha_bytes(b),
        }

    # priority 3: teacher/X2DFD download tree
    hits = local_exact(
        [DL],
        filename,
        download_tokens,
    )

    if hits:

        hashes = {
            sha_file(p)
            for p in hits
        }

        if len(hashes) != 1:
            fail(
                f"Conflicting downloaded {dataset}/{filename}: {hits}"
            )

        p = hits[0]
        b = p.read_bytes()

        return b, {
            "source": str(p),
            "method": "DOWNLOAD_TREE",
            "sha256": sha_bytes(b),
        }

    fail(
        f"Required metadata unresolved: {dataset}/{filename}"
    )


# ============================================================================
# CELEB-DF-v2
#
# DeepfakeBench special test list.
# In our local copy screenshot/audit already showed 518/518 available.
# ============================================================================

print()
print("=" * 80)
print("CELEB-DF-v2")
print("=" * 80)

CDF_SRC = DATA / "Celeb-DF-v2"

cdf_bytes, cdf_prov = resolve_exact_metadata(
    "Celeb-DF-v2",
    "List_of_testing_videos.txt",
    CDF_SRC,
    RAW / "Celeb-DF-v2.zip",
    ["celeb", "cdf"],
)

CDF_DST = SHADOW / "Celeb-DF-v2"
CDF_DST.mkdir(parents=True, exist_ok=True)

(CDF_DST / "List_of_testing_videos.txt").write_bytes(
    cdf_bytes
)

cdf_valid = []
cdf_missing = []

for line in cdf_bytes.decode(
    "utf-8",
    errors="ignore",
).splitlines():

    line = line.strip()

    if not line:
        continue

    tokens = line.split()

    mp4 = next(
        (
            x
            for x in tokens
            if x.lower().endswith(".mp4")
        ),
        None,
    )

    if mp4 is None:
        continue

    rel = Path(mp4)
    cls = rel.parent.name
    vid = rel.stem

    source_frames = (
        CDF_SRC
        / cls
        / "frames"
        / vid
    )

    frames = pngs(source_frames)

    if not frames:

        cdf_missing.append({
            "metadata": mp4,
            "expected": str(source_frames),
        })
        continue

    link_dir(
        source_frames,
        CDF_DST
        / cls
        / "frames"
        / vid,
    )

    cdf_valid.append({
        "video_id": vid,
        "class_dir": cls,
        "frames": len(frames),
    })


if cdf_missing:
    fail(
        "Celeb test-list entries without frames found. "
        f"Count={len(cdf_missing)}. "
        "Unlike DFDCP, we are not silently changing the known 518-video "
        f"Celeb test protocol. Examples={cdf_missing[:10]}"
    )

if not cdf_valid:
    fail("Celeb frozen test set is empty.")

print("official test videos =", len(cdf_valid))
print("missing =", len(cdf_missing))


# ============================================================================
# DFDCP — EXACT DEEPFAKEBENCH REARRANGE SEMANTICS
#
# Upstream:
#
# index   = dataset.split('/')[0]
# vidname = dataset.split('/')[-1].split('.')[0]
#
# if index/frames/vidname exists:
#     frame_paths = *.png
#     if len(frame_paths) == 0:
#         continue
#     ...
#
# Therefore metadata rows without preprocessed frames are NOT members of the
# benchmark's rearranged test set.
# ============================================================================

print()
print("=" * 80)
print("DFDCP — UPSTREAM-COMPATIBLE FILTER")
print("=" * 80)

DFDCP_SRC = DATA / "DFDCP"

dfdcp_bytes, dfdcp_prov = resolve_exact_metadata(
    "DFDCP",
    "dataset.json",
    DFDCP_SRC,
    RAW / "DFDCP.zip",
    ["dfdcp"],
)

try:
    dfdcp_raw = json.loads(
        dfdcp_bytes.decode(
            "utf-8",
            errors="ignore",
        )
    )

except Exception as e:
    fail(f"Invalid DFDCP dataset.json: {e}")


if not isinstance(dfdcp_raw, dict):
    fail("DFDCP dataset.json top-level is not dict.")


DFDCP_DST = SHADOW / "DFDCP"
DFDCP_DST.mkdir(parents=True, exist_ok=True)


filtered_dfdcp = {}

metadata_test = 0
valid_test = 0
skipped_missing_dir = []
skipped_empty_dir = []
invalid_label = []


for key, info in dfdcp_raw.items():

    if not isinstance(info, dict):
        continue

    if str(
        info.get("set", "")
    ).strip().lower() != "test":
        continue

    metadata_test += 1

    parts = str(key).split("/")

    if len(parts) < 2:
        fail(f"Malformed DFDCP metadata key: {key}")

    index = parts[0]

    vidname = (
        parts[-1]
        .split(".")[0]
    )

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
        invalid_label.append(
            {
                "key": key,
                "label": label,
            }
        )
        continue

    frame_dir = (
        DFDCP_SRC
        / index
        / "frames"
        / vidname
    )

    if not frame_dir.is_dir():

        skipped_missing_dir.append({
            "key": key,
            "expected": str(frame_dir),
        })

        # Exact upstream semantics: skip.
        continue

    frame_paths = pngs(
        frame_dir
    )

    if not frame_paths:

        skipped_empty_dir.append({
            "key": key,
            "expected": str(frame_dir),
        })

        # Exact upstream semantics: skip.
        continue

    # Only now does it enter the rearranged benchmark.
    valid_test += 1

    filtered_dfdcp[
        key
    ] = info

    link_dir(
        frame_dir,
        DFDCP_DST
        / index
        / "frames"
        / vidname,
    )


if invalid_label:
    fail(
        f"DFDCP contains invalid test labels: {invalid_label[:20]}"
    )

if valid_test == 0:
    fail("DFDCP filtered test set is empty.")


# Crucial scientific invariant.
if (
    valid_test
    + len(skipped_missing_dir)
    + len(skipped_empty_dir)
    != metadata_test
):

    fail(
        "DFDCP accounting invariant failed."
    )


(DFDCP_DST / "dataset.json").write_text(
    json.dumps(
        filtered_dfdcp,
        indent=2,
    ),
    encoding="utf-8",
)


(AUDIT / "DFDCP_UPSTREAM_SKIPPED.json").write_text(
    json.dumps(
        {
            "metadata_test_rows":
                metadata_test,

            "benchmark_valid_test_videos":
                valid_test,

            "skipped_missing_frame_directory":
                skipped_missing_dir,

            "skipped_empty_frame_directory":
                skipped_empty_dir,

            "rule":
                "Matches DeepfakeBench rearrange.py: "
                "only videos with existing non-empty preprocessed "
                "frame directory enter the rearranged dataset.",
        },
        indent=2,
    ),
    encoding="utf-8",
)


print("metadata test rows =", metadata_test)
print("valid benchmark videos =", valid_test)
print(
    "skipped missing dir =",
    len(skipped_missing_dir),
)
print(
    "skipped empty dir =",
    len(skipped_empty_dir),
)


# ============================================================================
# DFDC METADATA DISCOVERY
#
# DeepfakeBench upstream definition:
#
#   test/labels.csv
#   label 0 = REAL
#   label 1 = FAKE
#   test/frames/<video>/*.png
#   zero-frame rows -> continue
#
# We first look for labels.csv.
#
# If teacher package instead contains a rearranged DeepfakeBench JSON, we may
# recover the SAME test labels from DFDC_Real/test and DFDC_Fake/test branches.
# That is metadata provenance, not inference from image folder names.
# ============================================================================

print()
print("=" * 80)
print("DFDC — UPSTREAM-COMPATIBLE FILTER")
print("=" * 80)

DFDC_SRC = DATA / "DFDC"


def normalize_dfdc_label(x):

    s = str(x).strip().lower()

    if s in {
        "0",
        "0.0",
        "real",
        "dfdc_real",
    }:
        return 0

    if s in {
        "1",
        "1.0",
        "fake",
        "dfdc_fake",
    }:
        return 1

    return None


def parse_labels_csv(data):

    reader = csv.DictReader(
        io.StringIO(
            data.decode(
                "utf-8-sig",
                errors="ignore",
            )
        )
    )

    fields = reader.fieldnames or []

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

    if file_col is None or label_col is None:
        return None

    mapping = {}

    for row in reader:

        filename = str(
            row.get(
                file_col,
                "",
            )
        ).strip()

        label = normalize_dfdc_label(
            row.get(
                label_col,
                "",
            )
        )

        if not filename or label is None:
            continue

        vid = Path(filename).stem

        old = mapping.get(vid)

        if old is not None and old != label:
            return None

        mapping[vid] = label

    return mapping or None


def parse_deepfakebench_json(data):

    try:
        obj = json.loads(
            data.decode(
                "utf-8-sig",
                errors="ignore",
            )
        )

    except Exception:
        return None

    candidates = []


    def recurse(node):

        if not isinstance(node, dict):
            return

        lower = {
            str(k).lower(): k
            for k in node
        }

        real_key = lower.get(
            "dfdc_real"
        )

        fake_key = lower.get(
            "dfdc_fake"
        )

        if (
            real_key is not None
            or fake_key is not None
        ):

            mapping = {}

            for key, label in [
                (real_key, 0),
                (fake_key, 1),
            ]:

                if key is None:
                    continue

                branch = node.get(
                    key
                )

                if not isinstance(
                    branch,
                    dict,
                ):
                    continue

                test = branch.get(
                    "test"
                )

                if not isinstance(
                    test,
                    dict,
                ):
                    continue

                for vid in test:

                    name = Path(
                        str(vid)
                    ).stem

                    old = mapping.get(
                        name
                    )

                    if (
                        old is not None
                        and old != label
                    ):
                        mapping = {}
                        break

                    mapping[
                        name
                    ] = label

            if mapping:
                candidates.append(
                    mapping
                )

        for value in node.values():

            if isinstance(
                value,
                dict,
            ):
                recurse(value)

            elif isinstance(
                value,
                list,
            ):

                for x in value:
                    recurse(x)


    recurse(obj)

    if not candidates:
        return None

    candidates.sort(
        key=len,
        reverse=True,
    )

    return candidates[0]


def add_candidate(
    candidates,
    source,
    kind,
    data,
    priority,
):

    if kind == "csv":
        mapping = parse_labels_csv(
            data
        )

    else:
        mapping = parse_deepfakebench_json(
            data
        )

    if mapping:

        candidates.append({
            "source":
                source,

            "kind":
                kind,

            "data":
                data,

            "mapping":
                mapping,

            "priority":
                priority,
        })


dfdc_candidates = []


# --------------------------------------------------------------------------
# 1. Exact labels.csv anywhere in extracted DFDC.
# --------------------------------------------------------------------------

for p in DFDC_SRC.rglob(
    "labels.csv"
):

    if p.is_file():

        add_candidate(
            dfdc_candidates,
            str(
                p.resolve()
            ),
            "csv",
            p.read_bytes(),
            1000,
        )


# --------------------------------------------------------------------------
# 2. Exact labels.csv in raw zip.
# --------------------------------------------------------------------------

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

            name = info.filename
            base = (
                PurePosixPath(
                    name
                ).name.lower()
            )

            if base == "labels.csv":

                add_candidate(
                    dfdc_candidates,
                    f"{dfdc_zip}::{name}",
                    "csv",
                    z.read(
                        name
                    ),
                    950,
                )


# --------------------------------------------------------------------------
# 3. Download tree labels.csv.
# --------------------------------------------------------------------------

if DL.exists():

    for p in DL.rglob(
        "labels.csv"
    ):

        if (
            p.is_file()
            and "dfdc"
            in str(p).lower()
        ):

            add_candidate(
                dfdc_candidates,
                str(
                    p.resolve()
                ),
                "csv",
                p.read_bytes(),
                900,
            )


# --------------------------------------------------------------------------
# 4. Rearranged DeepfakeBench JSON sources.
#    Scan only reasonably-sized JSONs.
# --------------------------------------------------------------------------

json_roots = [
    DFDC_SRC,
    DL,
    BASE
    / "projects"
    / "X2DFD",
    BASE
    / "outputs"
    / "x2dfd_router_integration",
]


for root in json_roots:

    if not root.exists():
        continue

    for p in root.rglob(
        "*.json"
    ):

        try:

            if (
                p.stat().st_size
                <= 0
                or p.stat().st_size
                > 256
                * 1024
                * 1024
            ):
                continue

            add_candidate(
                dfdc_candidates,
                str(
                    p.resolve()
                ),
                "json",
                p.read_bytes(),
                500,
            )

        except Exception:
            continue


if dfdc_zip.is_file():

    with zipfile.ZipFile(
        dfdc_zip
    ) as z:

        for info in z.infolist():

            if (
                info.is_dir()
                or not info.filename.lower()
                .endswith(".json")
                or info.file_size <= 0
                or info.file_size
                > 256
                * 1024
                * 1024
            ):
                continue

            try:

                add_candidate(
                    dfdc_candidates,
                    f"{dfdc_zip}::{info.filename}",
                    "json",
                    z.read(
                        info.filename
                    ),
                    550,
                )

            except Exception:
                continue


if not dfdc_candidates:

    fail(
        "Could not resolve DFDC test labels from either "
        "labels.csv or a DeepfakeBench-style rearranged "
        "DFDC_Real/DFDC_Fake test JSON. "
        "Full benchmark correctly blocked."
    )


# ============================================================================
# Build catalog of actual preprocessed DFDC frame directories.
# ============================================================================

frame_dirs = defaultdict(
    list
)


for current, dirs, files in os.walk(
    DFDC_SRC
):

    if not any(
        x.lower().endswith(
            ".png"
        )
        for x in files
    ):
        continue

    p = Path(
        current
    ).resolve()

    frame_dirs[
        p.name
    ].append(p)


def resolve_dfdc_frame_dir(
    vid
):

    # exact official path first
    preferred = (
        DFDC_SRC
        / "test"
        / "frames"
        / vid
    )

    if pngs(preferred):
        return preferred.resolve()

    matches = [
        p
        for p in frame_dirs.get(
            vid,
            []
        )
        if pngs(p)
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        fail(
            f"Ambiguous DFDC frame directories "
            f"for video={vid}: {matches}"
        )

    return None


# Score metadata candidates by how many metadata test videos have frames.
for c in dfdc_candidates:

    valid = []
    skipped = []

    for vid, label in (
        c["mapping"]
        .items()
    ):

        folder = (
            resolve_dfdc_frame_dir(
                vid
            )
        )

        if folder is None:

            skipped.append(
                vid
            )

        else:

            valid.append(
                (
                    vid,
                    label,
                    folder,
                )
            )

    c["valid"] = valid
    c["skipped"] = skipped
    c["valid_count"] = len(
        valid
    )


dfdc_candidates.sort(
    key=lambda c: (
        c[
            "valid_count"
        ],
        c[
            "priority"
        ],
    ),
    reverse=True,
)


candidate_report = [
    {
        "source":
            c["source"],

        "kind":
            c["kind"],

        "metadata_labels":
            len(
                c["mapping"]
            ),

        "valid_preprocessed_test_videos":
            c["valid_count"],

        "skipped_without_frames":
            len(
                c["skipped"]
            ),

        "priority":
            c["priority"],
    }
    for c in dfdc_candidates
]


(AUDIT / "DFDC_LABEL_SOURCE_CANDIDATES.json").write_text(
    json.dumps(
        candidate_report,
        indent=2,
    ),
    encoding="utf-8",
)


best_dfdc = (
    dfdc_candidates[0]
)


if (
    best_dfdc[
        "valid_count"
    ]
    == 0
):

    fail(
        "Resolved DFDC label metadata, "
        "but zero labelled videos have "
        "preprocessed frames."
    )


# Require any other candidate covering the same evaluated videos
# to agree on those labels.
best_eval = {
    vid: label
    for vid, label, _
    in best_dfdc[
        "valid"
    ]
}


for other in (
    dfdc_candidates[1:]
):

    overlap = (
        set(best_eval)
        & set(
            other[
                "mapping"
            ]
        )
    )

    conflicts = [
        vid
        for vid in overlap
        if (
            best_eval[
                vid
            ]
            != other[
                "mapping"
            ][vid]
        )
    ]

    if conflicts:

        fail(
            "DFDC metadata sources disagree "
            f"on evaluated videos. "
            f"A={best_dfdc['source']} "
            f"B={other['source']} "
            f"examples={conflicts[:20]}"
        )


DFDC_DST = (
    SHADOW
    / "DFDC"
    / "test"
)

(
    DFDC_DST
    / "frames"
).mkdir(
    parents=True,
    exist_ok=True,
)


with (
    DFDC_DST
    / "labels.csv"
).open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "label",
        ],
    )

    writer.writeheader()

    for (
        vid,
        label,
        folder,
    ) in sorted(
        best_dfdc[
            "valid"
        ],
        key=lambda x: x[0],
    ):

        writer.writerow({
            "filename":
                f"{vid}.mp4",

            "label":
                int(label),
        })

        link_dir(
            folder,
            DFDC_DST
            / "frames"
            / vid,
        )


(AUDIT / "DFDC_UPSTREAM_SKIPPED.json").write_text(
    json.dumps(
        {
            "selected_label_source":
                best_dfdc[
                    "source"
                ],

            "metadata_test_labels":
                len(
                    best_dfdc[
                        "mapping"
                    ]
                ),

            "benchmark_valid_test_videos":
                best_dfdc[
                    "valid_count"
                ],

            "skipped_without_preprocessed_frames":
                best_dfdc[
                    "skipped"
                ],

            "rule":
                "Matches DeepfakeBench DFDC rearrange logic: "
                "metadata-labelled videos with zero preprocessed "
                "frames are skipped.",
        },
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "selected metadata source =",
    best_dfdc[
        "source"
    ]
)

print(
    "valid benchmark videos =",
    best_dfdc[
        "valid_count"
    ]
)

print(
    "skipped without frames =",
    len(
        best_dfdc[
            "skipped"
        ]
    ),
)


# ============================================================================
# INDEPENDENT CANONICAL VALIDATION
# ============================================================================

print()
print("=" * 80)
print("INDEPENDENT CANONICAL VALIDATION")
print("=" * 80)


# DFDCP filtered JSON must map 1:1 to valid shadow frame dirs.
check_dfdcp = json.loads(
    (
        DFDCP_DST
        / "dataset.json"
    ).read_text()
)


for key, info in (
    check_dfdcp.items()
):

    if str(
        info.get(
            "set",
            "",
        )
    ).lower() != "test":
        fail(
            "Filtered DFDCP dataset.json "
            "contains non-test entry."
        )

    parts = key.split("/")

    index = parts[0]
    vid = parts[-1].split(".")[0]

    f = (
        DFDCP_DST
        / index
        / "frames"
        / vid
    )

    if not pngs(f):
        fail(
            f"Broken DFDCP shadow path: {f}"
        )


if len(
    check_dfdcp
) != valid_test:

    fail(
        "DFDCP filtered metadata count "
        "!= resolved video count."
    )


# DFDC labels must map exactly to shadow frame dirs.
with (
    DFDC_DST
    / "labels.csv"
).open(
    newline="",
    encoding="utf-8",
) as f:

    rows = list(
        csv.DictReader(f)
    )


dfdc_label_ids = {
    Path(
        r[
            "filename"
        ]
    ).stem
    for r in rows
}


dfdc_shadow_ids = {
    p.name
    for p in (
        DFDC_DST
        / "frames"
    ).iterdir()
    if p.is_dir()
}


if (
    dfdc_label_ids
    != dfdc_shadow_ids
):

    fail(
        "DFDC labels.csv and frozen frame "
        "directory IDs differ."
    )


if {
    int(
        r["label"]
    )
    for r in rows
} != {
    0,
    1,
}:

    fail(
        "DFDC frozen benchmark does not "
        "contain both classes."
    )


# ============================================================================
# FREEZE FRAME COUNTS + MANIFEST PROTOCOL
# ============================================================================

def collect_counts():

    result = {}

    # Celeb
    counts = []

    for r in cdf_valid:

        d = (
            CDF_DST
            / r[
                "class_dir"
            ]
            / "frames"
            / r[
                "video_id"
            ]
        )

        counts.append(
            len(
                pngs(d)
            )
        )

    result[
        "Celeb-DF-v2"
    ] = {
        "videos":
            len(counts),

        "frames":
            sum(counts),

        "min_frames_per_video":
            min(counts),

        "max_frames_per_video":
            max(counts),

        "frame_count_distribution":
            dict(
                sorted(
                    Counter(
                        counts
                    ).items()
                )
            ),
    }

    # DFDCP
    counts = []

    for key in check_dfdcp:

        parts = key.split("/")

        d = (
            DFDCP_DST
            / parts[0]
            / "frames"
            / parts[-1]
            .split(".")[0]
        )

        counts.append(
            len(
                pngs(d)
            )
        )

    result[
        "DFDCP"
    ] = {
        "metadata_test_rows":
            metadata_test,

        "videos_evaluated":
            len(counts),

        "videos_skipped_by_upstream_frame_presence_rule":
            metadata_test
            - len(counts),

        "frames":
            sum(counts),

        "min_frames_per_video":
            min(counts),

        "max_frames_per_video":
            max(counts),

        "frame_count_distribution":
            dict(
                sorted(
                    Counter(
                        counts
                    ).items()
                )
            ),
    }

    # DFDC
    counts = [
        len(
            pngs(
                DFDC_DST
                / "frames"
                / vid
            )
        )
        for vid in sorted(
            dfdc_shadow_ids
        )
    ]

    result[
        "DFDC"
    ] = {
        "metadata_source":
            best_dfdc[
                "source"
            ],

        "metadata_labelled_videos":
            len(
                best_dfdc[
                    "mapping"
                ]
            ),

        "videos_evaluated":
            len(counts),

        "videos_skipped_by_upstream_frame_presence_rule":
            len(
                best_dfdc[
                    "mapping"
                ]
            )
            - len(counts),

        "frames":
            sum(counts),

        "min_frames_per_video":
            min(counts),

        "max_frames_per_video":
            max(counts),

        "frame_count_distribution":
            dict(
                sorted(
                    Counter(
                        counts
                    ).items()
                )
            ),
    }

    return result


dataset_stats = (
    collect_counts()
)


protocol = {
    "status":
        "PASS",

    "protocol_name":
        "DeepfakeBench-compatible preprocessed-test-set freeze",

    "core_rule":
        "A metadata test video enters evaluation only if its "
        "preprocessed PNG frame directory exists and contains >=1 frame.",

    "DFDCP_rule_source":
        "DeepfakeBench rearrange.py",

    "DFDC_rule_source":
        "DeepfakeBench rearrange.py",

    "sampling": {
        "middle_frame_only":
            False,

        "random_frame_selection":
            False,

        "synthetic_frame_duplication":
            False,

        "new_face_detection":
            False,

        "new_video_decoding":
            False,

        "use_existing_preprocessed_frames":
            True,
    },

    "fairness": {
        "original_X2DFD_and_Router4_receive_identical_frozen_manifest":
            True,

        "same_frames":
            True,

        "same_labels":
            True,
    },

    "dataset_stats":
        dataset_stats,

    "metadata_provenance": {
        "Celeb-DF-v2":
            cdf_prov,

        "DFDCP":
            dfdcp_prov,

        "DFDC": {
            "source":
                best_dfdc[
                    "source"
                ],

            "kind":
                best_dfdc[
                    "kind"
                ],
        },
    },

    "audit_files": {
        "DFDCP_skipped":
            str(
                AUDIT
                / "DFDCP_UPSTREAM_SKIPPED.json"
            ),

        "DFDC_skipped":
            str(
                AUDIT
                / "DFDC_UPSTREAM_SKIPPED.json"
            ),

        "DFDC_label_candidates":
            str(
                AUDIT
                / "DFDC_LABEL_SOURCE_CANDIDATES.json"
            ),
    },
}


(AUDIT / "PAPER_PROTOCOL_LOCK.json").write_text(
    json.dumps(
        protocol,
        indent=2,
    ),
    encoding="utf-8",
)


(AUDIT / "PAPER_METHOD_NOTE.md").write_text(
    f"""# Frozen cross-dataset evaluation protocol

The evaluation set was constructed from the official/local benchmark metadata
and the already-preprocessed face-frame directories.

No middle-frame-only sampling, random sampling, frame duplication, new video
decoding, or new face detection was performed.

For DFDCP, the evaluation follows the DeepfakeBench rearrangement rule: a
metadata entry is included only when its expected preprocessed frame directory
exists and contains at least one PNG frame. Therefore:

- metadata test rows: {metadata_test}
- evaluated videos: {valid_test}
- skipped because no usable preprocessed frames:
  {metadata_test - valid_test}

For DFDC, the same frame-presence rule was applied after resolving the test
ground-truth metadata. Evaluated videos:
{best_dfdc['valid_count']}.

Original X2DFD and Router4 use the identical frozen image manifest. Frame-level
scores and video-level aggregated scores must be reported separately. AUC must
be computed from continuous REAL/FAKE token scores, not parsed hard text labels.
""",
    encoding="utf-8",
)


print()
print(
    json.dumps(
        protocol,
        indent=2,
    )
)

print()
print(
    "✅✅✅ ALL THREE DATASETS "
    "PROTOCOL-LOCKED ✅✅✅"
)


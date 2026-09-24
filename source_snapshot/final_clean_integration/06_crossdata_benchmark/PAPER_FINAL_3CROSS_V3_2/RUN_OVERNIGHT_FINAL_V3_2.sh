#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# GLOBAL PATHS
###############################################################################

BASE=/home/aiotlab/hoangan
ROOT="$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

RUNROOT="$ROOT/06_crossdata_benchmark/PAPER_FINAL_3CROSS_V3_2"

WORK="$RUNROOT/work"
PROTO="$RUNROOT/protocol"
LOGS="$RUNROOT/logs"
CONFIGS="$RUNROOT/configs"
SMOKE="$RUNROOT/smoke"
SNAP="$RUNROOT/snapshots"

DATA="$BASE/datasets/CROSSDATA_FINAL_EVAL"

X2="$BASE/projects/X2DFD"
INT="$BASE/projects/x2dfd_router_integration"

PY_X2=/home/aiotlab/miniconda3/envs/X2DFD/bin/python
PY_FREQ=/home/aiotlab/miniconda3/envs/DFFreq/bin/python
PY_TEX=/home/aiotlab/miniconda3/envs/TextureExpert/bin/python

RUN_X2="$BASE/router8_x2crop/run_x2.py"
RUN_FREQ="$BASE/router8_x2crop/run_frequency.py"
RUN_TEX="$BASE/router8_x2crop/run_texture.py"

DFF="$BASE/projects/DFFreq-main"

TEXWORK="$BASE/projects/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild/stylegan-ffhq"

FINAL_ROUTER="$ROOT/02_router/final_checkpoint/best.pt"
FINAL_CAL="$ROOT/01_teacher/calibrators_FINAL_CLEAN.joblib"

TRAIN_TEACHER="$ROOT/01_teacher/train_teacher_FINAL_CLEAN.csv"
VAL_TEACHER="$ROOT/01_teacher/val_teacher_FINAL_CLEAN.csv"

FINAL_LORA="$ROOT/05_lora/router4_x2dfd_FINAL_CLEAN"

ORIG_LORA="$X2/weights/_official_extract/weight/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]"

BASE_MODEL="$X2/weights/base/llava-v1.5-7b"

CACHE_SRC="$INT/build_router_moe_cache_full.py"
MAKE_SRC="$INT/make_full_configs.py"

OLD_CONFIG_ORIG="$X2/eval/configs/infer_full_DFDCP_DFDC_original.yaml"
OLD_CONFIG_R4="$X2/eval/configs/infer_full_DFDCP_DFDC_router_moe.yaml"

STOP="$RUNROOT/STOP_REASON.txt"

mkdir -p \
 "$WORK" \
 "$PROTO" \
 "$LOGS" \
 "$CONFIGS" \
 "$SMOKE" \
 "$SNAP/original" \
 "$SNAP/router4" \
 "$SNAP/smoke_original" \
 "$SNAP/smoke_router4"

rm -f \
 "$STOP" \
 "$RUNROOT/INFERENCE_COMPLETE" \
 "$RUNROOT/PAPER_GATE_PASS"

###############################################################################
# HELPERS
###############################################################################

stage() {
    CURRENT_STAGE="$1"

    echo
    echo "================================================================================"
    echo "$CURRENT_STAGE"
    date
    echo "================================================================================"
}

die() {
    MSG="$*"

    echo
    echo "[STOP] $MSG"

    printf '%s\n' "$MSG" > "$STOP"

    exit 1
}

trap '
RC=$?

echo
echo "[ERROR]"
echo "stage=${CURRENT_STAGE:-BOOT}"
echo "line=$LINENO"
echo "exit=$RC"

{
    echo "stage=${CURRENT_STAGE:-BOOT}"
    echo "line=$LINENO"
    echo "exit=$RC"
} > "$STOP"

exit $RC
' ERR

###############################################################################
# A. HARD PREFLIGHT
###############################################################################

stage "A. HARD PREFLIGHT"

for P in \
 "$PY_X2" \
 "$PY_FREQ" \
 "$PY_TEX" \
 "$RUN_X2" \
 "$RUN_FREQ" \
 "$RUN_TEX" \
 "$FINAL_ROUTER" \
 "$FINAL_CAL" \
 "$TRAIN_TEACHER" \
 "$VAL_TEACHER" \
 "$FINAL_LORA/adapter_model.safetensors" \
 "$FINAL_LORA/adapter_config.json" \
 "$ORIG_LORA/adapter_model.safetensors" \
 "$ORIG_LORA/adapter_config.json" \
 "$BASE_MODEL/config.json" \
 "$CACHE_SRC" \
 "$MAKE_SRC" \
 "$X2/eval/infer/runner.py"
do
    [ -e "$P" ] || \
        die "Required artifact missing: $P"
done


for D in \
 "$DATA/Celeb-DF-v2" \
 "$DATA/DFDC" \
 "$DATA/DFDCP"
do
    [ -d "$D" ] || \
        die "Dataset directory missing: $D"
done


"$PY_X2" - <<'PY'
import torch
import yaml
from PIL import Image

print("torch =", torch.__version__)
print("cuda =", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

print("gpu =", torch.cuda.get_device_name(0))
print("PyYAML = OK")
print("Pillow = OK")
PY


nvidia-smi \
 --query-gpu=name,memory.total,memory.free \
 --format=csv,noheader


###############################################################################
# B. FREEZE SOFTWARE / MODEL PROVENANCE
###############################################################################

stage "B. FREEZE SOFTWARE + MODEL PROVENANCE"

{
    echo "DATE=$(date -Is)"
    echo "HOST=$(hostname)"
    echo
    echo "===== X2DFD ====="

    cd "$X2"

    git rev-parse HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true

    echo
    echo "===== INTEGRATION ====="

    cd "$INT"

    git rev-parse HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true

} > "$PROTO/git_state.txt"


sha256sum \
 "$FINAL_ROUTER" \
 "$FINAL_CAL" \
 "$TRAIN_TEACHER" \
 "$VAL_TEACHER" \
 "$FINAL_LORA/adapter_model.safetensors" \
 "$FINAL_LORA/adapter_config.json" \
 "$ORIG_LORA/adapter_model.safetensors" \
 "$ORIG_LORA/adapter_config.json" \
 "$RUN_X2" \
 "$RUN_FREQ" \
 "$RUN_TEX" \
 "$CACHE_SRC" \
 "$MAKE_SRC" \
 "$X2/eval/infer/runner.py" \
 > "$PROTO/BASE_ARTIFACTS.sha256"


###############################################################################
# C. BUILD PAPER-GRADE OFFICIAL TEST MANIFEST
#
# RULES:
# - Celeb-DF-v2: official List_of_testing_videos.txt
# - DFDCP: official dataset.json, set == test
# - DFDC: test/labels.csv, numeric 0=real, 1=fake
#
# - Only preprocessed *.png used because DeepfakeBench rearrange uses *png.
# - NO middle frame
# - NO random sampling
# - NO duplicate frames
# - NO re-extraction/resampling
# - >32 frames/video = hard stop (local dataset/protocol mismatch)
# - <32 allowed and recorded (preprocessing yield)
###############################################################################

stage "C. BUILD OFFICIAL FROZEN MANIFEST"

MANIFEST="$WORK/manifest.csv"
LABELS="$PROTO/labels_and_video_ids.csv"
LOCK="$PROTO/PROTOCOL_LOCK.json"

CELEB_JSON="$WORK/Celeb-DF-v2_test.json"
DFDCP_JSON="$WORK/DFDCP_test.json"
DFDC_JSON="$WORK/DFDC_test.json"


"$PY_X2" - \
 "$DATA" \
 "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_3CROSS_V3_2/schema_template" \
 "$MANIFEST" \
 "$LABELS" \
 "$LOCK" \
 "$CELEB_JSON" \
 "$DFDCP_JSON" \
 "$DFDC_JSON" \
 <<'PYMAN'

from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import sys
import copy

(
    DATA,
    TEMPLATE_ROOT,
    MANIFEST,
    LABELS,
    LOCK,
    CELEB_JSON,
    DFDCP_JSON,
    DFDC_JSON,
) = map(Path, sys.argv[1:])


###############################################################################
# HELPERS
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


def find_one(root, filename):
    hits = list(root.rglob(filename))

    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename} under {root}; "
            f"found {len(hits)}: {hits[:20]}"
        )

    return hits[0]


def frames(frame_dir):
    if not frame_dir.is_dir():
        return []

    result = sorted(
        p.resolve()
        for p in frame_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".png"
    )

    # We intentionally follow DeepfakeBench's *png convention.
    others = [
        p
        for p in frame_dir.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in {".jpg", ".jpeg", ".webp", ".bmp"}
    ]

    if not result and others:
        raise RuntimeError(
            f"{frame_dir} contains image files but no PNG frames. "
            "Refusing to silently change DeepfakeBench protocol."
        )

    if len(result) > 32:
        raise RuntimeError(
            f"{frame_dir} contains {len(result)} PNG frames (>32). "
            "This does not match the expected preprocessed benchmark layout."
        )

    return result


###############################################################################
# DISCOVER ACTUAL LOCAL X2DFD INFERENCE JSON SCHEMA
#
# Supported image locator keys:
#   image
#   image_path
#   path
#
# We do NOT invent a schema.
# We clone an actual local X2DFD item.
###############################################################################

IMAGE_KEYS = (
    "image",
    "image_path",
    "path",
)

DROP_GT_KEYS = {
    "label",
    "labels",
    "target",
    "answer",
    "answers",
    "prediction",
    "pred",
    "ground_truth",
    "groundtruth",
    "gt",
    "real_score",
    "fake_score",
}


def discover_image_lists(obj, path=()):
    found = []

    if isinstance(obj, list):

        counts = Counter()

        for item in obj:

            if not isinstance(item, dict):
                continue

            for key in IMAGE_KEYS:

                value = item.get(key)

                if (
                    isinstance(value, str)
                    and value.strip()
                ):
                    counts[key] += 1

        if counts:

            image_key, count = max(
                counts.items(),
                key=lambda kv: kv[1],
            )

            prototype = next(
                item
                for item in obj
                if (
                    isinstance(item, dict)
                    and isinstance(
                        item.get(image_key),
                        str,
                    )
                )
            )

            found.append(
                (
                    count,
                    path,
                    image_key,
                    prototype,
                )
            )

        # Limited recursion is enough to discover nested datasets
        # without unnecessarily traversing giant files.
        for i, value in enumerate(obj[:20]):

            found.extend(
                discover_image_lists(
                    value,
                    path + (i,),
                )
            )

    elif isinstance(obj, dict):

        for key, value in obj.items():

            found.extend(
                discover_image_lists(
                    value,
                    path + (key,),
                )
            )

    return found


def sanitize_item(item):

    x = copy.deepcopy(item)

    # Do not copy explicit ground-truth fields into evaluation input.
    for key in list(x.keys()):

        if str(key).lower() in DROP_GT_KEYS:
            x.pop(key, None)

    # LLaVA-style conversations may contain the GT assistant answer.
    conv = x.get("conversations")

    if isinstance(conv, list):

        cleaned = []

        for message in conv:

            if not isinstance(message, dict):
                continue

            role = str(
                message.get(
                    "from",
                    message.get("role", ""),
                )
            ).strip().lower()

            if role in {
                "assistant",
                "gpt",
                "model",
                "bot",
            }:
                continue

            cleaned.append(
                copy.deepcopy(message)
            )

        if not cleaned:
            raise RuntimeError(
                "Template conversation became empty after "
                "removing assistant/GT turns"
            )

        x["conversations"] = cleaned

    return x


def replace_at_path(obj, path, value):

    # Root-level JSON list.
    if not path:
        return value

    cur = obj

    for key in path[:-1]:
        cur = cur[key]

    cur[path[-1]] = value

    return obj


template_candidates = []

if not TEMPLATE_ROOT.exists():
    raise RuntimeError(
        f"Template search root missing: {TEMPLATE_ROOT}"
    )


for candidate in TEMPLATE_ROOT.rglob("*.json"):

    try:

        size = candidate.stat().st_size

        if (
            size <= 0
            or size > 500 * 1024 * 1024
        ):
            continue

        text = candidate.read_text(
            errors="ignore"
        )

        if not any(
            f'"{key}"' in text
            for key in IMAGE_KEYS
        ):
            continue

        obj = json.loads(text)

        discovered = discover_image_lists(obj)

        for (
            count,
            list_path,
            image_key,
            prototype,
        ) in discovered:

            name = str(candidate).lower()

            preference = 0

            for token in (
                "test",
                "eval",
                "cross",
                "dfdc",
                "dfdcp",
                "wdf",
            ):
                if token in name:
                    preference += 1_000_000

            template_candidates.append(
                (
                    preference + count,
                    count,
                    size,
                    candidate,
                    obj,
                    list_path,
                    image_key,
                    prototype,
                )
            )

    except Exception:
        continue


if not template_candidates:

    raise RuntimeError(
        "No usable local X2DFD inference JSON schema found. "
        "Checked image/image_path/path."
    )


template_candidates.sort(
    key=lambda x: (
        x[0],
        x[1],
        x[2],
    ),
    reverse=True,
)


(
    _rank,
    template_item_count,
    _template_size,
    template_path,
    template_obj,
    image_list_path,
    template_image_key,
    raw_prototype,
) = template_candidates[0]


prototype = sanitize_item(
    raw_prototype
)


schema_info = {
    "template_path":
        str(template_path),

    "template_items":
        int(template_item_count),

    "image_list_path":
        [str(x) for x in image_list_path],

    "image_key":
        template_image_key,

    "prototype_keys":
        sorted(
            str(x)
            for x in prototype.keys()
        ),
}


(LOCK.parent / "JSON_TEMPLATE_SCHEMA.json").write_text(
    json.dumps(
        schema_info,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "JSON TEMPLATE =",
    template_path,
)

print(
    "IMAGE KEY =",
    template_image_key,
)

print(
    "IMAGE LIST PATH =",
    image_list_path,
)

print(
    "TEMPLATE ITEMS =",
    template_item_count,
)


def make_x2_json(paths, output):

    obj = copy.deepcopy(
        template_obj
    )

    payload = []

    for i, image_path in enumerate(paths):

        item = copy.deepcopy(
            prototype
        )

        changed = False

        for key in IMAGE_KEYS:

            if key in item:

                item[key] = str(
                    image_path
                )

                changed = True

        if not changed:

            item[
                template_image_key
            ] = str(
                image_path
            )

        # Existing schemas frequently contain IDs.
        # They must not remain duplicated.
        for key in (
            "id",
            "sample_id",
            "uid",
        ):

            if key in item:

                item[key] = (
                    f"paper_eval_{i:09d}"
                )

        for key in (
            "filename",
            "file_name",
        ):

            if key in item:

                item[key] = Path(
                    image_path
                ).name

        payload.append(
            item
        )

    obj = replace_at_path(
        obj,
        image_list_path,
        payload,
    )

    output.write_text(
        json.dumps(
            obj,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Post-write structural verification.
    check = json.loads(
        output.read_text()
    )

    target = check

    for key in image_list_path:
        target = target[key]

    if not isinstance(target, list):

        raise RuntimeError(
            f"Generated JSON target is not list: {output}"
        )

    recovered = []

    for item in target:

        if not isinstance(item, dict):

            raise RuntimeError(
                f"Generated non-dict item: {output}"
            )

        image = None

        for key in IMAGE_KEYS:

            value = item.get(key)

            if (
                isinstance(value, str)
                and value.strip()
            ):

                image = value
                break

        if image is None:

            raise RuntimeError(
                f"Generated item lacks image locator: {output}"
            )

        recovered.append(image)

    expected = [
        str(x)
        for x in paths
    ]

    if recovered != expected:

        raise RuntimeError(
            f"Generated JSON image paths mismatch: {output}"
        )

    print(
        "X2 JSON READY:",
        output,
        "N=",
        len(recovered),
    )


###############################################################################
# BUILD RECORDS
###############################################################################

records = []
zero_frame = []


def add(dataset, video_id, label, frame_paths, source):
    if label not in {"real", "fake"}:
        raise RuntimeError(
            f"Invalid normalized label: {label}"
        )

    if not frame_paths:
        zero_frame.append(
            {
                "dataset": dataset,
                "video_id": video_id,
                "source": source,
            }
        )

        return

    for p in frame_paths:
        records.append(
            {
                "dataset": dataset,
                "video_id": video_id,
                "label_text": label,
                "image_path": str(p),
                "source_metadata": source,
            }
        )


###############################################################################
# CELEB-DF-v2
###############################################################################

celeb_meta = find_one(
    DATA / "Celeb-DF-v2",
    "List_of_testing_videos.txt",
)

celeb_root = celeb_meta.parent

celeb_video_count = 0

for raw_line in celeb_meta.read_text(
    errors="ignore"
).splitlines():

    line = raw_line.strip()

    if not line:
        continue

    tokens = line.split()

    mp4 = next(
        (
            t
            for t in tokens
            if t.lower().endswith(".mp4")
        ),
        None,
    )

    if mp4 is None:
        continue

    rel = Path(mp4)

    category = rel.parent.name
    video_id = rel.stem

    if category in {
        "Celeb-real",
        "YouTube-real",
    }:
        label = "real"

    elif category == "Celeb-synthesis":
        label = "fake"

    else:
        raise RuntimeError(
            f"Unknown Celeb-DF-v2 category: {category}"
        )

    frame_dir = (
        celeb_root
        / category
        / "frames"
        / video_id
    )

    ff = frames(frame_dir)

    add(
        "Celeb-DF-v2",
        video_id,
        label,
        ff,
        f"{celeb_meta}:{mp4}",
    )

    celeb_video_count += 1


if celeb_video_count < 100:
    raise RuntimeError(
        f"Suspicious Celeb-DF-v2 official test count: "
        f"{celeb_video_count}"
    )


###############################################################################
# DFDCP
###############################################################################

dfdcp_meta = find_one(
    DATA / "DFDCP",
    "dataset.json",
)

dfdcp_root = dfdcp_meta.parent

dfdcp_info = json.loads(
    dfdcp_meta.read_text(
        errors="ignore"
    )
)

if not isinstance(dfdcp_info, dict):
    raise RuntimeError(
        "DFDCP dataset.json top level is not a mapping"
    )


dfdcp_video_count = 0

for dataset_key, info in dfdcp_info.items():

    if not isinstance(info, dict):
        continue

    split = str(
        info.get("set", "")
    ).strip().lower()

    if split != "test":
        continue

    raw_label = str(
        info.get("label", "")
    ).strip().lower()

    if raw_label == "real":
        label = "real"

    elif raw_label == "fake":
        label = "fake"

    else:
        raise RuntimeError(
            f"Invalid DFDCP label: "
            f"{dataset_key} -> {raw_label}"
        )

    parts = Path(dataset_key).parts

    if len(parts) < 2:
        raise RuntimeError(
            f"Unexpected DFDCP dataset key: {dataset_key}"
        )

    index = parts[0]
    video_id = Path(parts[-1]).stem

    # DeepfakeBench official layout:
    # <index>/frames/<video_id>/*.png
    frame_dir = (
        dfdcp_root
        / index
        / "frames"
        / video_id
    )

    ff = frames(frame_dir)

    add(
        "DFDCP",
        f"{index}/{video_id}",
        label,
        ff,
        f"{dfdcp_meta}:{dataset_key}",
    )

    dfdcp_video_count += 1


if dfdcp_video_count < 10:
    raise RuntimeError(
        f"Suspicious DFDCP test count: "
        f"{dfdcp_video_count}"
    )


###############################################################################
# DFDC
###############################################################################

dfdc_labels = find_one(
    DATA / "DFDC",
    "labels.csv",
)

if dfdc_labels.parent.name.lower() != "test":
    raise RuntimeError(
        f"DFDC labels.csv is not under test/: {dfdc_labels}"
    )

dfdc_root = dfdc_labels.parent

with dfdc_labels.open(
    "r",
    encoding="utf-8",
    errors="ignore",
    newline="",
) as f:

    rows = list(
        csv.DictReader(f)
    )


if not rows:
    raise RuntimeError(
        "DFDC labels.csv is empty"
    )


fields = rows[0].keys()

filename_col = next(
    (
        x for x in fields
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
        x for x in fields
        if x.lower()
        in {
            "label",
            "class",
            "target",
        }
    ),
    None,
)


if filename_col is None or label_col is None:
    raise RuntimeError(
        f"Unsupported DFDC labels schema: {list(fields)}"
    )


dfdc_video_count = 0

for row in rows:

    filename = str(
        row[filename_col]
    ).strip()

    raw_label = str(
        row[label_col]
    ).strip()

    # Official DeepfakeBench rearrange.py:
    # labels = ['DFDC_Real', 'DFDC_Fake']
    # label = labels[row['label']]
    if raw_label == "0":
        label = "real"

    elif raw_label == "1":
        label = "fake"

    elif raw_label.upper() == "REAL":
        label = "real"

    elif raw_label.upper() == "FAKE":
        label = "fake"

    else:
        raise RuntimeError(
            f"Unsupported DFDC label value: "
            f"{filename} -> {raw_label}"
        )

    video_id = Path(
        filename
    ).stem

    frame_dir = (
        dfdc_root
        / "frames"
        / video_id
    )

    ff = frames(frame_dir)

    add(
        "DFDC",
        video_id,
        label,
        ff,
        f"{dfdc_labels}:{filename}",
    )

    dfdc_video_count += 1


if dfdc_video_count < 10:
    raise RuntimeError(
        f"Suspicious DFDC test count: "
        f"{dfdc_video_count}"
    )


###############################################################################
# GLOBAL MANIFEST GATES
###############################################################################

if not records:
    raise RuntimeError(
        "Benchmark manifest is empty"
    )


paths = [
    r["image_path"]
    for r in records
]

if len(paths) != len(set(paths)):
    dup = len(paths) - len(set(paths))

    raise RuntimeError(
        f"Duplicate image paths in benchmark manifest: {dup}"
    )


for p in paths:
    if not Path(p).is_file():
        raise RuntimeError(
            f"Missing benchmark frame: {p}"
        )


###############################################################################
# FRAME-COUNT / LABEL DISTRIBUTIONS
###############################################################################

video_frames = defaultdict(int)
video_labels = {}

for r in records:

    key = (
        r["dataset"],
        r["video_id"],
    )

    video_frames[key] += 1

    old = video_labels.get(key)

    if old is not None and old != r["label_text"]:
        raise RuntimeError(
            f"Conflicting labels for video {key}"
        )

    video_labels[key] = r["label_text"]


dataset_stats = {}

for ds in [
    "Celeb-DF-v2",
    "DFDCP",
    "DFDC",
]:

    vf = {
        vid: n
        for (dataset, vid), n
        in video_frames.items()
        if dataset == ds
    }

    labels = Counter(
        label
        for (dataset, vid), label
        in video_labels.items()
        if dataset == ds
    )

    dist = Counter(
        vf.values()
    )

    if not vf:
        raise RuntimeError(
            f"No usable videos for {ds}"
        )

    if max(vf.values()) > 32:
        raise RuntimeError(
            f"{ds} has a video with >32 frames"
        )

    dataset_stats[ds] = {
        "videos": len(vf),
        "frames": sum(vf.values()),
        "real_videos": labels["real"],
        "fake_videos": labels["fake"],
        "frames_per_video_distribution": {
            str(k): v
            for k, v
            in sorted(dist.items())
        },
        "min_frames": min(vf.values()),
        "max_frames": max(vf.values()),
    }


###############################################################################
# ZERO-FRAME GATE
#
# DeepfakeBench rearrange skips zero-frame DFDCP/DFDC videos.
# We record them. If >1% of official test videos disappear, fail.
###############################################################################

official_total = (
    celeb_video_count
    + dfdcp_video_count
    + dfdc_video_count
)

zero_fraction = (
    len(zero_frame)
    / official_total
)

if zero_fraction > 0.01:
    raise RuntimeError(
        f"Too many official test videos have zero usable frames: "
        f"{len(zero_frame)}/{official_total} "
        f"({100*zero_fraction:.3f}%)"
    )


###############################################################################
# WRITE LABEL MANIFEST
###############################################################################

with LABELS.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    fieldnames = [
        "dataset",
        "video_id",
        "label_text",
        "image_path",
        "source_metadata",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    w.writeheader()
    w.writerows(records)


###############################################################################
# WRITE EXPERT MANIFEST
###############################################################################

with MANIFEST.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    fieldnames = [
        "json",
        "image_path",
        "dataset",
        "video_id",
        "label_text",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    w.writeheader()

    for r in records:

        dataset_json = {
            "Celeb-DF-v2": CELEB_JSON,
            "DFDCP": DFDCP_JSON,
            "DFDC": DFDC_JSON,
        }[r["dataset"]]

        w.writerow(
            {
                "json": str(dataset_json),
                "image_path": r["image_path"],
                "dataset": r["dataset"],
                "video_id": r["video_id"],
                "label_text": r["label_text"],
            }
        )


###############################################################################
# WRITE X2DFD JSONS USING EXISTING LOCAL TEMPLATE
###############################################################################

for ds, output in [
    ("Celeb-DF-v2", CELEB_JSON),
    ("DFDCP", DFDCP_JSON),
    ("DFDC", DFDC_JSON),
]:

    pp = [
        r["image_path"]
        for r in records
        if r["dataset"] == ds
    ]

    make_x2_json(
        pp,
        output,
    )


###############################################################################
# METADATA HASHES
###############################################################################

metadata_hashes = {
    "Celeb-DF-v2_test_list": {
        "path": str(celeb_meta),
        "sha256": sha256(celeb_meta),
    },
    "DFDCP_dataset_json": {
        "path": str(dfdcp_meta),
        "sha256": sha256(dfdcp_meta),
    },
    "DFDC_labels_csv": {
        "path": str(dfdc_labels),
        "sha256": sha256(dfdc_labels),
    },
}


###############################################################################
# FINAL PROTOCOL LOCK
###############################################################################

lock = {
    "protocol_name":
        "DeepfakeBench-compatible cross-data evaluation",

    "evaluation_unit_input":
        "preprocessed face frames",

    "sampling": {
        "middle_frame": False,
        "random": False,
        "resampled_after_download": False,
        "duplicated_to_32": False,
        "frame_source":
            "all available official preprocessed PNG frames",
        "maximum_expected_frames_per_video": 32,
    },

    "split_sources": metadata_hashes,

    "dataset_stats": dataset_stats,

    "zero_frame_official_videos": zero_frame,

    "json_template": {
        "path": str(template_path),
        "image_list_path": [
            str(x)
            for x in image_list_path
        ],
    },

    "official_video_counts_seen": {
        "Celeb-DF-v2": celeb_video_count,
        "DFDCP": dfdcp_video_count,
        "DFDC": dfdc_video_count,
    },

    "total_usable_frames": len(records),

    "notes": [
        "No middle-frame-only protocol.",
        "No random evaluation sampling.",
        "No duplication of frames.",
        "Videos with fewer than 32 valid preprocessed frames retain all valid frames.",
        "Any video with more than 32 PNG frames causes a hard failure.",
        "Both models will receive the exact same frozen image manifest.",
    ],
}


LOCK.write_text(
    json.dumps(
        lock,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    json.dumps(
        lock,
        indent=2,
    )
)

print()
print("TOTAL USABLE FRAMES =", len(records))
print("ZERO-FRAME VIDEOS =", len(zero_frame))

PYMAN


###############################################################################
# D. FREEZE MANIFEST
###############################################################################

stage "D. FREEZE MANIFEST + METADATA"

sha256sum \
 "$MANIFEST" \
 "$LABELS" \
 "$LOCK" \
 "$CELEB_JSON" \
 "$DFDCP_JSON" \
 "$DFDC_JSON" \
 > "$PROTO/MANIFEST_AND_PROTOCOL.sha256"


cat "$PROTO/MANIFEST_AND_PROTOCOL.sha256"


EXPECTED=$(
    tail -n +2 "$MANIFEST" \
    | wc -l
)


[ "$EXPECTED" -gt 0 ] || \
    die "Frozen benchmark manifest is empty"


echo "FROZEN FRAME COUNT = $EXPECTED"


###############################################################################
# E. EXACT-PATH TRAIN/TEST LEAKAGE
###############################################################################

stage "E. EXACT-PATH TRAIN/TEST LEAKAGE"

"$PY_X2" - \
 "$TRAIN_TEACHER" \
 "$MANIFEST" \
 "$PROTO/exact_path_leakage.json" \
 <<'PYPATH'

from pathlib import Path
import csv
import json
import sys

train_csv = Path(sys.argv[1])
test_csv = Path(sys.argv[2])
out = Path(sys.argv[3])


def load(path):

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
        newline="",
    ) as f:

        r = csv.DictReader(f)

        fields = r.fieldnames or []

        col = next(
            (
                x
                for x in [
                    "image_path",
                    "image",
                    "path",
                ]
                if x in fields
            ),
            None,
        )

        if col is None:
            raise RuntimeError(
                f"No image path field in {path}"
            )

        values = set()

        for row in r:

            p = str(
                row[col]
            ).strip()

            if p:
                values.add(
                    str(
                        Path(p).resolve()
                    )
                )

        return values


train = load(train_csv)
test = load(test_csv)

overlap = train & test

result = {
    "train_unique": len(train),
    "test_unique": len(test),
    "exact_path_overlap": len(overlap),
    "examples": sorted(overlap)[:100],
}

out.write_text(
    json.dumps(
        result,
        indent=2,
    )
)

print(
    json.dumps(
        result,
        indent=2,
    )
)

if overlap:
    raise SystemExit(10)

PYPATH


###############################################################################
# F. CREATE LOW-PRIORITY DECODED-RGB CONTENT LEAKAGE AUDIT
###############################################################################

stage "F. START DECODED-RGB LEAKAGE AUDIT"

HASH_SCRIPT="$RUNROOT/decoded_rgb_audit.py"

cat > "$HASH_SCRIPT" <<'PYHASH'

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import csv
import hashlib
import json
import sys

from PIL import Image


train_csv = Path(sys.argv[1])
test_csv = Path(sys.argv[2])
out = Path(sys.argv[3])


def train_paths(path):

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
        newline="",
    ) as f:

        r = csv.DictReader(f)
        fields = r.fieldnames or []

        col = next(
            (
                c
                for c in [
                    "image_path",
                    "image",
                    "path",
                ]
                if c in fields
            ),
            None,
        )

        if col is None:
            raise RuntimeError(
                "No train image-path column"
            )

        return sorted(
            set(
                str(row[col]).strip()
                for row in r
                if str(row[col]).strip()
            )
        )


def test_paths(path):

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
        newline="",
    ) as f:

        r = csv.DictReader(f)

        rows = []

        for row in r:

            p = str(
                row["image_path"]
            ).strip()

            if p:
                rows.append(
                    (
                        p,
                        str(
                            row.get(
                                "dataset",
                                "",
                            )
                        ),
                    )
                )

        unique = {}

        for p, ds in rows:
            unique[p] = ds

        return sorted(
            unique.items()
        )


def h_rgb(path):

    try:
        with Image.open(path) as im:

            im = im.convert("RGB")

            h = hashlib.sha256()

            h.update(
                f"{im.width}x{im.height}|RGB|".encode()
            )

            h.update(
                im.tobytes()
            )

        return path, h.hexdigest(), None

    except Exception as e:
        return path, None, repr(e)


tr_paths = train_paths(train_csv)
te_rows = test_paths(test_csv)

te_paths = [
    p
    for p, _
    in te_rows
]

te_dataset = dict(te_rows)


def compute(paths, tag):

    hash_to_paths = defaultdict(list)
    errors = []

    # deliberately conservative so GPU inference I/O is not starved
    with ThreadPoolExecutor(
        max_workers=2
    ) as ex:

        for i, result in enumerate(
            ex.map(
                h_rgb,
                paths,
            ),
            1,
        ):

            p, h, err = result

            if err is None:
                hash_to_paths[h].append(p)

            else:
                errors.append(
                    {
                        "path": p,
                        "error": err,
                    }
                )

            if i % 5000 == 0:
                print(
                    tag,
                    i,
                    "/",
                    len(paths),
                    flush=True,
                )

    return hash_to_paths, errors


print(
    "TRAIN FILES =",
    len(tr_paths),
    flush=True,
)

print(
    "TEST FILES =",
    len(te_paths),
    flush=True,
)


train_hash, train_err = compute(
    tr_paths,
    "TRAIN",
)

test_hash, test_err = compute(
    te_paths,
    "TEST",
)


common = sorted(
    set(train_hash)
    & set(test_hash)
)


cross_dataset_duplicates = []

for h, pp in test_hash.items():

    datasets = {
        te_dataset.get(p, "")
        for p in pp
    }

    if len(datasets) > 1:
        cross_dataset_duplicates.append(
            {
                "hash": h,
                "datasets": sorted(datasets),
                "paths": pp[:20],
            }
        )


result = {
    "train_files": len(tr_paths),
    "test_files": len(te_paths),

    "train_decode_errors": len(train_err),
    "test_decode_errors": len(test_err),

    "decoded_rgb_train_test_overlap":
        len(common),

    "overlap_examples": [
        {
            "hash": h,
            "train": train_hash[h][:5],
            "test": test_hash[h][:5],
        }
        for h in common[:100]
    ],

    "cross_dataset_duplicate_hashes":
        len(cross_dataset_duplicates),

    "cross_dataset_duplicate_examples":
        cross_dataset_duplicates[:100],

    "train_error_examples":
        train_err[:50],

    "test_error_examples":
        test_err[:50],
}


out.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    json.dumps(
        result,
        indent=2,
    ),
    flush=True,
)


if train_err or test_err:
    raise SystemExit(20)


if common:
    raise SystemExit(30)

PYHASH


HASH_RESULT="$PROTO/decoded_rgb_leakage.json"
HASH_LOG="$LOGS/decoded_rgb_leakage.log"


nice -n 10 \
 "$PY_X2" -u \
 "$HASH_SCRIPT" \
 "$TRAIN_TEACHER" \
 "$MANIFEST" \
 "$HASH_RESULT" \
 > "$HASH_LOG" \
 2>&1 &


HASH_PID=$!

echo "HASH AUDIT PID = $HASH_PID"


###############################################################################
# G. MAKE SMOKE MANIFEST
#
# 2 real + 2 fake frames per dataset where available.
###############################################################################

stage "G. BUILD SMOKE SET"

mkdir -p "$SMOKE/work"


"$PY_X2" - \
 "$MANIFEST" \
 "$SMOKE/work/manifest.csv" \
 <<'PYSMOKE'

from collections import defaultdict
import csv
import sys
from pathlib import Path


src = Path(sys.argv[1])
dst = Path(sys.argv[2])


with src.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    rr = csv.DictReader(f)

    rows = list(rr)
    fields = rr.fieldnames


groups = defaultdict(list)

for r in rows:

    groups[
        (
            r["dataset"],
            r["label_text"],
        )
    ].append(r)


chosen = []

for dataset in [
    "Celeb-DF-v2",
    "DFDCP",
    "DFDC",
]:

    for label in [
        "real",
        "fake",
    ]:

        arr = groups[
            (
                dataset,
                label,
            )
        ]

        if not arr:
            raise RuntimeError(
                f"No {dataset}/{label} samples "
                "for smoke test"
            )

        chosen.extend(
            arr[:2]
        )


with dst.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(chosen)


print(
    "SMOKE ROWS =",
    len(chosen),
)

PYSMOKE


###############################################################################
# H. RUN FOUR-EXPERT SMOKE
###############################################################################

stage "H1. SMOKE BLENDING"

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
 "$PY_X2" -u \
 "$RUN_X2" \
 --out "$SMOKE/work" \
 --expert blending \
 2>&1 \
 | tee "$LOGS/smoke_blending.log"


stage "H2. SMOKE DIFFUSION"

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
 "$PY_X2" -u \
 "$RUN_X2" \
 --out "$SMOKE/work" \
 --expert diffusion \
 2>&1 \
 | tee "$LOGS/smoke_diffusion.log"


stage "H3. SMOKE FREQUENCY"

cd "$DFF"

PYTHONPATH="$DFF:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=0 \
 "$PY_FREQ" -u \
 "$RUN_FREQ" \
 --out "$SMOKE/work" \
 --name PAPER_SMOKE \
 2>&1 \
 | tee "$LOGS/smoke_frequency.log"


stage "H4. SMOKE TEXTURE"

CUDA_VISIBLE_DEVICES=0 \
 "$PY_TEX" -u \
 "$RUN_TEX" \
 --out "$SMOKE/work" \
 --work "$TEXWORK" \
 2>&1 \
 | tee "$LOGS/smoke_texture.log"


SMOKE_N=$(
    tail -n +2 \
    "$SMOKE/work/manifest.csv" \
    | wc -l
)


for E in \
 blending \
 diffusion \
 frequency \
 texture
do

    CSV="$SMOKE/work/${E}_scores.csv"

    [ -s "$CSV" ] || \
        die "Smoke expert missing: $E"

    N=$(
        tail -n +2 "$CSV" \
        | wc -l
    )

    [ "$N" -eq "$SMOKE_N" ] || \
        die "Smoke coverage mismatch: $E $N/$SMOKE_N"

done


echo "✅ FOUR EXPERT SMOKE PASS"


###############################################################################
# I. PATCH ROUTER CACHE BUILDER FUNCTION
###############################################################################

patch_cache_builder() {

    local TARGET="$1"
    local WORKDIR="$2"

    cp -f \
     "$CACHE_SRC" \
     "$TARGET"

    "$PY_X2" - \
     "$TARGET" \
     "$WORKDIR" \
     "$FINAL_ROUTER" \
     "$FINAL_CAL" \
     "$TRAIN_TEACHER" \
     "$VAL_TEACHER" \
     <<'PYCACHE'

from pathlib import Path
import sys

p = Path(sys.argv[1])

work = sys.argv[2]
router = sys.argv[3]
cal = sys.argv[4]
train_teacher = sys.argv[5]
val_teacher = sys.argv[6]

s = p.read_text(
    errors="ignore"
)

repls = {
    "outputs/x2dfd_router_integration/full_DFDCP_DFDC":
        work,

    "/home/aiotlab/hoangan/outputs/x2dfd_router_integration/full_DFDCP_DFDC":
        work,

    "outputs/router_training_v2/run_effb0_3domain/best.pt":
        router,

    "/home/aiotlab/hoangan/outputs/router_training_v2/run_effb0_3domain/best.pt":
        router,

    "outputs/router_teacher_v2/calibrators_train_only.joblib":
        cal,

    "/home/aiotlab/hoangan/outputs/router_teacher_v2/calibrators_train_only.joblib":
        cal,

    "outputs/router_teacher_v2/train_teacher_v2.csv":
        train_teacher,

    "/home/aiotlab/hoangan/outputs/router_teacher_v2/train_teacher_v2.csv":
        train_teacher,

    "outputs/router_teacher_v2/val_teacher_v2.csv":
        val_teacher,

    "/home/aiotlab/hoangan/outputs/router_teacher_v2/val_teacher_v2.csv":
        val_teacher,
}


for old, new in repls.items():
    s = s.replace(
        old,
        new,
    )


forbidden = [
    "run_effb0_3domain/best.pt",
    "calibrators_train_only.joblib",
    "train_teacher_v2.csv",
    "val_teacher_v2.csv",
]


still = [
    x
    for x in forbidden
    if x in s
]


if still:
    raise RuntimeError(
        "Old fitted artifacts still referenced "
        f"after patch: {still}"
    )


p.write_text(
    s
)

print(
    "PATCHED CACHE BUILDER =",
    p,
)

PYCACHE

}


###############################################################################
# J. ROUTER SMOKE
###############################################################################

stage "J. ROUTER CACHE SMOKE"

SMOKE_CACHE="$SMOKE/build_router_cache.py"

patch_cache_builder \
 "$SMOKE_CACHE" \
 "$SMOKE/work"


CUDA_VISIBLE_DEVICES=0 \
 "$PY_X2" -u \
 "$SMOKE_CACHE" \
 2>&1 \
 | tee "$LOGS/smoke_router_cache.log"


[ -s "$SMOKE/work/router_moe_cache.csv" ] || \
    die "Smoke Router cache missing"


CACHE_N=$(
    tail -n +2 \
    "$SMOKE/work/router_moe_cache.csv" \
    | wc -l
)


[ "$CACHE_N" -eq "$SMOKE_N" ] || \
    die "Smoke Router cache mismatch: $CACHE_N/$SMOKE_N"


echo "✅ ROUTER SMOKE PASS"


###############################################################################
# K. PREPARE CONFIG GENERATOR SAFELY
###############################################################################

stage "K. PREPARE INFERENCE CONFIG TEMPLATE"

CFG_ORIG_BACKUP="$RUNROOT/original_repo_config.backup"
CFG_R4_BACKUP="$RUNROOT/router_repo_config.backup"


[ -f "$OLD_CONFIG_ORIG" ] && \
    cp -f \
    "$OLD_CONFIG_ORIG" \
    "$CFG_ORIG_BACKUP" || true


[ -f "$OLD_CONFIG_R4" ] && \
    cp -f \
    "$OLD_CONFIG_R4" \
    "$CFG_R4_BACKUP" || true


MAKE_RUN="$RUNROOT/make_full_configs_RUNTIME.py"

cp -f \
 "$MAKE_SRC" \
 "$MAKE_RUN"


"$PY_X2" - \
 "$MAKE_RUN" \
 "$SMOKE/work" \
 <<'PYMAKE'

from pathlib import Path
import sys

p = Path(sys.argv[1])
work = sys.argv[2]

s = p.read_text(
    errors="ignore"
)

s = s.replace(
    "/home/aiotlab/hoangan/outputs/"
    "x2dfd_router_integration/full_DFDCP_DFDC",
    work,
)

s = s.replace(
    "outputs/x2dfd_router_integration/"
    "full_DFDCP_DFDC",
    work,
)

p.write_text(
    s
)

PYMAKE


cd "$INT"

"$PY_X2" \
 "$MAKE_RUN" \
 2>&1 \
 | tee "$LOGS/config_generator.log"


[ -s "$OLD_CONFIG_ORIG" ] || \
    die "Original inference YAML not generated"


[ -s "$OLD_CONFIG_R4" ] || \
    die "Router inference YAML not generated"


BASE_CFG_ORIG="$CONFIGS/base_original.yaml"
BASE_CFG_R4="$CONFIGS/base_router4.yaml"


cp -f \
 "$OLD_CONFIG_ORIG" \
 "$BASE_CFG_ORIG"


cp -f \
 "$OLD_CONFIG_R4" \
 "$BASE_CFG_R4"


# Restore repository configs immediately.
[ -f "$CFG_ORIG_BACKUP" ] && \
    cp -f \
    "$CFG_ORIG_BACKUP" \
    "$OLD_CONFIG_ORIG" || true


[ -f "$CFG_R4_BACKUP" ] && \
    cp -f \
    "$CFG_R4_BACKUP" \
    "$OLD_CONFIG_R4" || true


###############################################################################
# L. CONFIG PATCHER
#
# It does NOT assume "infer.inputs".
# It finds exactly one list of JSON input paths.
###############################################################################

PATCH_CFG="$RUNROOT/patch_config.py"

cat > "$PATCH_CFG" <<'PYCFG'

from pathlib import Path
import copy
import sys
import yaml


src = Path(sys.argv[1])
dst = Path(sys.argv[2])
work = sys.argv[3]

inputs = sys.argv[4:]


cfg = yaml.safe_load(
    src.read_text(
        errors="ignore"
    )
)


def rewrite_strings(x):

    if isinstance(x, dict):

        return {
            k: rewrite_strings(v)
            for k, v in x.items()
        }

    if isinstance(x, list):

        return [
            rewrite_strings(v)
            for v in x
        ]

    if isinstance(x, str):

        return x.replace(
            "/home/aiotlab/hoangan/outputs/"
            "x2dfd_router_integration/full_DFDCP_DFDC",
            work,
        ).replace(
            "outputs/x2dfd_router_integration/"
            "full_DFDCP_DFDC",
            work,
        )

    return x


cfg = rewrite_strings(cfg)


candidates = []


def walk(x, path=()):

    if isinstance(x, dict):

        for k, v in x.items():

            if (
                isinstance(v, list)
                and v
                and all(
                    isinstance(z, str)
                    and z.lower().endswith(".json")
                    for z in v
                )
            ):

                candidates.append(
                    path + (k,)
                )

            walk(
                v,
                path + (k,),
            )

    elif isinstance(x, list):

        for i, v in enumerate(x):
            walk(
                v,
                path + (i,),
            )


walk(cfg)


if len(candidates) != 1:
    raise RuntimeError(
        "Expected exactly one JSON input list in config; "
        f"found {candidates}"
    )


target = candidates[0]


cur = cfg

for key in target[:-1]:
    cur = cur[key]


cur[target[-1]] = inputs


dst.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
    )
)


print(
    "PATCHED CONFIG =",
    dst,
)

print(
    "INPUT LIST PATH =",
    target,
)

print(
    "INPUTS =",
    inputs,
)

PYCFG


###############################################################################
# M. BUILD SMOKE JSONS FROM FULL JSONS
###############################################################################

stage "M. BUILD SMOKE INPUT JSONS"

"$PY_X2" - \
 "$SMOKE/work/manifest.csv" \
 "$CELEB_JSON" \
 "$DFDCP_JSON" \
 "$DFDC_JSON" \
 "$SMOKE" \
 <<'PYSMOKEJSON'

from pathlib import Path
import copy
import csv
import json
import sys


manifest = Path(sys.argv[1])

full_jsons = {
    "Celeb-DF-v2": Path(sys.argv[2]),
    "DFDCP": Path(sys.argv[3]),
    "DFDC": Path(sys.argv[4]),
}

outdir = Path(sys.argv[5])

IMAGE_KEYS = (
    "image",
    "image_path",
    "path",
)


with manifest.open(
    "r",
    newline="",
    encoding="utf-8",
) as f:

    rows = list(
        csv.DictReader(f)
    )


wanted = {
    ds: {
        r["image_path"]
        for r in rows
        if r["dataset"] == ds
    }
    for ds in full_jsons
}


def discover(obj, path=()):

    found = []

    if isinstance(obj, list):

        counts = {}

        for item in obj:

            if not isinstance(item, dict):
                continue

            for key in IMAGE_KEYS:

                value = item.get(key)

                if (
                    isinstance(value, str)
                    and value.strip()
                ):

                    counts[key] = (
                        counts.get(key, 0)
                        + 1
                    )

        if counts:

            image_key, count = max(
                counts.items(),
                key=lambda kv: kv[1],
            )

            found.append(
                (
                    count,
                    path,
                    image_key,
                )
            )

        for i, value in enumerate(obj[:20]):

            found.extend(
                discover(
                    value,
                    path + (i,),
                )
            )

    elif isinstance(obj, dict):

        for key, value in obj.items():

            found.extend(
                discover(
                    value,
                    path + (key,),
                )
            )

    return found


def get_at(obj, path):

    cur = obj

    for key in path:
        cur = cur[key]

    return cur


def replace_at(obj, path, value):

    if not path:
        return value

    cur = obj

    for key in path[:-1]:
        cur = cur[key]

    cur[path[-1]] = value

    return obj


for ds, src in full_jsons.items():

    obj = json.loads(
        src.read_text(
            errors="ignore"
        )
    )

    candidates = discover(obj)

    if not candidates:

        raise RuntimeError(
            f"No image list detected in {src}"
        )

    count, path, image_key = max(
        candidates,
        key=lambda x: x[0],
    )

    arr = get_at(
        obj,
        path,
    )

    selected = []
    recovered = set()

    for item in arr:

        if not isinstance(item, dict):
            continue

        image = None

        for key in IMAGE_KEYS:

            value = item.get(key)

            if (
                isinstance(value, str)
                and value.strip()
            ):

                image = value
                break

        if image in wanted[ds]:

            selected.append(
                copy.deepcopy(item)
            )

            recovered.add(image)

    if recovered != wanted[ds]:

        missing = sorted(
            wanted[ds] - recovered
        )

        raise RuntimeError(
            f"Smoke JSON could not recover all {ds} samples. "
            f"Missing={missing[:20]}"
        )

    obj = replace_at(
        obj,
        path,
        selected,
    )

    output = (
        outdir
        / f"{ds}_smoke.json"
    )

    output.write_text(
        json.dumps(
            obj,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "SMOKE JSON:",
        ds,
        "N=",
        len(selected),
        "key=",
        image_key,
        "->",
        output,
    )

PYSMOKEJSON


###############################################################################
# N. BUILD SMOKE CONFIGS
###############################################################################

stage "N. BUILD SMOKE CONFIGS"

SMOKE_CFG_ORIG="$SMOKE/original.yaml"
SMOKE_CFG_R4="$SMOKE/router4.yaml"


"$PY_X2" \
 "$PATCH_CFG" \
 "$BASE_CFG_ORIG" \
 "$SMOKE_CFG_ORIG" \
 "$SMOKE/work" \
 "$SMOKE/Celeb-DF-v2_smoke.json" \
 "$SMOKE/DFDCP_smoke.json" \
 "$SMOKE/DFDC_smoke.json"


"$PY_X2" \
 "$PATCH_CFG" \
 "$BASE_CFG_R4" \
 "$SMOKE_CFG_R4" \
 "$SMOKE/work" \
 "$SMOKE/Celeb-DF-v2_smoke.json" \
 "$SMOKE/DFDCP_smoke.json" \
 "$SMOKE/DFDC_smoke.json"


###############################################################################
# O. SNAPSHOT HELPER
###############################################################################

SNAPSHOT="$RUNROOT/snapshot_run.py"

cat > "$SNAPSHOT" <<'PYSNAP'

from pathlib import Path
import json
import shutil
import sys


pointer = Path(
    sys.argv[1]
)

outdir = Path(
    sys.argv[2]
)

outdir.mkdir(
    parents=True,
    exist_ok=True,
)


obj = json.loads(
    pointer.read_text(
        errors="ignore"
    )
)


(outdir / "latest_run.json").write_text(
    json.dumps(
        obj,
        indent=2,
    )
)


def strings(x):

    if isinstance(x, dict):

        for v in x.values():
            yield from strings(v)

    elif isinstance(x, list):

        for v in x:
            yield from strings(v)

    elif isinstance(x, str):

        yield x


existing = []


for s in strings(obj):

    p = Path(s)

    if p.exists():
        existing.append(
            str(
                p.resolve()
            )
        )


existing = list(
    dict.fromkeys(existing)
)


(outdir / "referenced_paths.json").write_text(
    json.dumps(
        existing,
        indent=2,
    )
)


artifact_dir = (
    outdir
    / "artifacts"
)

artifact_dir.mkdir(
    exist_ok=True
)


copied = []


def maybe_copy(p):

    allowed = {
        ".json",
        ".jsonl",
        ".csv",
        ".txt",
    }

    if (
        p.is_file()
        and p.suffix.lower() in allowed
    ):

        try:
            if p.stat().st_size > 2 * 1024**3:
                return
        except:
            return

        name = (
            str(p.resolve())
            .replace("/", "__")
        )

        target = (
            artifact_dir
            / name
        )

        shutil.copy2(
            p,
            target,
        )

        copied.append(
            {
                "source": str(p),
                "copy": str(target),
            }
        )

    elif p.is_dir():

        for q in p.rglob("*"):

            if not q.is_file():
                continue

            if q.suffix.lower() not in allowed:
                continue

            try:
                if q.stat().st_size > 2 * 1024**3:
                    continue
            except:
                continue

            name = (
                str(q.resolve())
                .replace("/", "__")
            )

            target = (
                artifact_dir
                / name
            )

            shutil.copy2(
                q,
                target,
            )

            copied.append(
                {
                    "source": str(q),
                    "copy": str(target),
                }
            )


for s in existing:
    maybe_copy(
        Path(s)
    )


(outdir / "COPIED_ARTIFACTS.json").write_text(
    json.dumps(
        copied,
        indent=2,
    )
)


print(
    "REFERENCED =",
    len(existing),
)

print(
    "COPIED =",
    len(copied),
)

PYSNAP


###############################################################################
# P. TOKEN SCORE CONTRACT CHECK
###############################################################################

TOKEN_CHECK="$RUNROOT/check_token_scores.py"

cat > "$TOKEN_CHECK" <<'PYTOKEN'

from pathlib import Path
import csv
import json
import math
import re
import sys


root = Path(
    sys.argv[1]
)

outfile = Path(
    sys.argv[2]
)


files = []

for ext in [
    "*.json",
    "*.jsonl",
    "*.csv",
]:

    files.extend(
        root.rglob(ext)
    )


keys_found = {}

numeric_count = 0


real_rx = re.compile(
    r"real.*(?:score|prob)|"
    r"(?:score|prob).*real",
    re.I,
)

fake_rx = re.compile(
    r"fake.*(?:score|prob)|"
    r"(?:score|prob).*fake",
    re.I,
)


def walk(x):

    global numeric_count

    if isinstance(x, dict):

        for k, v in x.items():

            lk = str(k)

            if (
                real_rx.search(lk)
                or fake_rx.search(lk)
            ):

                try:
                    fv = float(v)

                    if math.isfinite(fv):

                        keys_found.setdefault(
                            lk,
                            0,
                        )

                        keys_found[lk] += 1
                        numeric_count += 1

                except:
                    pass

            walk(v)

    elif isinstance(x, list):

        for v in x:
            walk(v)


for p in files:

    try:

        if p.suffix.lower() == ".json":

            walk(
                json.loads(
                    p.read_text(
                        errors="ignore"
                    )
                )
            )

        elif p.suffix.lower() == ".jsonl":

            for line in p.read_text(
                errors="ignore"
            ).splitlines():

                try:
                    walk(
                        json.loads(line)
                    )
                except:
                    continue

        elif p.suffix.lower() == ".csv":

            with p.open(
                "r",
                encoding="utf-8",
                errors="ignore",
                newline="",
            ) as f:

                rr = csv.DictReader(f)

                for row in rr:

                    for k, v in row.items():

                        if (
                            real_rx.search(k)
                            or fake_rx.search(k)
                        ):

                            try:
                                fv = float(v)

                                if math.isfinite(fv):

                                    keys_found.setdefault(
                                        k,
                                        0,
                                    )

                                    keys_found[k] += 1
                                    numeric_count += 1

                            except:
                                pass

    except Exception:
        continue


result = {
    "files_scanned": len(files),
    "numeric_real_fake_score_values":
        numeric_count,
    "keys_found": keys_found,
}


outfile.write_text(
    json.dumps(
        result,
        indent=2,
    )
)


print(
    json.dumps(
        result,
        indent=2,
    )
)


if numeric_count == 0:
    raise SystemExit(40)

PYTOKEN


###############################################################################
# Q. SMOKE ORIGINAL X2DFD
###############################################################################

stage "Q. ORIGINAL X2DFD SMOKE INFERENCE"

cd "$X2"


CUDA_VISIBLE_DEVICES=0 \
 "$PY_X2" -u \
 -m eval.infer.runner \
 --config "$SMOKE_CFG_ORIG" \
 --model-path "$ORIG_LORA" \
 --model-base "$BASE_MODEL" \
 2>&1 \
 | tee "$LOGS/smoke_original_inference.log"


LATEST="$X2/eval/outputs/infer/latest_run.json"


[ -s "$LATEST" ] || \
    die "Original smoke latest_run.json missing"


"$PY_X2" \
 "$SNAPSHOT" \
 "$LATEST" \
 "$SNAP/smoke_original"


"$PY_X2" \
 "$TOKEN_CHECK" \
 "$SNAP/smoke_original" \
 "$SMOKE/original_token_score_gate.json"


echo "✅ ORIGINAL SMOKE + TOKEN SCORE PASS"


###############################################################################
# R. SMOKE ROUTER4
###############################################################################

stage "R. ROUTER4 FINAL SMOKE INFERENCE"

cd "$X2"


CUDA_VISIBLE_DEVICES=0 \
 "$PY_X2" -u \
 -m eval.infer.runner \
 --config "$SMOKE_CFG_R4" \
 --model-path "$FINAL_LORA" \
 --model-base "$BASE_MODEL" \
 2>&1 \
 | tee "$LOGS/smoke_router4_inference.log"


[ -s "$LATEST" ] || \
    die "Router4 smoke latest_run.json missing"


"$PY_X2" \
 "$SNAPSHOT" \
 "$LATEST" \
 "$SNAP/smoke_router4"


"$PY_X2" \
 "$TOKEN_CHECK" \
 "$SNAP/smoke_router4" \
 "$SMOKE/router4_token_score_gate.json"


echo "✅ ROUTER4 SMOKE + TOKEN SCORE PASS"


touch "$SMOKE/SMOKE_GATE_PASS"


###############################################################################
# S. RUN FULL FOUR EXPERTS
###############################################################################

stage "S1. FULL BLENDING"

cd "$X2"


CUDA_VISIBLE_DEVICES=0 \
 "$PY_X2" -u \
 "$RUN_X2" \
 --out "$WORK" \
 --expert blending \
 2>&1 \
 | tee "$LOGS/full_blending.log"


stage "S2. FULL DIFFUSION"

cd "$X2"


CUDA_VISIBLE_DEVICES=0 \
 "$PY_X2" -u \
 "$RUN_X2" \
 --out "$WORK" \
 --expert diffusion \
 2>&1 \
 | tee "$LOGS/full_diffusion.log"


stage "S3. FULL FREQUENCY"

cd "$DFF"


PYTHONPATH="$DFF:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=0 \
 "$PY_FREQ" -u \
 "$RUN_FREQ" \
 --out "$WORK" \
 --name PAPER_FINAL_3CROSS_V3_2 \
 2>&1 \
 | tee "$LOGS/full_frequency.log"


stage "S4. FULL TEXTURE"


CUDA_VISIBLE_DEVICES=0 \
 "$PY_TEX" -u \
 "$RUN_TEX" \
 --out "$WORK" \
 --work "$TEXWORK" \
 2>&1 \
 | tee "$LOGS/full_texture.log"


###############################################################################
# T. FULL EXPERT COVERAGE GATE
###############################################################################

stage "T. FULL FOUR-EXPERT COVERAGE GATE"

for E in \
 blending \
 diffusion \
 frequency \
 texture
do

    CSV="$WORK/${E}_scores.csv"

    [ -s "$CSV" ] || \
        die "Missing full expert CSV: $CSV"

    N=$(
        tail -n +2 "$CSV" \
        | wc -l
    )

    printf "%-12s %10d / %d\n" \
        "$E" \
        "$N" \
        "$EXPECTED"

    [ "$N" -eq "$EXPECTED" ] || \
        die "$E coverage mismatch: $N/$EXPECTED"

done


echo "✅ FULL EXPERT COVERAGE PASS"


###############################################################################
# U. WAIT FOR CONTENT-LEAKAGE HASH AUDIT
###############################################################################

stage "U. WAIT FOR CONTENT LEAKAGE AUDIT"

if ! wait "$HASH_PID"
then

    echo
    echo "===== DECODED RGB AUDIT LOG ====="

    tail -n 200 \
     "$HASH_LOG" \
     || true

    die \
      "Decoded-RGB train/test leakage gate failed"

fi


[ -s "$HASH_RESULT" ] || \
    die "Decoded-RGB leakage result missing"


cat "$HASH_RESULT"


touch "$PROTO/LEAKAGE_GATE_PASS"


echo "✅ TRAIN/TEST CONTENT LEAKAGE GATE PASS"


###############################################################################
# V. BUILD FULL ROUTER CACHE
###############################################################################

stage "V. BUILD FINAL ROUTER CACHE"

FULL_CACHE_RUN="$RUNROOT/build_router_cache_FINAL.py"


patch_cache_builder \
 "$FULL_CACHE_RUN" \
 "$WORK"


CUDA_VISIBLE_DEVICES=0 \
 "$PY_X2" -u \
 "$FULL_CACHE_RUN" \
 2>&1 \
 | tee "$LOGS/full_router_cache.log"


FULL_CACHE="$WORK/router_moe_cache.csv"


[ -s "$FULL_CACHE" ] || \
    die "Final Router cache missing"


CACHE_N=$(
    tail -n +2 \
    "$FULL_CACHE" \
    | wc -l
)


[ "$CACHE_N" -eq "$EXPECTED" ] || \
    die "Final Router cache coverage mismatch: $CACHE_N/$EXPECTED"


echo "✅ FINAL ROUTER CACHE PASS"


###############################################################################
# W. BUILD FULL LOCKED CONFIGS
###############################################################################

stage "W. BUILD FULL LOCKED CONFIGS"

FULL_CFG_ORIG="$CONFIGS/original_FINAL_LOCKED.yaml"
FULL_CFG_R4="$CONFIGS/router4_FINAL_LOCKED.yaml"


"$PY_X2" \
 "$PATCH_CFG" \
 "$BASE_CFG_ORIG" \
 "$FULL_CFG_ORIG" \
 "$WORK" \
 "$CELEB_JSON" \
 "$DFDCP_JSON" \
 "$DFDC_JSON"


"$PY_X2" \
 "$PATCH_CFG" \
 "$BASE_CFG_R4" \
 "$FULL_CFG_R4" \
 "$WORK" \
 "$CELEB_JSON" \
 "$DFDCP_JSON" \
 "$DFDC_JSON"


sha256sum \
 "$FULL_CFG_ORIG" \
 "$FULL_CFG_R4" \
 > "$PROTO/FINAL_CONFIGS.sha256"


###############################################################################
# X. FULL OFFICIAL ORIGINAL X2DFD
###############################################################################

stage "X. FULL OFFICIAL ORIGINAL X2DFD"

cd "$X2"


CUDA_VISIBLE_DEVICES=0 \
 "$PY_X2" -u \
 -m eval.infer.runner \
 --config "$FULL_CFG_ORIG" \
 --model-path "$ORIG_LORA" \
 --model-base "$BASE_MODEL" \
 2>&1 \
 | tee "$LOGS/full_original_x2dfd.log"


[ -s "$LATEST" ] || \
    die "Original full latest_run.json missing"


"$PY_X2" \
 "$SNAPSHOT" \
 "$LATEST" \
 "$SNAP/original"


"$PY_X2" \
 "$TOKEN_CHECK" \
 "$SNAP/original" \
 "$RUNROOT/original_token_score_gate.json"


touch "$SNAP/original/INFERENCE_DONE"


echo "✅ ORIGINAL FULL INFERENCE PASS"


###############################################################################
# Y. FULL ROUTER4 FINAL
###############################################################################

stage "Y. FULL ROUTER4 × X2DFD FINAL"

cd "$X2"


CUDA_VISIBLE_DEVICES=0 \
 "$PY_X2" -u \
 -m eval.infer.runner \
 --config "$FULL_CFG_R4" \
 --model-path "$FINAL_LORA" \
 --model-base "$BASE_MODEL" \
 2>&1 \
 | tee "$LOGS/full_router4_final.log"


[ -s "$LATEST" ] || \
    die "Router4 full latest_run.json missing"


"$PY_X2" \
 "$SNAPSHOT" \
 "$LATEST" \
 "$SNAP/router4"


"$PY_X2" \
 "$TOKEN_CHECK" \
 "$SNAP/router4" \
 "$RUNROOT/router4_token_score_gate.json"


touch "$SNAP/router4/INFERENCE_DONE"


echo "✅ ROUTER4 FULL INFERENCE PASS"


###############################################################################
# Z. FINAL FREEZE
###############################################################################

stage "Z. FINAL EXPERIMENT FREEZE"


sha256sum \
 "$FINAL_ROUTER" \
 "$FINAL_CAL" \
 "$FINAL_LORA/adapter_model.safetensors" \
 "$ORIG_LORA/adapter_model.safetensors" \
 "$MANIFEST" \
 "$LABELS" \
 "$LOCK" \
 "$FULL_CFG_ORIG" \
 "$FULL_CFG_R4" \
 "$FULL_CACHE" \
 > "$PROTO/FINAL_EXPERIMENT.sha256"


touch "$RUNROOT/PAPER_GATE_PASS"
touch "$RUNROOT/INFERENCE_COMPLETE"


stage "✅ PAPER-GRADE OVERNIGHT BENCHMARK COMPLETE"


echo
echo "============================================================"
echo "FINAL STATUS"
echo "============================================================"

echo "PROTOCOL:"
echo "$LOCK"

echo
echo "MANIFEST:"
echo "$MANIFEST"

echo
echo "MANIFEST HASHES:"
cat "$PROTO/MANIFEST_AND_PROTOCOL.sha256"

echo
echo "LEAKAGE:"
cat "$PROTO/exact_path_leakage.json"
cat "$PROTO/decoded_rgb_leakage.json"

echo
echo "ORIGINAL:"
cat "$RUNROOT/original_token_score_gate.json"

echo
echo "ROUTER4:"
cat "$RUNROOT/router4_token_score_gate.json"

echo
echo "READY FOR PAPER METRICS ✅"


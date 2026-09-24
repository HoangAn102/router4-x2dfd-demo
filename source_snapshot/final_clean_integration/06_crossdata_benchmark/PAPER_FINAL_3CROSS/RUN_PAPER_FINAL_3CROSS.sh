#!/usr/bin/env bash
set -Eeuo pipefail

BASE=/home/aiotlab/hoangan
ROOT="$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

RUNROOT="$ROOT/06_crossdata_benchmark/PAPER_FINAL_3CROSS"
WORK="$RUNROOT/work"
PROTO="$RUNROOT/protocol"
LOGS="$RUNROOT/logs"
SNAP="$RUNROOT/run_snapshots"

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

mkdir -p \
  "$WORK" \
  "$PROTO" \
  "$LOGS" \
  "$SNAP/original" \
  "$SNAP/router4" \
  "$RUNROOT/configs"

STOP="$RUNROOT/STOP_REASON.txt"
rm -f "$STOP"

stage () {
    echo
    echo "================================================================================"
    echo "$1"
    date
    echo "================================================================================"
}

die () {
    echo
    echo "[STOP] $1"
    printf '%s\n' "$1" > "$STOP"
    exit 1
}

trap '
rc=$?
echo "[ERROR] line=$LINENO exit=$rc command=$BASH_COMMAND"
echo "line=$LINENO exit=$rc command=$BASH_COMMAND" > "'"$STOP"'"
exit $rc
' ERR


################################################################################
# 0. HARD PREFLIGHT
################################################################################

stage "0. HARD PREFLIGHT"

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
 "$ORIG_LORA/adapter_model.safetensors" \
 "$BASE_MODEL/config.json" \
 "$INT/build_router_moe_cache_full.py" \
 "$INT/make_full_configs.py"
do
    [ -e "$P" ] || die "Missing required artifact: $P"
done

for D in \
 "$DATA/Celeb-DF-v2" \
 "$DATA/DFDC" \
 "$DATA/DFDCP"
do
    [ -d "$D" ] || die "Dataset missing: $D"
done

echo "✅ model artifacts"
echo "✅ expert runtimes"
echo "✅ all three extracted cross-datasets"


################################################################################
# 1. BUILD OFFICIAL TEST MANIFEST
#
# Celeb-DF-v2:
#   List_of_testing_videos.txt
#
# DFDCP:
#   dataset.json
#   only set == test
#
# DFDC:
#   test/labels.csv
#
# Absolutely NO middle frame / random sampling / resampling / duplication.
################################################################################

stage "1. BUILD OFFICIAL TEST MANIFEST"

MANIFEST="$WORK/manifest.csv"
LABELS="$PROTO/labels_and_video_ids.csv"
CELEB_JSON="$WORK/Celeb-DF-v2_test.json"
DFDCP_JSON="$WORK/DFDCP_test.json"
DFDC_JSON="$WORK/DFDC_test.json"

"$PY_X2" - \
 "$DATA" \
 "$MANIFEST" \
 "$LABELS" \
 "$CELEB_JSON" \
 "$DFDCP_JSON" \
 "$DFDC_JSON" \
 "$PROTO/PROTOCOL_LOCK.json" <<'PY'
from pathlib import Path
from collections import Counter
import csv
import json
import sys
import hashlib

(
    DATA,
    MANIFEST,
    LABELS,
    CELEB_JSON,
    DFDCP_JSON,
    DFDC_JSON,
    LOCK_JSON,
) = map(Path, sys.argv[1:])

IMG_EXT = {".png"}

records = []
missing_videos = []
protocol = {}


def images_in(d):
    if not d.is_dir():
        return []
    return sorted(
        p.resolve()
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )


def add(dataset, video_id, label, frames, source):
    if not frames:
        missing_videos.append(
            {
                "dataset": dataset,
                "video_id": video_id,
                "source": source,
            }
        )
        return

    for p in frames:
        records.append(
            {
                "dataset": dataset,
                "video_id": video_id,
                "label_text": label,
                "image_path": str(p),
                "source_metadata": source,
            }
        )


# =============================================================================
# CELEB-DF-v2
# =============================================================================

root = DATA / "Celeb-DF-v2"

test_list = root / "List_of_testing_videos.txt"

if not test_list.is_file():
    candidates = []
    for p in root.rglob("*.txt"):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        n = sum(".mp4" in x.lower() for x in txt.splitlines())
        if n:
            candidates.append((n, p))

    if not candidates:
        raise RuntimeError(
            "Celeb-DF-v2 official testing-video list not found"
        )

    candidates.sort(reverse=True, key=lambda x: x[0])
    test_list = candidates[0][1]

celeb_videos = 0

for line in test_list.read_text(errors="ignore").splitlines():
    line = line.strip()

    if not line or ".mp4" not in line.lower():
        continue

    toks = line.split()

    rel = next(
        (x for x in toks if x.lower().endswith(".mp4")),
        None
    )

    if rel is None:
        continue

    relp = Path(rel)
    category = relp.parent.name
    vid = relp.stem

    if category in {"Celeb-real", "YouTube-real"}:
        label = "real"
    elif category == "Celeb-synthesis":
        label = "fake"
    else:
        raise RuntimeError(
            f"Unknown Celeb category in official list: {line}"
        )

    frame_dir = root / category / "frames" / vid
    frames = images_in(frame_dir)

    add(
        "Celeb-DF-v2",
        vid,
        label,
        frames,
        f"{test_list}:{rel}",
    )

    celeb_videos += 1

if celeb_videos < 100:
    raise RuntimeError(
        f"Suspicious Celeb test-video count: {celeb_videos}"
    )


# =============================================================================
# DFDCP
# Official DeepfakeBench rearrangement:
# dataset.json -> label + set; only set=test.
# =============================================================================

root = DATA / "DFDCP"
meta = root / "dataset.json"

if not meta.is_file():
    candidates = list(root.rglob("dataset.json"))

    if len(candidates) != 1:
        raise RuntimeError(
            f"DFDCP dataset.json unresolved: {candidates}"
        )

    meta = candidates[0]

obj = json.loads(meta.read_text(errors="ignore"))

dfdcp_videos = 0

for key, info in obj.items():
    if not isinstance(info, dict):
        continue

    split = str(info.get("set", "")).lower()

    if split != "test":
        continue

    label = str(info.get("label", "")).lower()

    if label not in {"real", "fake"}:
        raise RuntimeError(
            f"Invalid DFDCP label: {key} -> {label}"
        )

    path = Path(key)
    index = path.parts[0]
    vid = path.stem

    frame_dir = root / index / "frames" / vid

    if not frame_dir.is_dir():
        # strict unique fallback, still only under a directory named frames
        hits = [
            p
            for p in root.rglob(vid)
            if p.is_dir()
            and p.parent.name == "frames"
        ]

        if len(hits) == 1:
            frame_dir = hits[0]
        else:
            raise RuntimeError(
                f"DFDCP frame directory ambiguous/missing: "
                f"{key} -> {hits}"
            )

    frames = images_in(frame_dir)

    add(
        "DFDCP",
        f"{index}/{vid}",
        label,
        frames,
        f"{meta}:{key}",
    )

    dfdcp_videos += 1

if dfdcp_videos < 10:
    raise RuntimeError(
        f"Suspicious DFDCP test-video count: {dfdcp_videos}"
    )


# =============================================================================
# DFDC
# Official DeepfakeBench rearrangement uses test/labels.csv.
# =============================================================================

root = DATA / "DFDC"

label_csv_candidates = [
    root / "test" / "labels.csv",
    root / "labels.csv",
]

label_csv = next(
    (p for p in label_csv_candidates if p.is_file()),
    None,
)

if label_csv is None:
    hits = list(root.rglob("labels.csv"))

    if len(hits) != 1:
        raise RuntimeError(
            f"DFDC labels.csv unresolved: {hits}"
        )

    label_csv = hits[0]

with label_csv.open(
    "r",
    encoding="utf-8",
    errors="ignore",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise RuntimeError("DFDC labels.csv empty")

fields = rows[0].keys()

filename_col = next(
    (
        c for c in fields
        if c.lower() in {"filename", "file", "video", "video_name"}
    ),
    None,
)

label_col = next(
    (
        c for c in fields
        if c.lower() in {"label", "class", "target"}
    ),
    None,
)

if filename_col is None or label_col is None:
    raise RuntimeError(
        f"DFDC labels schema unsupported: {list(fields)}"
    )

dfdc_videos = 0

for r in rows:
    filename = str(r[filename_col]).strip()
    raw_label = str(r[label_col]).strip().lower()

    if raw_label in {"real", "0", "false"}:
        label = "real"
    elif raw_label in {"fake", "1", "true"}:
        label = "fake"
    else:
        raise RuntimeError(
            f"Unrecognized DFDC label: {filename} -> {raw_label}"
        )

    vid = Path(filename).stem

    direct = label_csv.parent / "frames" / vid

    if direct.is_dir():
        frame_dir = direct
    else:
        hits = [
            p
            for p in root.rglob(vid)
            if p.is_dir()
            and p.parent.name == "frames"
        ]

        if len(hits) != 1:
            raise RuntimeError(
                f"DFDC frame directory ambiguous/missing: "
                f"{filename} -> {hits[:10]}"
            )

        frame_dir = hits[0]

    frames = images_in(frame_dir)

    add(
        "DFDC",
        vid,
        label,
        frames,
        f"{label_csv}:{filename}",
    )

    dfdc_videos += 1

if dfdc_videos < 10:
    raise RuntimeError(
        f"Suspicious DFDC test-video count: {dfdc_videos}"
    )


# =============================================================================
# GLOBAL GATES
# =============================================================================

if not records:
    raise RuntimeError("Empty benchmark manifest")

seen = {}
duplicates = []

for r in records:
    p = r["image_path"]

    if p in seen:
        duplicates.append((p, seen[p], r))
    else:
        seen[p] = r

if duplicates:
    raise RuntimeError(
        f"Duplicate image paths in benchmark: {len(duplicates)}"
    )

for r in records:
    if not Path(r["image_path"]).is_file():
        raise RuntimeError(
            f"Missing image: {r['image_path']}"
        )

counts_images = Counter(r["dataset"] for r in records)
video_sets = {}

for ds in ["Celeb-DF-v2", "DFDCP", "DFDC"]:
    vids = {
        r["video_id"]
        for r in records
        if r["dataset"] == ds
    }

    video_sets[ds] = len(vids)

    if not vids:
        raise RuntimeError(
            f"No test videos for {ds}"
        )

frame_distribution = {}

for ds in ["Celeb-DF-v2", "DFDCP", "DFDC"]:
    by_video = Counter()

    for r in records:
        if r["dataset"] == ds:
            by_video[r["video_id"]] += 1

    dist = Counter(by_video.values())

    frame_distribution[ds] = {
        str(k): v
        for k, v in sorted(dist.items())
    }


# =============================================================================
# WRITE THREE X2DFD INPUT JSONS
# =============================================================================

json_map = {
    "Celeb-DF-v2": CELEB_JSON,
    "DFDCP": DFDCP_JSON,
    "DFDC": DFDC_JSON,
}

for ds, out in json_map.items():
    imgs = [
        {"image_path": r["image_path"]}
        for r in records
        if r["dataset"] == ds
    ]

    payload = {
        "Description": str(DATA / ds),
        "images": imgs,
    }

    out.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# WRITE EXPERT MANIFEST
# Keep json field because old integration tooling uses provenance JSON.
# =============================================================================

with MANIFEST.open(
    "w",
    encoding="utf-8",
    newline=""
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
        w.writerow(
            {
                "json": str(json_map[r["dataset"]]),
                "image_path": r["image_path"],
                "dataset": r["dataset"],
                "video_id": r["video_id"],
                "label_text": r["label_text"],
            }
        )


with LABELS.open(
    "w",
    encoding="utf-8",
    newline=""
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


lock = {
    "protocol": (
        "DeepfakeBench-style official test metadata; "
        "use all locally available preprocessed PNG frames "
        "for every official test video; no resampling"
    ),
    "no_middle_frame": True,
    "no_random_sampling": True,
    "no_duplicate_to_32": True,
    "datasets": {
        ds: {
            "frames": counts_images[ds],
            "videos": video_sets[ds],
            "frames_per_video_distribution":
                frame_distribution[ds],
        }
        for ds in [
            "Celeb-DF-v2",
            "DFDCP",
            "DFDC",
        ]
    },
    "missing_zero_frame_videos": missing_videos,
    "sources": {
        "Celeb-DF-v2": str(test_list),
        "DFDCP": str(meta),
        "DFDC": str(label_csv),
    },
}

LOCK_JSON.write_text(
    json.dumps(lock, indent=2),
    encoding="utf-8",
)

print(json.dumps(lock, indent=2))

print()
print("TOTAL FRAMES =", len(records))
print("MISSING/ZERO-FRAME VIDEOS =", len(missing_videos))

if missing_videos:
    print(
        "NOTE: zero-frame videos are recorded in PROTOCOL_LOCK.json; "
        "no artificial replacement/duplication was performed."
    )
PY


################################################################################
# 2. FREEZE MANIFEST + RECORD SHA256
################################################################################

stage "2. FREEZE MANIFEST"

sha256sum "$MANIFEST" \
  > "$PROTO/manifest.sha256"

sha256sum "$LABELS" \
  > "$PROTO/labels_and_video_ids.sha256"

cat "$PROTO/manifest.sha256"
cat "$PROTO/labels_and_video_ids.sha256"

EXPECTED=$(
  tail -n +2 "$MANIFEST" | wc -l
)

[ "$EXPECTED" -gt 0 ] || die "Empty manifest"

echo "LOCKED FRAMES = $EXPECTED"


################################################################################
# 3. EXACT-PATH LEAKAGE GATE
################################################################################

stage "3. EXACT PATH TRAIN/TEST LEAKAGE GATE"

"$PY_X2" - \
 "$TRAIN_TEACHER" \
 "$MANIFEST" \
 "$PROTO/exact_path_leakage.json" <<'PY'
from pathlib import Path
import csv
import json
import sys

train_csv = Path(sys.argv[1])
test_csv = Path(sys.argv[2])
out = Path(sys.argv[3])


def load_paths(path):
    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
        newline=""
    ) as f:
        r = csv.DictReader(f)

        fields = r.fieldnames or []

        c = next(
            (
                x for x in [
                    "image_path",
                    "image",
                    "path",
                ]
                if x in fields
            ),
            None,
        )

        if c is None:
            raise RuntimeError(
                f"No image-path column in {path}: {fields}"
            )

        result = set()

        for row in r:
            p = str(row[c]).strip()

            if p:
                try:
                    p = str(Path(p).resolve())
                except Exception:
                    pass

                result.add(p)

        return result


train = load_paths(train_csv)
test = load_paths(test_csv)

overlap = train & test

result = {
    "train_unique_paths": len(train),
    "test_unique_paths": len(test),
    "exact_path_overlap": len(overlap),
    "examples": sorted(overlap)[:100],
}

out.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print(json.dumps(result, indent=2))

if overlap:
    raise SystemExit(20)
PY

echo "✅ exact-path overlap = 0"


################################################################################
# 4. START DECODED-RGB SHA256 LEAKAGE AUDIT IN BACKGROUND
#
# This catches same decoded image content even when file encoding differs.
################################################################################

stage "4. START DECODED-RGB CONTENT AUDIT"

HASH_SCRIPT="$RUNROOT/decoded_rgb_leak_audit.py"

cat > "$HASH_SCRIPT" <<'PY'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import sys

from PIL import Image

train_csv = Path(sys.argv[1])
test_csv = Path(sys.argv[2])
out = Path(sys.argv[3])


def get_paths(path):
    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
        newline=""
    ) as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []

        c = next(
            (
                x for x in [
                    "image_path",
                    "image",
                    "path",
                ]
                if x in fields
            ),
            None,
        )

        if c is None:
            raise RuntimeError(
                f"No image column in {path}"
            )

        arr = []

        for row in r:
            p = str(row[c]).strip()

            if p:
                arr.append(p)

    return sorted(set(arr))


def rgb_hash(p):
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")

            h = hashlib.sha256()
            h.update(
                f"{im.width}x{im.height}|RGB|".encode()
            )
            h.update(im.tobytes())

        return p, h.hexdigest(), None

    except Exception as e:
        return p, None, repr(e)


train = get_paths(train_csv)
test = get_paths(test_csv)

print("TRAIN =", len(train), flush=True)
print("TEST  =", len(test), flush=True)

workers = 8

def compute(paths, label):
    hashes = {}
    errors = []

    with ThreadPoolExecutor(
        max_workers=workers
    ) as ex:
        for i, (p, h, err) in enumerate(
            ex.map(rgb_hash, paths),
            1
        ):
            if err:
                errors.append(
                    {"path": p, "error": err}
                )
            else:
                hashes.setdefault(h, []).append(p)

            if i % 5000 == 0:
                print(
                    label,
                    i,
                    "/",
                    len(paths),
                    flush=True,
                )

    return hashes, errors


tr, tr_err = compute(train, "TRAIN")
te, te_err = compute(test, "TEST")

common = set(tr) & set(te)

examples = []

for h in list(common)[:100]:
    examples.append(
        {
            "hash": h,
            "train": tr[h][:5],
            "test": te[h][:5],
        }
    )

result = {
    "train_files": len(train),
    "test_files": len(test),
    "train_decode_errors": len(tr_err),
    "test_decode_errors": len(te_err),
    "decoded_rgb_hash_overlap": len(common),
    "examples": examples,
    "train_error_examples": tr_err[:50],
    "test_error_examples": te_err[:50],
}

out.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print(json.dumps(result, indent=2))

if tr_err or te_err or common:
    raise SystemExit(30)
PY

HASH_LOG="$LOGS/decoded_rgb_leak_audit.log"
HASH_JSON="$PROTO/decoded_rgb_leakage.json"

if command -v ionice >/dev/null 2>&1; then
    ionice -c2 -n7 \
    nice -n 10 \
    "$PY_X2" -u \
    "$HASH_SCRIPT" \
    "$TRAIN_TEACHER" \
    "$MANIFEST" \
    "$HASH_JSON" \
    > "$HASH_LOG" 2>&1 &
else
    nice -n 10 \
    "$PY_X2" -u \
    "$HASH_SCRIPT" \
    "$TRAIN_TEACHER" \
    "$MANIFEST" \
    "$HASH_JSON" \
    > "$HASH_LOG" 2>&1 &
fi

HASH_PID=$!

echo "content-audit PID = $HASH_PID"
echo "content-audit log = $HASH_LOG"


################################################################################
# 5. FOUR EXPERTS
# CSV scripts support resume.
################################################################################

stage "5A. BLENDING"

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
"$RUN_X2" \
--out "$WORK" \
--expert blending \
2>&1 | tee "$LOGS/blending.log"


stage "5B. DIFFUSION"

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
"$RUN_X2" \
--out "$WORK" \
--expert diffusion \
2>&1 | tee "$LOGS/diffusion.log"


stage "5C. FREQUENCY"

cd "$DFF"

PYTHONPATH="$DFF:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=0 \
"$PY_FREQ" -u \
"$RUN_FREQ" \
--out "$WORK" \
--name PAPER_FINAL_3CROSS \
2>&1 | tee "$LOGS/frequency.log"


stage "5D. TEXTURE"

CUDA_VISIBLE_DEVICES=0 \
"$PY_TEX" -u \
"$RUN_TEX" \
--out "$WORK" \
--work "$TEXWORK" \
2>&1 | tee "$LOGS/texture.log"


################################################################################
# 6. EXPERT COVERAGE GATE
################################################################################

stage "6. FOUR-EXPERT COVERAGE GATE"

for E in blending diffusion frequency texture
do
    CSV="$WORK/${E}_scores.csv"

    [ -s "$CSV" ] || \
      die "Missing expert CSV: $CSV"

    N=$(
        tail -n +2 "$CSV" | wc -l
    )

    printf "%-12s %10d / %d\n" \
        "$E" "$N" "$EXPECTED"

    [ "$N" -eq "$EXPECTED" ] || \
      die "$E coverage mismatch: $N/$EXPECTED"
done

echo "✅ identical expert coverage"


################################################################################
# 7. WAIT FOR DECODED-RGB LEAKAGE AUDIT
################################################################################

stage "7. CONTENT-LEAKAGE GATE"

if ! wait "$HASH_PID"; then
    echo
    echo "===== HASH AUDIT LOG ====="
    tail -n 120 "$HASH_LOG" || true

    die \
      "Decoded-RGB leakage audit failed. " \
      "Inference intentionally blocked."
fi

[ -s "$HASH_JSON" ] || \
  die "Decoded-RGB audit result missing"

cat "$HASH_JSON"

touch "$PROTO/LEAKAGE_GATE_PASS"

echo "✅ decoded RGB overlap = 0"


################################################################################
# 8. BUILD ROUTER4 FINAL CACHE
################################################################################

stage "8. BUILD FINAL ROUTER CACHE"

CACHE_SRC="$INT/build_router_moe_cache_full.py"
CACHE_RUN="$RUNROOT/build_router_moe_cache_FINAL.py"

cp -f "$CACHE_SRC" "$CACHE_RUN"

"$PY_X2" - \
 "$CACHE_RUN" \
 "$WORK" \
 "$FINAL_ROUTER" \
 "$FINAL_CAL" \
 "$TRAIN_TEACHER" \
 "$VAL_TEACHER" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])

work = sys.argv[2]
router = sys.argv[3]
cal = sys.argv[4]
train_teacher = sys.argv[5]
val_teacher = sys.argv[6]

s = p.read_text(errors="ignore")

repls = {
    "outputs/x2dfd_router_integration/full_DFDCP_DFDC":
        work,

    "outputs/router_training_v2/run_effb0_3domain/best.pt":
        router,

    "outputs/router_teacher_v2/calibrators_train_only.joblib":
        cal,

    "outputs/router_teacher_v2/train_teacher_v2.csv":
        train_teacher,

    "outputs/router_teacher_v2/val_teacher_v2.csv":
        val_teacher,
}

for old, new in repls.items():
    s = s.replace(old, new)

p.write_text(s)

print("PATCHED:", p)
PY

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
"$CACHE_RUN" \
2>&1 | tee "$LOGS/router_cache.log"

CACHE="$WORK/router_moe_cache.csv"

[ -s "$CACHE" ] || \
  die "Router cache not produced"

N_CACHE=$(
    tail -n +2 "$CACHE" | wc -l
)

[ "$N_CACHE" -eq "$EXPECTED" ] || \
  die "Router cache coverage mismatch: $N_CACHE/$EXPECTED"

echo "✅ router cache = $N_CACHE"


################################################################################
# 9. BUILD INFERENCE CONFIGS WITHOUT KEEPING REPO MODIFICATIONS
################################################################################

stage "9. BUILD LOCKED INFERENCE CONFIGS"

MAKE_SRC="$INT/make_full_configs.py"
MAKE_RUN="$RUNROOT/make_full_configs_FINAL.py"

cp -f "$MAKE_SRC" "$MAKE_RUN"

"$PY_X2" - \
 "$MAKE_RUN" \
 "$WORK" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
work = sys.argv[2]

s = p.read_text(errors="ignore")

s = s.replace(
    "outputs/x2dfd_router_integration/full_DFDCP_DFDC",
    work,
)

p.write_text(s)
PY

CFG_ORIG_REPO="$X2/eval/configs/infer_full_DFDCP_DFDC_original.yaml"
CFG_R4_REPO="$X2/eval/configs/infer_full_DFDCP_DFDC_router_moe.yaml"

[ -f "$CFG_ORIG_REPO" ] && \
 cp -f "$CFG_ORIG_REPO" "$RUNROOT/configs/original.before.yaml" || true

[ -f "$CFG_R4_REPO" ] && \
 cp -f "$CFG_R4_REPO" "$RUNROOT/configs/router4.before.yaml" || true

cd "$INT"

"$PY_X2" \
 "$MAKE_RUN" \
2>&1 | tee "$LOGS/config_builder.log"

[ -s "$CFG_ORIG_REPO" ] || \
 die "Original config generator failed"

[ -s "$CFG_R4_REPO" ] || \
 die "Router config generator failed"

CFG_ORIG="$RUNROOT/configs/original.LOCKED.yaml"
CFG_R4="$RUNROOT/configs/router4.LOCKED.yaml"

cp -f "$CFG_ORIG_REPO" "$CFG_ORIG"
cp -f "$CFG_R4_REPO" "$CFG_R4"


################################################################################
# Force both configs to use EXACT SAME three JSON inputs.
# Also recursively rewrite stale work paths.
################################################################################

"$PY_X2" - \
 "$CFG_ORIG" \
 "$CFG_R4" \
 "$WORK" \
 "$CELEB_JSON" \
 "$DFDCP_JSON" \
 "$DFDC_JSON" <<'PY'
from pathlib import Path
import sys
import yaml

orig = Path(sys.argv[1])
r4 = Path(sys.argv[2])
work = sys.argv[3]

inputs = [
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
]


def rewrite(x):
    if isinstance(x, dict):
        return {
            k: rewrite(v)
            for k, v in x.items()
        }

    if isinstance(x, list):
        return [
            rewrite(v)
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


for p in [orig, r4]:
    cfg = yaml.safe_load(
        p.read_text()
    )

    cfg = rewrite(cfg)

    if "infer" not in cfg:
        raise RuntimeError(
            f"No infer block in {p}"
        )

    cfg["infer"]["inputs"] = inputs

    p.write_text(
        yaml.safe_dump(
            cfg,
            sort_keys=False,
        )
    )

    print("LOCKED:", p)
    print("inputs:", inputs)
PY

sha256sum "$CFG_ORIG" "$CFG_R4" \
 > "$PROTO/inference_configs.sha256"


################################################################################
# 10. OFFICIAL ORIGINAL X2DFD
################################################################################

stage "10. OFFICIAL ORIGINAL X2DFD — FULL 3 CROSS DATA"

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
-m eval.infer.runner \
--config "$CFG_ORIG" \
--model-path "$ORIG_LORA" \
--model-base "$BASE_MODEL" \
--experts blending,diffusion_detector \
2>&1 | tee "$LOGS/original_x2dfd.log"

[ -s "$X2/eval/outputs/infer/latest_run.json" ] || \
 die "Original latest_run.json missing"

cp -f \
 "$X2/eval/outputs/infer/latest_run.json" \
 "$SNAP/original/latest_run.json"

touch "$SNAP/original/INFERENCE_DONE"


################################################################################
# 11. ROUTER4 FINAL
################################################################################

stage "11. ROUTER4 × X2DFD FINAL — FULL 3 CROSS DATA"

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
-m eval.infer.runner \
--config "$CFG_R4" \
--model-path "$FINAL_LORA" \
--model-base "$BASE_MODEL" \
2>&1 | tee "$LOGS/router4_final.log"

[ -s "$X2/eval/outputs/infer/latest_run.json" ] || \
 die "Router4 latest_run.json missing"

cp -f \
 "$X2/eval/outputs/infer/latest_run.json" \
 "$SNAP/router4/latest_run.json"

touch "$SNAP/router4/INFERENCE_DONE"


################################################################################
# 12. TOKEN-SCORE CONTRACT CHECK
#
# Do NOT calculate metrics here.
# We only verify that continuous REAL/FAKE scores exist somewhere
# in output artifacts referenced by each run.
################################################################################

stage "12. TOKEN-SCORE OUTPUT CONTRACT"

"$PY_X2" - \
 "$SNAP/original/latest_run.json" \
 "$SNAP/router4/latest_run.json" \
 "$RUNROOT/TOKEN_SCORE_GATE.json" <<'PY'
from pathlib import Path
import json
import re
import sys

pointers = {
    "original": Path(sys.argv[1]),
    "router4": Path(sys.argv[2]),
}

out = Path(sys.argv[3])

result = {}


def collect_strings(x):
    arr = []

    if isinstance(x, dict):
        for v in x.values():
            arr.extend(
                collect_strings(v)
            )

    elif isinstance(x, list):
        for v in x:
            arr.extend(
                collect_strings(v)
            )

    elif isinstance(x, str):
        arr.append(x)

    return arr


for name, ptr in pointers.items():
    obj = json.loads(
        ptr.read_text(errors="ignore")
    )

    candidates = []

    for s in collect_strings(obj):
        p = Path(s)

        if p.is_file() and p.suffix.lower() in {
            ".json",
            ".jsonl",
        }:
            candidates.append(p)

    candidates = list(dict.fromkeys(candidates))

    real_found = False
    fake_found = False

    scanned = []

    for p in candidates[:100]:
        try:
            txt = p.read_text(
                errors="ignore"
            )
        except:
            continue

        # enough for contract check
        sample = txt[:8_000_000]

        real = bool(
            re.search(
                r'real[\s_\-]*score',
                sample,
                re.I,
            )
        )

        fake = bool(
            re.search(
                r'fake[\s_\-]*score',
                sample,
                re.I,
            )
        )

        scanned.append(
            {
                "file": str(p),
                "real_score": real,
                "fake_score": fake,
            }
        )

        real_found |= real
        fake_found |= fake

        if real_found and fake_found:
            break

    result[name] = {
        "real_score_found": real_found,
        "fake_score_found": fake_found,
        "candidate_outputs": len(candidates),
        "scanned": scanned,
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
    )
)

# Important:
# Do not destroy expensive inference results if contract discovery fails.
# Just record the status for metric parsing tomorrow.
PY


################################################################################
# 13. FINAL ARTIFACT FINGERPRINT
################################################################################

stage "13. FREEZE EXPERIMENT PROVENANCE"

{
    sha256sum \
      "$FINAL_ROUTER" \
      "$FINAL_CAL" \
      "$FINAL_LORA/adapter_model.safetensors" \
      "$ORIG_LORA/adapter_model.safetensors" \
      "$MANIFEST" \
      "$LABELS" \
      "$CFG_ORIG" \
      "$CFG_R4"
} > "$PROTO/FINAL_EXPERIMENT_ARTIFACTS.sha256"

touch "$RUNROOT/INFERENCE_COMPLETE"

stage "✅ OVERNIGHT PAPER BENCHMARK INFERENCE COMPLETE"

echo
echo "Protocol:"
echo "$PROTO/PROTOCOL_LOCK.json"

echo
echo "Manifest hash:"
cat "$PROTO/manifest.sha256"

echo
echo "Original run:"
echo "$SNAP/original/latest_run.json"

echo
echo "Router4 run:"
echo "$SNAP/router4/latest_run.json"

echo
echo "Token gate:"
echo "$RUNROOT/TOKEN_SCORE_GATE.json"

echo
echo "NEXT:"
echo "Compute frame-level and video-level paper metrics from frozen token scores."

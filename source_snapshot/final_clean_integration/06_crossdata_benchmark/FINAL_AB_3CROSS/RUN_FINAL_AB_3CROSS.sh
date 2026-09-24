#!/usr/bin/env bash
set -euo pipefail

BASE=/home/aiotlab/hoangan
ROOT="$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

X2="$BASE/projects/X2DFD"
INT="$BASE/projects/x2dfd_router_integration"

BENCH="$ROOT/06_crossdata_benchmark/FINAL_AB_3CROSS"
WORK="$BENCH/work"

PY_X2=/home/aiotlab/miniconda3/envs/X2DFD/bin/python
PY_FREQ=/home/aiotlab/miniconda3/envs/DFFreq/bin/python
PY_TEX=/home/aiotlab/miniconda3/envs/TextureExpert/bin/python

RUN_X2="$BASE/router8_x2crop/run_x2.py"
RUN_FREQ="$BASE/router8_x2crop/run_frequency.py"
RUN_TEX="$BASE/router8_x2crop/run_texture.py"

DFF="$BASE/projects/DFFreq-main"
TEX="$BASE/projects/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild/stylegan-ffhq"

FINAL_ROUTER="$ROOT/02_router/final_checkpoint/best.pt"
FINAL_CAL="$ROOT/01_teacher/calibrators_FINAL_CLEAN.joblib"
FINAL_TRAIN_TEACHER="$ROOT/01_teacher/train_teacher_FINAL_CLEAN.csv"
FINAL_VAL_TEACHER="$ROOT/01_teacher/val_teacher_FINAL_CLEAN.csv"
FINAL_LORA="$ROOT/05_lora/router4_x2dfd_FINAL_CLEAN"

ORIGINAL_LORA="$X2/weights/_official_extract/weight/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]"
BASE_MODEL="$X2/weights/base/llava-v1.5-7b"

OLD_OUT="$BASE/outputs/x2dfd_router_integration/full_DFDCP_DFDC"
OLD_MANIFEST="$OLD_OUT/manifest.csv"

CELEB="$BASE/datasets/CROSSDATA_FINAL_EVAL/Celeb-DF-v2"

mkdir -p \
  "$WORK" \
  "$BENCH/raw/original" \
  "$BENCH/raw/router4" \
  "$BENCH/configs" \
  "$BENCH/results"

log () {
    echo
    echo "======================================================================"
    echo "$1"
    date
    echo "======================================================================"
}

fail () {
    echo "[STOP] $1"
    echo "$1" > "$BENCH/STOP_REASON.txt"
    exit 1
}

###############################################################################
# 1. HARD PREFLIGHT
###############################################################################

log "1. HARD PREFLIGHT"

for F in \
 "$PY_X2" \
 "$PY_FREQ" \
 "$PY_TEX" \
 "$RUN_X2" \
 "$RUN_FREQ" \
 "$RUN_TEX" \
 "$FINAL_ROUTER" \
 "$FINAL_CAL" \
 "$FINAL_LORA/adapter_model.safetensors" \
 "$ORIGINAL_LORA/adapter_model.safetensors" \
 "$OLD_MANIFEST"
do
    [ -e "$F" ] || fail "Missing: $F"
done

[ -d "$CELEB" ] || fail "Missing Celeb-DF-v2: $CELEB"

echo "✅ final Router"
echo "✅ final LoRA"
echo "✅ original official X2DFD"
echo "✅ expert runtimes"
echo "✅ canonical DFDCP+DFDC manifest"

###############################################################################
# 2. BUILD ONE LOCKED MANIFEST
#
# DFDCP + DFDC:
#     preserve EXACT old canonical integration manifest.
#
# Celeb-DF-v2:
#     official test-video list only,
#     one deterministic middle frame per test video.
#
# This is an internal locked A/B protocol.
# Do NOT claim identical to a published frame-sampling protocol unless verified.
###############################################################################

log "2. BUILD LOCKED 3-DATASET MANIFEST"

"$PY_X2" \
 "$OLD_MANIFEST" \
 "$CELEB" \
 "$WORK/manifest.csv" \
 "$BENCH/benchmark_manifest.csv" <<'PY'
import csv
import os
import sys
from pathlib import Path
from collections import Counter

old_manifest = Path(sys.argv[1])
celeb_root   = Path(sys.argv[2])
out_manifest = Path(sys.argv[3])
audit_csv    = Path(sys.argv[4])

with old_manifest.open("r", encoding="utf-8", errors="ignore", newline="") as f:
    reader = csv.DictReader(f)
    old_rows = list(reader)
    fields = list(reader.fieldnames or [])

if not old_rows:
    raise RuntimeError("Old DFDCP+DFDC manifest is empty")

image_key = next(
    (x for x in ["image_path", "image", "path"] if x in fields),
    None
)
if not image_key:
    raise RuntimeError(f"No image path column in {fields}")

label_key = next(
    (x for x in ["label", "ground_truth_label", "target", "class"] if x in fields),
    None
)

if label_key is None:
    label_key = "label"
    fields.append(label_key)

dataset_key = next(
    (x for x in ["dataset", "domain", "dataset_name"] if x in fields),
    None
)

if dataset_key is None:
    dataset_key = "dataset"
    fields.append(dataset_key)

def infer_text_label(path):
    s = str(path).lower().replace("\\", "/")
    if (
        "youtube-real" in s or
        "celeb-real" in s or
        "/real/" in s or
        "/0_real/" in s
    ):
        return "real"
    if (
        "celeb-synthesis" in s or
        "/fake/" in s or
        "/1_fake/" in s
    ):
        return "fake"
    return None

# Learn old label representation from paths instead of assuming 0/1 semantics.
real_vals = Counter()
fake_vals = Counter()

for r in old_rows:
    t = infer_text_label(r.get(image_key, ""))
    v = str(r.get(label_key, "")).strip()

    if not v:
        continue

    if t == "real":
        real_vals[v] += 1
    elif t == "fake":
        fake_vals[v] += 1

real_encoded = real_vals.most_common(1)[0][0] if real_vals else "real"
fake_encoded = fake_vals.most_common(1)[0][0] if fake_vals else "fake"

print("IMAGE COLUMN =", image_key)
print("LABEL COLUMN =", label_key)
print("DATASET COLUMN =", dataset_key)
print("REAL ENCODING =", real_encoded)
print("FAKE ENCODING =", fake_encoded)

# Preserve canonical DFDCP + DFDC rows.
combined = []

for r in old_rows:
    rr = dict(r)

    if not str(rr.get(dataset_key, "")).strip():
        s = str(rr.get(image_key, "")).lower()
        if "dfdcp" in s:
            rr[dataset_key] = "DFDCP"
        elif "dfdc" in s:
            rr[dataset_key] = "DFDC"

    combined.append(rr)

# Find official Celeb testing-video list.
test_lists = []

for p in celeb_root.rglob("*.txt"):
    low = p.name.lower()
    if "test" in low and "video" in low:
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue

        video_lines = [
            x for x in txt.splitlines()
            if ".mp4" in x.lower()
        ]

        if video_lines:
            test_lists.append((len(video_lines), p, video_lines))

if not test_lists:
    raise RuntimeError(
        "Celeb-DF-v2 official testing-video list not found. "
        "Refusing to invent a test split."
    )

test_lists.sort(reverse=True, key=lambda x: x[0])
_, test_list, lines = test_lists[0]

print("CELEB TEST LIST =", test_list)
print("CELEB TEST VIDEOS =", len(lines))

# Build frame-directory index once.
frame_index = {}

for cur, dirs, files in os.walk(celeb_root):
    p = Path(cur)

    if p.parent.name.lower() == "frames":
        category = p.parent.parent.name.lower()
        frame_index[(category, p.name)] = p

print("FRAME DIRECTORIES =", len(frame_index))

celeb_rows = []
audit_rows = []

for line in lines:
    toks = line.strip().split()

    video_token = next(
        (x for x in reversed(toks) if ".mp4" in x.lower()),
        None
    )

    if not video_token:
        continue

    vp = Path(video_token)
    category = vp.parent.name.lower()
    stem = vp.stem

    text_label = infer_text_label(video_token)

    if text_label not in {"real", "fake"}:
        raise RuntimeError(
            f"Ambiguous Celeb label: {video_token}"
        )

    d = frame_index.get((category, stem))

    # Conservative fallback: unique frame dir with same stem.
    if d is None:
        matches = [
            path
            for (cat, st), path in frame_index.items()
            if st == stem
        ]

        if len(matches) == 1:
            d = matches[0]

    if d is None:
        raise RuntimeError(
            f"Frame directory not found for official test video: {video_token}"
        )

    imgs = sorted(
        p for p in d.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if not imgs:
        raise RuntimeError(f"No frames: {d}")

    # One deterministic middle frame per official test video.
    img = imgs[len(imgs) // 2].resolve()

    row = {k: "" for k in fields}
    row[image_key] = str(img)
    row[label_key] = real_encoded if text_label == "real" else fake_encoded
    row[dataset_key] = "Celeb-DF-v2"

    combined.append(row)
    celeb_rows.append(row)

    audit_rows.append({
        "image_path": str(img),
        "label_text": text_label,
        "dataset": "Celeb-DF-v2",
        "source_video": video_token,
    })

# Build normalized audit for ALL rows.
for r in old_rows:
    p = str(r.get(image_key, ""))
    txt = infer_text_label(p)

    if txt is None:
        enc = str(r.get(label_key, "")).strip()
        if enc == real_encoded:
            txt = "real"
        elif enc == fake_encoded:
            txt = "fake"

    ds = str(r.get(dataset_key, "")).strip()

    if not ds:
        s = p.lower()
        if "dfdcp" in s:
            ds = "DFDCP"
        elif "dfdc" in s:
            ds = "DFDC"
        else:
            ds = "UNKNOWN"

    audit_rows.append({
        "image_path": str(Path(p).resolve()),
        "label_text": txt or "UNKNOWN",
        "dataset": ds,
        "source_video": "",
    })

out_manifest.parent.mkdir(parents=True, exist_ok=True)

with out_manifest.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(combined)

# De-duplicate audit by image.
seen = {}
for r in audit_rows:
    seen[r["image_path"]] = r

with audit_csv.open("w", encoding="utf-8", newline="") as f:
    keys = ["image_path", "label_text", "dataset", "source_video"]
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(seen.values())

counts = Counter(
    str(r.get(dataset_key, "")).strip()
    for r in combined
)

print("TOTAL =", len(combined))
print("CELEB ADDED =", len(celeb_rows))
print("DATASETS =", dict(counts))

if len(celeb_rows) < 100:
    raise RuntimeError(
        f"Suspicious Celeb test size: {len(celeb_rows)}"
    )
PY

echo
echo "Manifest rows:"
tail -n +2 "$WORK/manifest.csv" | wc -l

###############################################################################
# 3. FOUR EXPERTS
###############################################################################

log "3A. EXPERT — BLENDING"

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u "$RUN_X2" \
  --out "$WORK" \
  --expert blending


log "3B. EXPERT — DIFFUSION"

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u "$RUN_X2" \
  --out "$WORK" \
  --expert diffusion


log "3C. EXPERT — FREQUENCY"

cd "$DFF"

PYTHONPATH="$DFF:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=0 \
"$PY_FREQ" -u "$RUN_FREQ" \
  --out "$WORK" \
  --name ROUTER4_FINAL_3CROSS


log "3D. EXPERT — TEXTURE"

CUDA_VISIBLE_DEVICES=0 \
"$PY_TEX" -u "$RUN_TEX" \
  --out "$WORK" \
  --work "$TEX"


###############################################################################
# 4. HARD VERIFY EXPERT COVERAGE
###############################################################################

log "4. VERIFY FOUR EXPERT CSVs"

EXPECTED=$(tail -n +2 "$WORK/manifest.csv" | wc -l)

for E in blending diffusion frequency texture
do
    CSV="$WORK/${E}_scores.csv"

    [ -s "$CSV" ] || fail "Missing $CSV"

    N=$(tail -n +2 "$CSV" | wc -l)

    printf "%-10s %8d / %d\n" "$E" "$N" "$EXPECTED"

    [ "$N" -eq "$EXPECTED" ] || \
      fail "$E coverage mismatch: $N/$EXPECTED"
done

echo "✅ all four experts cover identical manifest"

###############################################################################
# 5. PATCH A COPY OF ROUTER CACHE BUILDER
###############################################################################

log "5. BUILD FINAL ROUTER CACHE"

CACHE_SRC="$INT/build_router_moe_cache_full.py"
CACHE_FINAL="$BENCH/build_router_moe_cache_FINAL.py"

cp -f "$CACHE_SRC" "$CACHE_FINAL"

"$PY_X2" \
 "$CACHE_FINAL" \
 "$WORK" \
 "$FINAL_ROUTER" \
 "$FINAL_CAL" \
 "$FINAL_TRAIN_TEACHER" \
 "$FINAL_VAL_TEACHER" <<'PY'
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
    "outputs/x2dfd_router_integration/full_DFDCP_DFDC": work,

    "outputs/router_training_v2/run_effb0_3domain/best.pt":
        router,

    "outputs/router_teacher_v2/calibrators_train_only.joblib":
        cal,

    "outputs/router_teacher_v2/train_teacher_v2.csv":
        train_teacher,

    "outputs/router_teacher_v2/val_teacher_v2.csv":
        val_teacher,
}

for a, b in repls.items():
    s = s.replace(a, b)

p.write_text(s)

print("PATCHED:", p)
print("WORK:", work)
print("ROUTER:", router)
print("CAL:", cal)
PY

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u "$CACHE_FINAL"

CACHE="$WORK/router_moe_cache.csv"

[ -s "$CACHE" ] || fail "router_moe_cache.csv not produced"

N_CACHE=$(tail -n +2 "$CACHE" | wc -l)

echo "CACHE rows = $N_CACHE / $EXPECTED"

[ "$N_CACHE" -eq "$EXPECTED" ] || \
  fail "Router cache coverage mismatch"

###############################################################################
# 6. PATCH + RUN CONFIG GENERATOR COPY
###############################################################################

log "6. GENERATE INFERENCE CONFIGS"

MAKE_SRC="$INT/make_full_configs.py"
MAKE_FINAL="$BENCH/make_full_configs_FINAL.py"

[ -s "$MAKE_SRC" ] || fail "Missing make_full_configs.py"

cp -f "$MAKE_SRC" "$MAKE_FINAL"

"$PY_X2" "$MAKE_FINAL" "$WORK" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
work = sys.argv[2]

s = p.read_text(errors="ignore")

s = s.replace(
    "outputs/x2dfd_router_integration/full_DFDCP_DFDC",
    work
)

p.write_text(s)
print("PATCHED:", p)
PY

cd "$INT"

"$PY_X2" "$MAKE_FINAL"

# Locate generated configs.
ORIG_CFG=$(
"$PY_X2" - "$X2" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

c = list(root.rglob("infer_full_DFDCP_DFDC_original.yaml"))
if not c:
    raise SystemExit(1)

print(max(c, key=lambda p: p.stat().st_mtime))
PY
) || fail "Original config not generated"

ROUTER_CFG=$(
"$PY_X2" - "$X2" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

c = list(root.rglob("infer_full_DFDCP_DFDC_router_moe.yaml"))
if not c:
    raise SystemExit(1)

print(max(c, key=lambda p: p.stat().st_mtime))
PY
) || fail "Router config not generated"

cp -f "$ORIG_CFG" "$BENCH/configs/original.yaml"
cp -f "$ROUTER_CFG" "$BENCH/configs/router4_final.yaml"

ORIG_CFG="$BENCH/configs/original.yaml"
ROUTER_CFG="$BENCH/configs/router4_final.yaml"

# Ensure copied configs point to our benchmark WORK.
sed -i \
 "s#${BASE}/outputs/x2dfd_router_integration/full_DFDCP_DFDC#${WORK}#g" \
 "$ORIG_CFG" "$ROUTER_CFG"

sed -i \
 "s#outputs/x2dfd_router_integration/full_DFDCP_DFDC#${WORK}#g" \
 "$ORIG_CFG" "$ROUTER_CFG"

echo "ORIGINAL CONFIG:"
grep -nE 'model|input|output|expert|cache|data' "$ORIG_CFG" || true

echo
echo "ROUTER CONFIG:"
grep -nE 'model|input|output|expert|cache|data' "$ROUTER_CFG" || true

###############################################################################
# 7. ORIGINAL OFFICIAL X2DFD
###############################################################################

log "7. OFFICIAL ORIGINAL X2DFD"

cd "$X2"

MARK_ORIG="$BENCH/.orig_start"
touch "$MARK_ORIG"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
-m eval.infer.runner \
--config "$ORIG_CFG" \
--model-path "$ORIGINAL_LORA" \
--model-base "$BASE_MODEL" \
2>&1 | tee "$BENCH/original_x2dfd.log"

cp \
 "$X2/eval/outputs/infer/latest_run.json" \
 "$BENCH/latest_run_original.json" \
 2>/dev/null || true

ORIG_RUN=$(
find "$X2/eval/outputs/infer/runs" \
 -mindepth 1 \
 -maxdepth 1 \
 -type d \
 -newer "$MARK_ORIG" \
 -printf '%T@ %p\n' \
 2>/dev/null \
| sort -nr \
| sed -n '1s/^[^ ]* //p'
)

[ -n "${ORIG_RUN:-}" ] || \
  fail "Could not identify Original X2DFD run directory"

echo "ORIGINAL RUN = $ORIG_RUN"

rm -rf "$BENCH/raw/original"
cp -a "$ORIG_RUN" "$BENCH/raw/original"

###############################################################################
# 8. ROUTER4 FINAL
###############################################################################

log "8. ROUTER4 × X2DFD FINAL"

cd "$X2"

MARK_R4="$BENCH/.router4_start"
touch "$MARK_R4"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
-m eval.infer.runner \
--config "$ROUTER_CFG" \
--model-path "$FINAL_LORA" \
--model-base "$BASE_MODEL" \
2>&1 | tee "$BENCH/router4_final.log"

cp \
 "$X2/eval/outputs/infer/latest_run.json" \
 "$BENCH/latest_run_router4.json" \
 2>/dev/null || true

R4_RUN=$(
find "$X2/eval/outputs/infer/runs" \
 -mindepth 1 \
 -maxdepth 1 \
 -type d \
 -newer "$MARK_R4" \
 -printf '%T@ %p\n' \
 2>/dev/null \
| sort -nr \
| sed -n '1s/^[^ ]* //p'
)

[ -n "${R4_RUN:-}" ] || \
  fail "Could not identify Router4 run directory"

echo "ROUTER4 RUN = $R4_RUN"

rm -rf "$BENCH/raw/router4"
cp -a "$R4_RUN" "$BENCH/raw/router4"

###############################################################################
# 9. PARSE PREDICTIONS + METRICS
###############################################################################

log "9. FINAL METRICS"

"$PY_X2" \
 "$BENCH/benchmark_manifest.csv" \
 "$BENCH/raw/original" \
 "$BENCH/raw/router4" \
 "$BENCH/results" <<'PY'
import csv
import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict

manifest_file = Path(sys.argv[1])
orig_root = Path(sys.argv[2])
r4_root = Path(sys.argv[3])
out = Path(sys.argv[4])

out.mkdir(parents=True, exist_ok=True)

labels = {}

with manifest_file.open("r", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        p = str(Path(r["image_path"]).resolve())
        labels[p] = {
            "label": r["label_text"].lower(),
            "dataset": r["dataset"],
        }

def prediction_from_text(text):
    if not isinstance(text, str):
        return None

    t = text.lower()

    # Prefer direct declarative answer.
    pats = [
        r'\b(?:this|the)\s+image\s+(?:is|appears|looks)\s+(real|fake)\b',
        r'^\s*(real|fake)\b',
        r'\bprediction\s*[:=]\s*(real|fake)\b',
        r'\banswer\s*[:=]\s*(real|fake)\b',
    ]

    for pat in pats:
        m = re.search(pat, t)
        if m:
            return m.group(1)

    # Last-resort only when exactly one class token is present.
    has_real = bool(re.search(r'\breal\b', t))
    has_fake = bool(re.search(r'\bfake\b', t))

    if has_real ^ has_fake:
        return "real" if has_real else "fake"

    return None

def iter_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_dicts(v)

    elif isinstance(obj, list):
        for x in obj:
            yield from iter_dicts(x)

def extract(root):
    pred = {}

    for jf in root.rglob("*.json"):
        try:
            obj = json.loads(jf.read_text(errors="ignore"))
        except Exception:
            continue

        for d in iter_dicts(obj):
            image = None

            for k in ["image_path", "image", "path"]:
                v = d.get(k)
                if isinstance(v, str):
                    image = v
                    break

            if not image:
                continue

            try:
                ip = str(Path(image).resolve())
            except Exception:
                continue

            answer = None

            for k in [
                "prediction",
                "answer",
                "response",
                "generated_text",
                "output",
                "text",
            ]:
                v = d.get(k)
                if isinstance(v, str):
                    q = prediction_from_text(v)
                    if q:
                        answer = q
                        break

            if answer is None:
                conv = d.get("conversations")

                if isinstance(conv, list):
                    for msg in reversed(conv):
                        if not isinstance(msg, dict):
                            continue

                        role = str(
                            msg.get("from", msg.get("role", ""))
                        ).lower()

                        if role not in {
                            "gpt", "assistant", "model"
                        }:
                            continue

                        val = msg.get(
                            "value",
                            msg.get("content", "")
                        )

                        q = prediction_from_text(val)

                        if q:
                            answer = q
                            break

            if answer is None:
                continue

            fake_score = None

            for k in [
                "fake_score",
                "prob_fake",
                "p_fake",
                "fake_probability",
            ]:
                v = d.get(k)

                try:
                    if v is not None:
                        fake_score = float(v)
                        break
                except Exception:
                    pass

            pred[ip] = {
                "prediction": answer,
                "fake_score": fake_score,
                "source_json": str(jf),
            }

    return pred

orig = extract(orig_root)
r4 = extract(r4_root)

print("MANIFEST =", len(labels))
print("ORIGINAL PREDICTIONS =", len(orig))
print("ROUTER4 PREDICTIONS =", len(r4))

def metrics(rows):
    if not rows:
        return {}

    tp = tn = fp = fn = 0

    for r in rows:
        y = r["label"]
        p = r["prediction"]

        if y == "fake" and p == "fake":
            tp += 1
        elif y == "real" and p == "real":
            tn += 1
        elif y == "real" and p == "fake":
            fp += 1
        elif y == "fake" and p == "real":
            fn += 1

    n = tp + tn + fp + fn

    acc = (tp + tn) / n if n else float("nan")

    fake_recall = tp / (tp + fn) if tp + fn else float("nan")
    real_recall = tn / (tn + fp) if tn + fp else float("nan")

    bacc = (
        (fake_recall + real_recall) / 2
        if not math.isnan(fake_recall)
        and not math.isnan(real_recall)
        else float("nan")
    )

    precision = tp / (tp + fp) if tp + fp else 0.0

    f1 = (
        2 * precision * fake_recall
        / (precision + fake_recall)
        if precision + fake_recall
        else 0.0
    )

    result = {
        "N": n,
        "ACC": acc,
        "BACC": bacc,
        "F1_fake": f1,
        "Real_Recall": real_recall,
        "Fake_Recall": fake_recall,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    # AUC only when actual numeric fake probabilities exist.
    scored = [
        r for r in rows
        if r.get("fake_score") is not None
    ]

    if len(scored) == len(rows) and len(rows) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            yy = [
                1 if r["label"] == "fake" else 0
                for r in scored
            ]

            ss = [
                r["fake_score"]
                for r in scored
            ]

            result["AUC"] = float(
                roc_auc_score(yy, ss)
            )
        except Exception:
            result["AUC"] = None
    else:
        result["AUC"] = None

    return result

all_predictions = []

for model_name, P in [
    ("Original_X2DFD", orig),
    ("Router4_FINAL", r4),
]:
    for image, truth in labels.items():
        if image not in P:
            continue

        all_predictions.append({
            "model": model_name,
            "image_path": image,
            "dataset": truth["dataset"],
            "label": truth["label"],
            "prediction": P[image]["prediction"],
            "fake_score": P[image]["fake_score"],
            "correct": int(
                truth["label"] == P[image]["prediction"]
            ),
            "source_json": P[image]["source_json"],
        })

with (out / "predictions.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    keys = [
        "model",
        "image_path",
        "dataset",
        "label",
        "prediction",
        "fake_score",
        "correct",
        "source_json",
    ]

    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(all_predictions)

summary = {}

datasets = sorted(
    set(x["dataset"] for x in labels.values())
)

for model_name, P in [
    ("Original_X2DFD", orig),
    ("Router4_FINAL", r4),
]:
    summary[model_name] = {}

    for ds in datasets + ["OVERALL"]:
        rows = []

        for image, truth in labels.items():
            if image not in P:
                continue

            if ds != "OVERALL" and truth["dataset"] != ds:
                continue

            rows.append({
                "label": truth["label"],
                "prediction": P[image]["prediction"],
                "fake_score": P[image]["fake_score"],
            })

        m = metrics(rows)

        expected = (
            len(labels)
            if ds == "OVERALL"
            else sum(
                1 for v in labels.values()
                if v["dataset"] == ds
            )
        )

        m["Expected"] = expected
        m["Coverage"] = (
            m.get("N", 0) / expected
            if expected else 0
        )

        summary[model_name][ds] = m

common = sorted(
    set(labels)
    & set(orig)
    & set(r4)
)

dis = []

for image in common:
    o = orig[image]["prediction"]
    r = r4[image]["prediction"]
    y = labels[image]["label"]

    if o == r:
        continue

    dis.append({
        "image_path": image,
        "dataset": labels[image]["dataset"],
        "label": y,
        "original_prediction": o,
        "router4_prediction": r,
        "original_correct": int(o == y),
        "router4_correct": int(r == y),
    })

with (out / "disagreements.csv").open(
    "w", encoding="utf-8", newline=""
) as f:

    keys = [
        "image_path",
        "dataset",
        "label",
        "original_prediction",
        "router4_prediction",
        "original_correct",
        "router4_correct",
    ]

    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(dis)

summary["comparison"] = {
    "manifest_images": len(labels),
    "original_predictions": len(orig),
    "router4_predictions": len(r4),
    "common_predictions": len(common),
    "disagreements": len(dis),
    "router4_fixes_original": sum(
        1 for x in dis
        if x["router4_correct"] == 1
        and x["original_correct"] == 0
    ),
    "original_fixes_router4": sum(
        1 for x in dis
        if x["original_correct"] == 1
        and x["router4_correct"] == 0
    ),
}

(out / "FINAL_SUMMARY.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

# Flat metrics table.
with (out / "metrics.csv").open(
    "w", encoding="utf-8", newline=""
) as f:

    keys = [
        "model", "dataset",
        "Expected", "N", "Coverage",
        "ACC", "BACC", "AUC", "F1_fake",
        "Real_Recall", "Fake_Recall",
        "TN", "FP", "FN", "TP",
    ]

    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()

    for model in [
        "Original_X2DFD",
        "Router4_FINAL",
    ]:
        for ds, m in summary[model].items():
            row = {"model": model, "dataset": ds}
            row.update(m)
            w.writerow(row)

print()
print(json.dumps(summary, indent=2))
print()
print("SAVED:")
print(out / "metrics.csv")
print(out / "predictions.csv")
print(out / "disagreements.csv")
print(out / "FINAL_SUMMARY.json")
PY

touch "$BENCH/BENCHMARK_READY"

log "✅ FINAL 3-CROSS A/B BENCHMARK COMPLETE"

cat "$BENCH/results/metrics.csv"

echo
echo "RESULTS:"
echo "$BENCH/results"

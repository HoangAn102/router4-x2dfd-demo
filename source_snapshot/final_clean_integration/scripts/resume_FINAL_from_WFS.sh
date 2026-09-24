#!/usr/bin/env bash
set -euo pipefail

BASE=/home/aiotlab/hoangan
PY=/home/aiotlab/miniconda3/envs/X2DFD/bin/python

ROOT="$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

RX="$BASE/projects/router_x2dfd"
RV2="$BASE/projects/router_v2"
X2="$BASE/projects/X2DFD"

OLD_TRAIN="$BASE/outputs/router_teacher_v2/train_teacher_v2.csv"
OLD_VAL="$BASE/outputs/router_teacher_v2/val_teacher_v2.csv"
OLD_ROUTER="$BASE/outputs/router_training_v2/run_effb0_3domain/best.pt"

NEW_TRAIN="$ROOT/01_teacher/train_teacher_FINAL_CLEAN.csv"
NEW_VAL="$ROOT/01_teacher/val_teacher_FINAL_CLEAN.csv"

ROUTER_OUT="$ROOT/02_router/run_final_clean"

log () {
    echo
    echo "======================================================================"
    echo "$1"
    echo "$(date)"
    echo "======================================================================"
}

fail () {
    echo "[STOP] $1"
    echo "$1" > "$ROOT/STOP_REASON.txt"
    exit 1
}


###############################################################################

###############################################################################
# RESUME OVERRIDE
###############################################################################

ROUTER_BEST="/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/02_router/final_checkpoint/best.pt"

echo
echo "======================================================================"
echo "RESUMING FINAL PIPELINE"
echo "======================================================================"
echo "Clean teacher : READY"
echo "Router        : $ROUTER_BEST"
echo "Starting      : WFS -> SFT -> FINAL LoRA"
echo

# C. REBUILD ROUTED WFS USING REAL PROJECT SCRIPT
###############################################################################

log "C. REBUILD ROUTED WFS"

SRC="$RX/build_router4_train_wfs.py"
PATCHED="$ROOT/03_wfs/build_router4_train_wfs_FINAL.py"

test -f "$SRC" || fail "build_router4_train_wfs.py missing"

"$PY" - "$SRC" "$PATCHED" "$NEW_TRAIN" "$NEW_VAL" "$OLD_ROUTER" "$ROUTER_BEST" "$ROOT/03_wfs" <<'PY'
from pathlib import Path
import sys,re

src=Path(sys.argv[1]).read_text(errors="ignore")

patched=Path(sys.argv[2])
train=sys.argv[3]
val=sys.argv[4]
old_router=sys.argv[5]
router=sys.argv[6]
outdir=sys.argv[7]

src=src.replace(
"/home/aiotlab/hoangan/outputs/router_teacher_v2/train_teacher_v2.csv",
train
)

src=src.replace(
"/home/aiotlab/hoangan/outputs/router_teacher_v2/val_teacher_v2.csv",
val
)

src=src.replace(
old_router,
router
)

# redirect common absolute Router4 output paths away from old experiment
src=re.sub(
r'(["\'])/home/aiotlab/hoangan/outputs/ROUTER4[^"\']*',
lambda m:m.group(1)+outdir,
src
)

patched.write_text(src)
PY

mkdir -p "$ROOT/03_wfs"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$PATCHED" \
    >> "$ROOT/logs/wfs_build.log" 2>&1

ROUTED=$(
find "$ROOT/03_wfs" \
    -type f \
    -name '*.csv' \
    -size +1M \
    | sort \
    | tail -1
)

[ -n "$ROUTED" ] || fail "No routed WFS CSV produced"

echo "$ROUTED" > "$ROOT/03_wfs/ROUTED_WFS_PATH.txt"

echo "ROUTED WFS = $ROUTED"


###############################################################################
# D. REBUILD CLEAN SFT
###############################################################################

log "D. REBUILD CLEAN SFT"

MERGE=$(
find "$RX" "$BASE/outputs" \
    -type f \
    -name '03_router_wfs_merge.py' \
    2>/dev/null \
    | head -1
)

[ -n "$MERGE" ] || fail "03_router_wfs_merge.py not found"

PATCHED_MERGE="$ROOT/04_sft/03_router_wfs_merge_FINAL.py"

BASE_SFT="$BASE/outputs/V1_3_STRICT_CLEAN_20260919/router4_x2dfd_native_sft_V1_3_STRICT_CLEAN.json"

test -s "$BASE_SFT" || fail "V1.3 strict source SFT missing"

"$PY" - "$MERGE" "$PATCHED_MERGE" "$ROUTED" "$BASE_SFT" "$ROOT/04_sft" <<'PY'
from pathlib import Path
import sys,re

src=Path(sys.argv[1]).read_text(errors="ignore")

routed=sys.argv[3]
sft=sys.argv[4]
outdir=sys.argv[5]

# Replace known old WFS CSV-ish absolute references.
src=re.sub(
r'(["\'])/home/aiotlab/hoangan/outputs/[^"\']+\.csv',
lambda m:m.group(1)+routed,
src
)

# Replace known SFT JSON references.
src=re.sub(
r'(["\'])/home/aiotlab/hoangan/outputs/[^"\']+\.json',
lambda m:m.group(1)+sft,
src
)

# Redirect output directory if script hardcodes one.
src=re.sub(
r'(["\'])/home/aiotlab/hoangan/outputs/ROUTER4[^"\']*',
lambda m:m.group(1)+outdir,
src
)

Path(sys.argv[2]).write_text(src)
PY

mkdir -p "$ROOT/04_sft"

"$PY" "$PATCHED_MERGE" \
    >> "$ROOT/logs/sft_build.log" 2>&1

FINAL_SFT=$(
find "$ROOT/04_sft" \
    -type f \
    -name '*.json' \
    -size +5M \
    | sort \
    | tail -1
)

[ -n "$FINAL_SFT" ] || fail "Final SFT JSON not produced"

echo "$FINAL_SFT" > "$ROOT/04_sft/FINAL_SFT_PATH.txt"

echo "FINAL SFT = $FINAL_SFT"


###############################################################################
# E. RETRAIN FINAL LORA FROM BASE
###############################################################################

log "E. RETRAIN FINAL LORA"

LAUNCHER=$(
find "$BASE/outputs" "$RX" \
    -type f \
    \( -name '*.sh' -o -name '*.bash' \) \
    2>/dev/null \
    | while read f
      do
          grep -q 'v1_3_strict_clean' "$f" 2>/dev/null && {
              echo "$f"
              break
          }
      done
)

[ -n "$LAUNCHER" ] || fail "Could not recover V1.3 LoRA launcher"

FINAL_MODEL="$ROOT/05_lora/router4_x2dfd_FINAL_CLEAN"

PATCHED_LAUNCHER="$ROOT/05_lora/train_FINAL_LORA.sh"

"$PY" - "$LAUNCHER" "$PATCHED_LAUNCHER" "$FINAL_SFT" "$FINAL_MODEL" <<'PY'
from pathlib import Path
import sys,re

src=Path(sys.argv[1]).read_text(errors="ignore")

sft=sys.argv[3]
out=sys.argv[4]

# Replace old V1.x SFT json.
src=re.sub(
r'/home/aiotlab/hoangan/outputs/[^ \'"\\]+V1_3[^ \'"\\]+\.json',
sft,
src
)

src=re.sub(
r'/home/aiotlab/hoangan/outputs/[^ \'"\\]+router4_x2dfd[^ \'"\\]*v1_3[^ \'"\\]*',
out,
src
)

# Also explicit data_path/output_dir forms.
src=re.sub(
r'(--data_path\s+)[^ \n]+',
r'\1'+sft,
src
)

src=re.sub(
r'(--output_dir\s+)[^ \n]+',
r'\1'+out,
src
)

Path(sys.argv[2]).write_text(src)
PY

chmod +x "$PATCHED_LAUNCHER"

CUDA_VISIBLE_DEVICES=0 \
bash "$PATCHED_LAUNCHER" \
    >> "$ROOT/logs/lora_train.log" 2>&1

test -s "$FINAL_MODEL/adapter_model.safetensors" \
    || fail "Final LoRA adapter missing"

touch "$FINAL_MODEL/MODEL_READY"

sha256sum \
"$FINAL_MODEL/adapter_model.safetensors" \
> "$FINAL_MODEL/adapter_model.sha256"

echo "$FINAL_MODEL" > "$ROOT/05_lora/FINAL_MODEL_PATH.txt"

echo "FINAL MODEL = $FINAL_MODEL"


###############################################################################

###############################################################################
# FINAL MODEL FREEZE
###############################################################################

log "FINAL MODEL FREEZE"

FINAL_MODEL="$ROOT/05_lora/router4_x2dfd_FINAL_CLEAN"

test -s "$FINAL_MODEL/adapter_model.safetensors" \
    || fail "Final adapter_model.safetensors missing"

test -s "$FINAL_MODEL/adapter_config.json" \
    || fail "Final adapter_config.json missing"


sha256sum \
    "$FINAL_MODEL/adapter_model.safetensors" \
    > "$FINAL_MODEL/adapter_model.sha256"


echo "$FINAL_MODEL" \
    > "$ROOT/05_lora/FINAL_MODEL_PATH.txt"


# Freeze deployable model separately.
FREEZE="$ROOT/05_lora/DEPLOY_FINAL_CLEAN"

rm -rf "$FREEZE"
mkdir -p "$FREEZE"

for f in \
    adapter_model.safetensors \
    adapter_config.json \
    non_lora_trainables.bin \
    config.json \
    trainer_state.json \
    README.md \
    MODEL_READY \
    adapter_model.sha256
do
    if [ -e "$FINAL_MODEL/$f" ]; then
        cp -a "$FINAL_MODEL/$f" "$FREEZE/"
    fi
done


touch "$FREEZE/FINAL_MODEL_READY"


PKG="$ROOT/05_lora/ROUTER4_FINAL_CLEAN_MODEL.tar.gz"

rm -f "$PKG"

tar -C "$ROOT/05_lora" \
    -czf "$PKG" \
    DEPLOY_FINAL_CLEAN


sha256sum "$PKG" \
    > "$PKG.sha256"


cat > "$ROOT/FINAL_MODEL_STATUS.txt" <<STATUS
============================================================
ROUTER4 / X2DFD FINAL CLEAN MODEL
============================================================

Clean calibrator:
$ROOT/01_teacher/calibrators_FINAL_CLEAN.joblib

Clean teacher:
$ROOT/01_teacher/train_teacher_FINAL_CLEAN.csv
$ROOT/01_teacher/val_teacher_FINAL_CLEAN.csv

Final Router:
$ROUTER_BEST

Final routed WFS:
$(cat "$ROOT/03_wfs/ROUTED_WFS_PATH.txt" 2>/dev/null || true)

Final SFT:
$(cat "$ROOT/04_sft/FINAL_SFT_PATH.txt" 2>/dev/null || true)

Final LoRA:
$FINAL_MODEL

Frozen deploy model:
$FREEZE

Package:
$PKG

STATUS: FINAL MODEL READY
STATUS


touch "$ROOT/FINAL_MODEL_READY"


echo
echo "======================================================================"
echo "✅ FINAL MODEL READY"
echo "======================================================================"

cat "$ROOT/FINAL_MODEL_STATUS.txt"

echo
echo "PACKAGE:"
ls -lh "$PKG"

echo
echo "SHA256:"
cat "$PKG.sha256"


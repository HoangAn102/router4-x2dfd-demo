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
# A. CLEAN + TEACHER
###############################################################################

log "A. MAX CLEAN + REBUILD TEACHER"

"$PY" "$ROOT/scripts/rebuild_clean_teacher.py"

test -s "$NEW_TRAIN" || fail "clean teacher TRAIN missing"
test -s "$NEW_VAL"   || fail "clean teacher VAL missing"


###############################################################################
# B. RETRAIN ROUTER FROM CLEAN TEACHER
###############################################################################

log "B. RETRAIN ROUTER"

SRC="$RV2/train_router_v2.py"
PATCHED="$ROOT/02_router/train_router_FINAL.py"

test -f "$SRC" || fail "train_router_v2.py missing"

"$PY" - "$SRC" "$PATCHED" "$NEW_TRAIN" "$NEW_VAL" "$ROUTER_OUT" <<'PY'
from pathlib import Path
import sys,re

src=Path(sys.argv[1]).read_text(errors="ignore")

train=sys.argv[3]
val=sys.argv[4]
out=sys.argv[5]

replacements={
"/home/aiotlab/hoangan/outputs/router_teacher_v2/train_teacher_v2.csv":train,
"/home/aiotlab/hoangan/outputs/router_teacher_v2/val_teacher_v2.csv":val,
"/home/aiotlab/hoangan/outputs/router_training_v2/run_effb0_3domain":out,
}

for a,b in replacements.items():
    src=src.replace(a,b)

Path(sys.argv[2]).write_text(src)
PY

mkdir -p "$ROUTER_OUT"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$PATCHED" \
    >> "$ROOT/logs/router_train.log" 2>&1

ROUTER_BEST=$(find "$ROUTER_OUT" -type f -name best.pt | head -1)

[ -n "$ROUTER_BEST" ] || fail "Router finished but best.pt not found"

echo "$ROUTER_BEST" > "$ROOT/02_router/BEST_ROUTER_PATH.txt"

echo "ROUTER BEST = $ROUTER_BEST"


###############################################################################
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
# F. SAME-LIST COMPARISON AGAINST ORIGINAL X2DFD
###############################################################################

log "F. ORIGINAL X2DFD vs FINAL ROUTER4"

COMPARE="$RX/preflight_crossdata_first_comparison.py"

if [ ! -f "$COMPARE" ]; then
    echo "Comparison script missing." \
      > "$ROOT/06_eval/EVAL_NEEDS_MANUAL.txt"
else

    PATCHED_COMPARE="$ROOT/06_eval/compare_FINAL_vs_X2DFD.py"

    "$PY" - "$COMPARE" "$PATCHED_COMPARE" "$OLD_ROUTER" "$ROUTER_BEST" "$FINAL_MODEL" "$ROOT/06_eval" <<'PY'
from pathlib import Path
import sys,re

src=Path(sys.argv[1]).read_text(errors="ignore")

old_router=sys.argv[3]
new_router=sys.argv[4]
model=sys.argv[5]
outdir=sys.argv[6]

src=src.replace(
old_router,
new_router
)

# Replace V1 / V1.1 / V1.3 adapter references by final adapter.
src=re.sub(
r'/home/aiotlab/hoangan/outputs/[^ \'"\\]+lora_router4[^ \'"\\]*',
model,
src
)

# Redirect comparison outputs.
src=re.sub(
r'(["\'])/home/aiotlab/hoangan/outputs/[^"\']*(?:comparison|eval|crossdata)[^"\']*',
lambda m:m.group(1)+outdir,
src,
flags=re.I
)

Path(sys.argv[2]).write_text(src)
PY

    set +e

    CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$PATCHED_COMPARE" \
      > "$ROOT/logs/final_comparison.log" 2>&1

    RC=$?

    set -e

    echo "$RC" > "$ROOT/06_eval/COMPARE_EXIT_CODE.txt"

    if [ "$RC" -ne 0 ]; then
        {
          echo "Automatic comparison script failed safely."
          echo "Final model itself is preserved."
          echo
          tail -n 100 "$ROOT/logs/final_comparison.log"
        } > "$ROOT/06_eval/EVAL_NEEDS_MANUAL.txt"
    fi
fi


###############################################################################
# G. PACKAGE EVERYTHING
###############################################################################

log "G. PACKAGE FINAL"

cat > "$ROOT/FINAL_STATUS.txt" <<EOF
FINAL CLEAN PIPELINE FINISHED

MAX CLEAN:
$ROOT/00_clean

CLEAN TEACHER:
$ROOT/01_teacher

NEW ROUTER:
$ROUTER_BEST

ROUTED WFS:
$ROUTED

FINAL SFT:
$FINAL_SFT

FINAL MODEL:
$FINAL_MODEL

EVALUATION:
$ROOT/06_eval

LOGS:
$ROOT/logs
EOF

tar -C "$BASE/outputs" \
    -czf "$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919_RESULTS.tar.gz" \
    "ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

sha256sum \
"$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919_RESULTS.tar.gz" \
> "$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919_RESULTS.tar.gz.sha256"

touch "$ROOT/FINAL_PIPELINE_READY"

cat "$ROOT/FINAL_STATUS.txt"


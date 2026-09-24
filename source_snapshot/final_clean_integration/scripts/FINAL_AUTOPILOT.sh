#!/usr/bin/env bash
set -euo pipefail

BASE=/home/aiotlab/hoangan
ENV=/home/aiotlab/miniconda3/envs/X2DFD
PY="$ENV/bin/python"

ROOT="$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

TRAIN="$ROOT/01_teacher/train_teacher_FINAL_CLEAN.csv"
ROUTER="$ROOT/02_router/final_checkpoint/best.pt"

WFSDIR="$ROOT/03_wfs"
WFS="$WFSDIR/router4_FINAL_CLEAN_routed_wfs.csv"

REPAIR="$ROOT/scripts/repair_final_wfs.py"

SRC_RUNNER="$ROOT/scripts/run_full_final.sh"

CONTINUE="$ROOT/scripts/AUTO_SFT_LORA_FINAL.sh"

log () {
    echo
    echo "======================================================================"
    echo "$1"
    echo "$(date)"
    echo "======================================================================"
}

fail () {
    echo "[STOP] $1"
    echo "$1" > "$ROOT/AUTOPILOT_STOP_REASON.txt"
    exit 1
}


###############################################################################
# 0. CUDA / BITSANDBYTES LIBRARY FIX
###############################################################################

log "0. CUDA LIBRARY PREP"

LIBDIRS=()

for d in \
    "$ENV/lib" \
    "$ENV/lib/python3.10/site-packages/nvidia/cusparse/lib" \
    "$ENV/lib/python3.10/site-packages/nvidia/cublas/lib" \
    "$ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib" \
    /usr/local/cuda/lib64 \
    /usr/local/cuda-12/lib64
do
    [ -d "$d" ] && LIBDIRS+=("$d")
done

# Find actual libcusparse location, bounded search only.
CUSPARSE=$(
    find \
      "$ENV" \
      /usr/local/cuda* \
      -type f \
      -name 'libcusparse.so*' \
      2>/dev/null \
      | head -1 || true
)

if [ -n "$CUSPARSE" ]; then
    LIBDIRS+=("$(dirname "$CUSPARSE")")
    echo "libcusparse found:"
    echo "$CUSPARSE"
else
    echo "[WARN] libcusparse not found by bounded search."
fi

NEW_LD=""

for d in "${LIBDIRS[@]}"; do
    case ":$NEW_LD:" in
        *":$d:"*) ;;
        *)
            if [ -z "$NEW_LD" ]; then
                NEW_LD="$d"
            else
                NEW_LD="$NEW_LD:$d"
            fi
            ;;
    esac
done

export LD_LIBRARY_PATH="$NEW_LD:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

"$PY" - <<'PY' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch check:",repr(e))
PY


###############################################################################
# 1. VERIFY PREVIOUS CLEAN STAGES
###############################################################################

log "1. VERIFY CLEAN TEACHER + ROUTER"

test -s "$TRAIN" \
    || fail "clean teacher missing"

test -s "$ROUTER" \
    || fail "final clean Router missing"

echo "TRAIN:"
ls -lh "$TRAIN"

echo
echo "ROUTER:"
ls -lh "$ROUTER"


###############################################################################
# 2. VALIDATOR FUNCTION
###############################################################################

validate_wfs () {

"$PY" - "$TRAIN" "$WFS" <<'PY'
import sys
import pandas as pd
from pathlib import Path

train=Path(sys.argv[1])
wfs=Path(sys.argv[2])

if not wfs.is_file():
    raise SystemExit(10)

a=pd.read_csv(
    train,
    usecols=["image_path"],
    low_memory=False
)

try:
    b=pd.read_csv(
        wfs,
        usecols=[
            "image_path",
            "selected_expert",
            "wfs_text"
        ],
        low_memory=False
    )
except Exception:
    raise SystemExit(11)

print("clean_train_rows =",len(a))
print("wfs_rows         =",len(b))

if len(a) != len(b):
    raise SystemExit(12)

if not (
    a["image_path"]
    .astype(str)
    .reset_index(drop=True)
    .equals(
        b["image_path"]
        .astype(str)
        .reset_index(drop=True)
    )
):
    raise SystemExit(13)

allowed={
    "blending",
    "diffusion",
    "frequency",
    "texture",
}

if not (
    b["selected_expert"]
    .astype(str)
    .str.lower()
    .isin(allowed)
    .all()
):
    raise SystemExit(14)

if (
    b["wfs_text"]
    .isna()
    .any()
):
    raise SystemExit(15)

print("[PASS] WFS matches clean train exactly")
PY
}


###############################################################################
# 3. WAIT FOR CURRENT REPAIR
###############################################################################

log "2. WAIT FOR CURRENT WFS REPAIR"

while true
do

    if validate_wfs; then
        echo "[PASS] WFS already ready."
        touch "$WFSDIR/WFS_READY"
        printf '%s\n' "$WFS" \
          > "$WFSDIR/ROUTED_WFS_PATH.txt"
        break
    fi

    # Existing repair process from current terminal.
    if pgrep -f \
        '^/home/aiotlab/miniconda3/envs/X2DFD/bin/python /home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/scripts/repair_final_wfs.py$' \
        >/dev/null
    then
        echo "[WAIT] existing repair_final_wfs.py still running"

        ps -eo pid,etime,%cpu,%mem,cmd \
        | grep 'repair_final_wfs.py' \
        | grep -v grep || true

        echo
        echo "GPU:"
        nvidia-smi \
          --query-gpu=utilization.gpu,memory.used,memory.total \
          --format=csv,noheader \
          2>/dev/null || true

        sleep 60
        continue
    fi

    echo
    echo "[INFO] existing repair stopped without valid WFS."
    echo "[INFO] rerunning repair with fixed CUDA libraries."

    test -f "$REPAIR" \
        || fail "repair_final_wfs.py missing"

    rm -f \
      "$WFSDIR/WFS_READY" \
      "$WFSDIR/ROUTED_WFS_PATH.txt"

    set +e

    "$PY" "$REPAIR" \
      >> "$ROOT/logs/wfs_repair_autopilot.log" 2>&1

    RC=$?

    set -e

    echo "repair exit = $RC"

    if validate_wfs; then
        echo "[PASS] repaired WFS is valid"
        touch "$WFSDIR/WFS_READY"
        printf '%s\n' "$WFS" \
          > "$WFSDIR/ROUTED_WFS_PATH.txt"
        break
    fi

    echo
    echo "LAST WFS REPAIR LOG:"
    tail -n 100 \
      "$ROOT/logs/wfs_repair_autopilot.log" \
      2>/dev/null || true

    fail "WFS repair failed or output does not match clean TRAIN"
done


###############################################################################
# 4. WFS FINAL AUDIT
###############################################################################

log "3. FINAL WFS VERIFIED"

validate_wfs

echo "WFS:"
ls -lh "$WFS"

echo
echo "WFS path:"
echo "$WFS"


###############################################################################
# 5. AVOID DUPLICATE FINAL TRAIN
###############################################################################

if [ -f "$ROOT/FINAL_MODEL_READY" ]; then
    log "FINAL MODEL ALREADY READY"
    exit 0
fi

if pgrep -af \
   'model_train.py.*router4_x2dfd_FINAL_CLEAN' \
   >/dev/null
then
    echo "[INFO] Final LoRA already training."
    echo "[INFO] Autopilot will monitor it instead of launching duplicate."
else

###############################################################################
# 6. BUILD CONTINUATION SCRIPT: D + E ONLY
###############################################################################

log "4. PREPARE SFT -> FINAL LORA"

test -s "$SRC_RUNNER" \
    || fail "saved original runner missing"

DLINE=$(
    grep -n '^# D\. REBUILD CLEAN SFT' \
      "$SRC_RUNNER" \
    | head -1 \
    | cut -d: -f1
)

FLINE=$(
    grep -n '^# F\. SAME-LIST COMPARISON' \
      "$SRC_RUNNER" \
    | head -1 \
    | cut -d: -f1
)

ALINE=$(
    grep -n '^# A\. CLEAN' \
      "$SRC_RUNNER" \
    | head -1 \
    | cut -d: -f1
)

[ -n "$DLINE" ] \
    || fail "cannot find Stage D"

[ -n "$FLINE" ] \
    || fail "cannot find Stage F"

[ -n "$ALINE" ] \
    || fail "cannot find Stage A"


head -n $((ALINE-1)) \
    "$SRC_RUNNER" \
    > "$CONTINUE"


cat >> "$CONTINUE" <<EOF

###############################################################################
# AUTOPILOT OVERRIDES
###############################################################################

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

ROUTER_BEST="$ROUTER"
ROUTED="$WFS"

echo
echo "======================================================================"
echo "AUTOPILOT CONTINUE"
echo "======================================================================"
echo "Router : \$ROUTER_BEST"
echo "WFS    : \$ROUTED"
echo "Next   : CLEAN SFT -> FINAL LoRA"
EOF


sed -n \
  "${DLINE},$((FLINE-1))p" \
  "$SRC_RUNNER" \
  >> "$CONTINUE"


###############################################################################
# 7. FREEZE / PACKAGE BLOCK
###############################################################################

cat >> "$CONTINUE" <<'EOF'

###############################################################################
# FREEZE FINAL MODEL
###############################################################################

log "FINAL MODEL FREEZE"

FINAL_MODEL="$ROOT/05_lora/router4_x2dfd_FINAL_CLEAN"
FREEZE="$ROOT/05_lora/DEPLOY_FINAL_CLEAN"

test -s "$FINAL_MODEL/adapter_model.safetensors" \
    || fail "final adapter_model.safetensors missing"

test -s "$FINAL_MODEL/adapter_config.json" \
    || fail "final adapter_config.json missing"


rm -rf "$FREEZE"
mkdir -p "$FREEZE"


for f in \
    adapter_model.safetensors \
    adapter_config.json \
    non_lora_trainables.bin \
    config.json \
    trainer_state.json \
    README.md \
    MODEL_READY
do
    if [ -e "$FINAL_MODEL/$f" ]; then
        cp -a \
          "$FINAL_MODEL/$f" \
          "$FREEZE/"
    fi
done


sha256sum \
  "$FINAL_MODEL/adapter_model.safetensors" \
  > "$FREEZE/adapter_model.sha256"


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
ROUTER4 X2DFD — FINAL CLEAN MODEL
============================================================

Clean teacher:
$ROOT/01_teacher/train_teacher_FINAL_CLEAN.csv

Clean calibrator:
$ROOT/01_teacher/calibrators_FINAL_CLEAN.joblib

Final Router:
$ROUTER_BEST

Verified clean WFS:
$ROUTED

Final SFT:
$(cat "$ROOT/04_sft/FINAL_SFT_PATH.txt" 2>/dev/null || true)

Final LoRA:
$FINAL_MODEL

Deploy folder:
$FREEZE

Model package:
$PKG

STATUS:
FINAL MODEL READY
STATUS


touch "$ROOT/FINAL_MODEL_READY"


echo
echo "======================================================================"
echo "✅✅✅ FINAL MODEL READY ✅✅✅"
echo "======================================================================"

cat "$ROOT/FINAL_MODEL_STATUS.txt"

echo
echo "PACKAGE:"
ls -lh "$PKG"

echo
echo "SHA256:"
cat "$PKG.sha256"

EOF


chmod +x "$CONTINUE"

bash -n "$CONTINUE" \
    || fail "continuation syntax invalid"


###############################################################################
# 8. RUN D + E DIRECTLY
###############################################################################

log "5. START CLEAN SFT + FINAL LORA"

bash "$CONTINUE" \
  >> "$ROOT/logs/sft_lora_autopilot.log" 2>&1

fi


###############################################################################
# 9. WAIT / VERIFY FINAL MODEL
###############################################################################

log "6. VERIFY FINAL MODEL"

while true
do

    if [ -f "$ROOT/FINAL_MODEL_READY" ]; then
        break
    fi

    if pgrep -af \
       'model_train.py.*router4_x2dfd_FINAL_CLEAN' \
       >/dev/null
    then
        echo "[WAIT] final LoRA training..."

        grep -E \
          '[0-9]+/[0-9]+|loss.*learning_rate.*epoch' \
          "$ROOT/logs/lora_train.log" \
          2>/dev/null \
          | tail -n 4 || true

        sleep 60
        continue
    fi

    # If continuation ended normally it should have created flag.
    if [ ! -f "$ROOT/FINAL_MODEL_READY" ]; then
        echo
        echo "SFT/LORA LOG:"
        tail -n 100 \
          "$ROOT/logs/sft_lora_autopilot.log" \
          2>/dev/null || true

        fail "final training stopped without FINAL_MODEL_READY"
    fi
done


log "AUTOPILOT COMPLETE"

cat "$ROOT/FINAL_MODEL_STATUS.txt"


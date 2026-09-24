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
# RECOVERED FINAL STATE
###############################################################################

FINAL_SFT="/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/04_sft/router4_x2dfd_FINAL_CLEAN_SFT.json"
ROUTER_BEST="/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/02_router/final_checkpoint/best.pt"
ROUTED="/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/03_wfs/router4_FINAL_CLEAN_routed_wfs.csv"

echo "============================================================"
echo "FINAL LORA ONLY"
echo "============================================================"
echo "SFT   : $FINAL_SFT"
echo "Router: $ROUTER_BEST"
echo "WFS   : $ROUTED"
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
# FREEZE FINAL
###############################################################################

log "FREEZE FINAL MODEL"

FINAL_MODEL="$ROOT/05_lora/router4_x2dfd_FINAL_CLEAN"
FREEZE="$ROOT/05_lora/DEPLOY_FINAL_CLEAN"

test -s "$FINAL_MODEL/adapter_model.safetensors" \
    || fail "final adapter missing"

test -s "$FINAL_MODEL/adapter_config.json" \
    || fail "adapter config missing"

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
    [ -e "$FINAL_MODEL/$f" ] && \
      cp -a "$FINAL_MODEL/$f" "$FREEZE/"
done

sha256sum \
"$FINAL_MODEL/adapter_model.safetensors" \
> "$FREEZE/adapter_model.sha256"

touch "$FREEZE/FINAL_MODEL_READY"

PKG="$ROOT/05_lora/ROUTER4_FINAL_CLEAN_MODEL.tar.gz"

tar -C "$ROOT/05_lora" \
-czf "$PKG" \
DEPLOY_FINAL_CLEAN

sha256sum "$PKG" > "$PKG.sha256"

cat > "$ROOT/FINAL_MODEL_STATUS.txt" <<STATUS
ROUTER4 X2DFD FINAL CLEAN

Final Router:
$ROUTER_BEST

Final WFS:
$ROUTED

Final SFT:
$FINAL_SFT

Final LoRA:
$FINAL_MODEL

Deploy:
$FREEZE

Package:
$PKG

STATUS:
FINAL MODEL READY
STATUS

touch "$ROOT/FINAL_MODEL_READY"

echo
echo "============================================================"
echo "✅✅✅ FINAL MODEL READY ✅✅✅"
echo "============================================================"

cat "$ROOT/FINAL_MODEL_STATUS.txt"


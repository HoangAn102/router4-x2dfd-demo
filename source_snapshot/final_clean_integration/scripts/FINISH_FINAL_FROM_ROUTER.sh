#!/usr/bin/env bash
set -euo pipefail

BASE=/home/aiotlab/hoangan
ENV=/home/aiotlab/miniconda3/envs/X2DFD
PY="$ENV/bin/python"

ROOT="$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

ROUTED="$ROOT/03_wfs/router4_FINAL_CLEAN_routed_wfs.csv"
ROUTER_BEST="$ROOT/02_router/final_checkpoint/best.pt"

WRAPPER="$ROOT/scripts/force_final_wfs.py"
ORIGINAL_RUNNER="$ROOT/scripts/run_full_final.sh"

log () {
    echo
    echo "======================================================================"
    echo "$1"
    echo "$(date)"
    echo "======================================================================"
}

fail () {
    echo "[STOP] $1"
    echo "$1" > "$ROOT/FINAL_FINISH_STOP_REASON.txt"
    exit 1
}


###############################################################################
# A. FORCE + VERIFY WFS
###############################################################################

log "A. FORCE FINAL CLEAN WFS"

"$PY" "$WRAPPER" \
    >> "$ROOT/logs/force_final_wfs.log" 2>&1

test -s "$ROUTED" \
    || fail "Final WFS missing"

test -f "$ROOT/03_wfs/WFS_READY" \
    || fail "Final WFS did not pass validation"


###############################################################################
# B. BUILD CONTINUATION: SFT + LORA ONLY
###############################################################################

log "B. PREPARE FINAL SFT + LORA"

DLINE=$(
    grep -n '^# D\. REBUILD CLEAN SFT' \
      "$ORIGINAL_RUNNER" \
    | head -1 \
    | cut -d: -f1
)

FLINE=$(
    grep -n '^# F\. SAME-LIST COMPARISON' \
      "$ORIGINAL_RUNNER" \
    | head -1 \
    | cut -d: -f1
)

ALINE=$(
    grep -n '^# A\. CLEAN' \
      "$ORIGINAL_RUNNER" \
    | head -1 \
    | cut -d: -f1
)

[ -n "$DLINE" ] || fail "Stage D marker missing"
[ -n "$FLINE" ] || fail "Stage F marker missing"
[ -n "$ALINE" ] || fail "Stage A marker missing"

CONT="$ROOT/scripts/FINAL_SFT_LORA_ONLY.sh"

head -n $((ALINE-1)) \
    "$ORIGINAL_RUNNER" \
    > "$CONT"

cat >> "$CONT" <<EOF

ROUTER_BEST="$ROUTER_BEST"
ROUTED="$ROUTED"

echo "============================================================"
echo "FINAL CLEAN SFT -> LORA"
echo "============================================================"
echo "Router: \$ROUTER_BEST"
echo "WFS:    \$ROUTED"
EOF

sed -n \
  "${DLINE},$((FLINE-1))p" \
  "$ORIGINAL_RUNNER" \
  >> "$CONT"


###############################################################################
# C. FREEZE / PACKAGE
###############################################################################

cat >> "$CONT" <<'EOF'

log "FINAL MODEL FREEZE"

FINAL_MODEL="$ROOT/05_lora/router4_x2dfd_FINAL_CLEAN"
FREEZE="$ROOT/05_lora/DEPLOY_FINAL_CLEAN"

test -s "$FINAL_MODEL/adapter_model.safetensors" \
    || fail "final adapter missing"

test -s "$FINAL_MODEL/adapter_config.json" \
    || fail "adapter_config missing"

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
        cp -a "$FINAL_MODEL/$f" "$FREEZE/"
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
ROUTER4 X2DFD FINAL CLEAN MODEL
============================================================

Clean teacher:
$ROOT/01_teacher/train_teacher_FINAL_CLEAN.csv

Clean calibrator:
$ROOT/01_teacher/calibrators_FINAL_CLEAN.joblib

Final Router:
$ROUTER_BEST

Final WFS:
$ROUTED

Final SFT:
$(cat "$ROOT/04_sft/FINAL_SFT_PATH.txt" 2>/dev/null || true)

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

EOF

chmod +x "$CONT"
bash -n "$CONT" \
    || fail "Final continuation syntax error"


###############################################################################
# D. RUN SFT + LORA
###############################################################################

log "C. RUN FINAL SFT + LORA"

bash "$CONT" \
    >> "$ROOT/logs/final_sft_lora.log" 2>&1


###############################################################################
# DONE
###############################################################################

test -f "$ROOT/FINAL_MODEL_READY" \
    || fail "Pipeline ended without FINAL_MODEL_READY"

log "FINAL PIPELINE COMPLETE"

cat "$ROOT/FINAL_MODEL_STATUS.txt"


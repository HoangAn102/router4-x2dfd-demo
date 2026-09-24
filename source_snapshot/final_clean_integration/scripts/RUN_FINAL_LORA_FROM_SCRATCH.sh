#!/usr/bin/env bash
set -euo pipefail

BASE=/home/aiotlab/hoangan

ROOT="$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"
X2="$BASE/projects/X2DFD"

ENV=/home/aiotlab/miniconda3/envs/X2DFD

PY="$ENV/bin/python"
DEEPSPEED="$ENV/bin/deepspeed"

SFT="$ROOT/04_sft/router4_x2dfd_FINAL_CLEAN_SFT.json"

BASE_MODEL="$X2/weights/base/llava-v1.5-7b"
VISION="$X2/weights/base/clip-vit-large-patch14-336"
ZERO3="$X2/train/configs/zero3.json"

OUT="$ROOT/05_lora/router4_x2dfd_FINAL_CLEAN"

export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$X2:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0


###############################################################################
# CUDA LIBS
###############################################################################

LD_EXTRA="$ENV/lib"

while IFS= read -r d; do
    case ":$LD_EXTRA:" in
        *":$d:"*) ;;
        *) LD_EXTRA="$LD_EXTRA:$d" ;;
    esac
done < <(
    find \
      "$ENV/lib/python3.10/site-packages/nvidia" \
      -type d \
      -name lib \
      2>/dev/null || true
)

[ -d /usr/local/cuda/lib64 ] \
    && LD_EXTRA="$LD_EXTRA:/usr/local/cuda/lib64"

export LD_LIBRARY_PATH="$LD_EXTRA:${LD_LIBRARY_PATH:-}"


###############################################################################
# FREE MASTER PORT
###############################################################################

PORT=$(
"$PY" - <<'PY'
import socket
s=socket.socket()
s.bind(("",0))
print(s.getsockname()[1])
s.close()
PY
)


###############################################################################
# STATUS
###############################################################################

echo "======================================================================"
echo "ROUTER4 × X2DFD FINAL LORA"
date
echo "======================================================================"

echo "SFT:"
echo "$SFT"

echo
echo "OUTPUT:"
echo "$OUT"

echo
echo "MASTER PORT:"
echo "$PORT"

echo
echo "GPU:"
nvidia-smi \
  --query-gpu=name,memory.total \
  --format=csv,noheader

echo
echo "======================================================================"
echo "TRAIN START"
echo "======================================================================"

cd "$X2"


###############################################################################
# EXACT V1.3-SUCCESSFUL TRAINING RECIPE
###############################################################################

"$DEEPSPEED" \
  --master_port "$PORT" \
  --include localhost:0 \
  train/model_train.py \
  --lora_enable True \
  --lora_r 16 \
  --lora_alpha 32 \
  --mm_projector_lr 2e-5 \
  --deepspeed "$ZERO3" \
  --model_name_or_path "$BASE_MODEL" \
  --version v1 \
  --data_path "$SFT" \
  --image_folder / \
  --vision_tower "$VISION" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "$OUT" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 3000 \
  --save_total_limit 1 \
  --learning_rate 2e-4 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none


###############################################################################
# HARD SUCCESS CHECK
###############################################################################

echo
echo "======================================================================"
echo "TRAIN PROCESS EXITED"
echo "======================================================================"

if [ ! -s "$OUT/adapter_model.safetensors" ]; then
    echo "FINAL TRAIN EXITED WITHOUT adapter_model.safetensors" \
      > "$ROOT/FINAL_LORA_STOP_REASON.txt"

    echo "❌ adapter_model.safetensors missing"
    exit 50
fi

if [ ! -s "$OUT/adapter_config.json" ]; then
    echo "FINAL TRAIN EXITED WITHOUT adapter_config.json" \
      > "$ROOT/FINAL_LORA_STOP_REASON.txt"

    echo "❌ adapter_config.json missing"
    exit 51
fi


###############################################################################
# HASH FINAL ADAPTER
###############################################################################

sha256sum \
  "$OUT/adapter_model.safetensors" \
  > "$OUT/adapter_model.sha256"

touch "$OUT/MODEL_READY"


###############################################################################
# FREEZE
###############################################################################

FREEZE="$ROOT/05_lora/DEPLOY_FINAL_CLEAN"

rm -rf "$FREEZE"

mkdir -p \
  "$FREEZE/lora" \
  "$FREEZE/router" \
  "$FREEZE/calibrator"


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
    [ -e "$OUT/$f" ] \
      && cp -a "$OUT/$f" "$FREEZE/lora/"
done


cp -a \
  "$ROOT/02_router/final_checkpoint/best.pt" \
  "$FREEZE/router/best.pt"


cp -a \
  "$ROOT/01_teacher/calibrators_FINAL_CLEAN.joblib" \
  "$FREEZE/calibrator/calibrators_FINAL_CLEAN.joblib"


###############################################################################
# MANIFEST
###############################################################################

cat > "$FREEZE/PIPELINE_MANIFEST.txt" <<EOF
ROUTER4 X2DFD FINAL CLEAN

Final SFT:
$SFT

Final Router:
$ROOT/02_router/final_checkpoint/best.pt

Final Calibrator:
$ROOT/01_teacher/calibrators_FINAL_CLEAN.joblib

Final WFS:
$ROOT/03_wfs/router4_FINAL_CLEAN_routed_wfs.csv

Final LoRA:
$OUT

Base LLaVA:
$BASE_MODEL

Vision tower:
$VISION

NOTE:
Base LLaVA and expert model weights are external dependencies
and are not duplicated into this package.
EOF


touch "$FREEZE/FINAL_MODEL_READY"


###############################################################################
# PACKAGE
###############################################################################

PKG="$ROOT/05_lora/ROUTER4_FINAL_CLEAN_CORE.tar.gz"

rm -f "$PKG"

tar -C "$ROOT/05_lora" \
  -czf "$PKG" \
  DEPLOY_FINAL_CLEAN


sha256sum "$PKG" \
  > "$PKG.sha256"


###############################################################################
# FINAL STATUS
###############################################################################

cat > "$ROOT/FINAL_MODEL_STATUS.txt" <<EOF
============================================================
ROUTER4 × X2DFD FINAL CLEAN
============================================================

Final SFT:
$SFT

Final Router:
$ROOT/02_router/final_checkpoint/best.pt

Final WFS:
$ROOT/03_wfs/router4_FINAL_CLEAN_routed_wfs.csv

Final LoRA:
$OUT

Frozen core:
$FREEZE

Package:
$PKG

STATUS:
FINAL MODEL READY
EOF


touch "$ROOT/FINAL_MODEL_READY"


echo
echo "======================================================================"
echo "✅✅✅ FINAL MODEL READY ✅✅✅"
echo "======================================================================"

cat "$ROOT/FINAL_MODEL_STATUS.txt"


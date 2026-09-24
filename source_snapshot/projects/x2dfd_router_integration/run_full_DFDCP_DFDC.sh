#!/usr/bin/env bash

BASE=/home/aiotlab/hoangan

X2="$BASE/projects/X2DFD"
INT="$BASE/projects/x2dfd_router_integration"
OUT="$BASE/outputs/x2dfd_router_integration/full_DFDCP_DFDC"

PY_X2=/home/aiotlab/miniconda3/envs/X2DFD/bin/python
PY_FREQ=/home/aiotlab/miniconda3/envs/DFFreq/bin/python
PY_TEX=/home/aiotlab/miniconda3/envs/TextureExpert/bin/python

RUN_X2="$BASE/router8_x2crop/run_x2.py"
RUN_FREQ="$BASE/router8_x2crop/run_frequency.py"
RUN_TEX="$BASE/router8_x2crop/run_texture.py"

DFF="$BASE/projects/DFFreq-main"

TEX="$BASE/projects/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild/stylegan-ffhq"


echo "============================================================"
echo "FULL DFDCP + DFDC"
echo "X2DFD ORIGINAL vs ROUTER-MOE"
echo "============================================================"

echo ""
echo "Images:"
tail -n +2 "$OUT/manifest.csv" | wc -l


# ============================================================
# EXPERT 1 — BLENDING
# Resume supported by CSV.
# ============================================================

echo ""
echo "===== EXPERT 1/4: BLENDING ====="

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
"$RUN_X2" \
--out "$OUT" \
--expert blending


# ============================================================
# EXPERT 2 — DIFFUSION
# ============================================================

echo ""
echo "===== EXPERT 2/4: DIFFUSION ====="

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
"$RUN_X2" \
--out "$OUT" \
--expert diffusion


# ============================================================
# EXPERT 3 — FREQUENCY
# ============================================================

echo ""
echo "===== EXPERT 3/4: FREQUENCY ====="

cd "$DFF"

PYTHONPATH="$DFF:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=0 \
"$PY_FREQ" -u \
"$RUN_FREQ" \
--out "$OUT" \
--name X2DFD_FULL_DFDCP_DFDC


# ============================================================
# EXPERT 4 — TEXTURE
# ============================================================

echo ""
echo "===== EXPERT 4/4: TEXTURE ====="

CUDA_VISIBLE_DEVICES=0 \
"$PY_TEX" -u \
"$RUN_TEX" \
--out "$OUT" \
--work "$TEX"


# ============================================================
# VERIFY 4 CSVs
# ============================================================

echo ""
echo "===== VERIFY EXPERT CSVs ====="

EXPECTED=$(tail -n +2 "$OUT/manifest.csv" | wc -l)

for E in blending diffusion frequency texture
do
    CSV="$OUT/${E}_scores.csv"

    if [ -f "$CSV" ]; then
        N=$(tail -n +2 "$CSV" | wc -l)
    else
        N=0
    fi

    printf "%-10s %8d / %d\n" \
        "$E" \
        "$N" \
        "$EXPECTED"
done


# ============================================================
# ROUTER
# ============================================================

echo ""
echo "===== BUILD ROUTER-MOE CACHE ====="

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
"$INT/build_router_moe_cache_full.py"


# ============================================================
# CONFIGS
# ============================================================

echo ""
echo "===== BUILD FULL X2DFD CONFIGS ====="

"$PY_X2" \
"$INT/make_full_configs.py"


# ============================================================
# X2DFD ORIGINAL
# ============================================================

echo ""
echo "============================================================"
echo "X2DFD ORIGINAL — FULL"
echo "============================================================"

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
-m eval.infer.runner \
--config \
eval/configs/infer_full_DFDCP_DFDC_original.yaml \
2>&1 | tee \
"$OUT/x2dfd_original.log"


# Preserve run pointer.
if [ -f "$X2/eval/outputs/infer/latest_run.json" ]
then
    cp \
    "$X2/eval/outputs/infer/latest_run.json" \
    "$OUT/latest_run_original.json"
fi


# ============================================================
# X2DFD ROUTER-MOE
# ============================================================

echo ""
echo "============================================================"
echo "X2DFD + ROUTER-MOE — FULL"
echo "============================================================"

cd "$X2"

CUDA_VISIBLE_DEVICES=0 \
"$PY_X2" -u \
-m eval.infer.runner \
--config \
eval/configs/infer_full_DFDCP_DFDC_router_moe.yaml \
2>&1 | tee \
"$OUT/x2dfd_router_moe.log"


if [ -f "$X2/eval/outputs/infer/latest_run.json" ]
then
    cp \
    "$X2/eval/outputs/infer/latest_run.json" \
    "$OUT/latest_run_router_moe.json"
fi


echo ""
echo "============================================================"
echo "FULL PIPELINE FINISHED ✅"
echo "============================================================"

echo "Output:"
echo "$OUT"

exec bash

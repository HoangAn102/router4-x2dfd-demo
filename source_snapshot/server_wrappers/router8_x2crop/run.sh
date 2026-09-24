#!/usr/bin/env bash

BASE=/home/aiotlab/hoangan
WORK="$BASE/router8_x2crop"

CONDA=/home/aiotlab/miniconda3/etc/profile.d/conda.sh

DB="$BASE/projects/DeepfakeBench"

GAN="$BASE/outputs/expert_profiling/full/GANGen"

FF="$BASE/outputs/expert_profiling/full/FFPP_X2DFD32"

X2="$BASE/projects/X2DFD"

DFF="$BASE/projects/DFFreq-main"

GRAM="$BASE/projects/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild"

GRAMWORK="$GRAM/stylegan-ffhq"


echo
echo "============================================================"
echo "X2DFD / DEEPFAKEBENCH 32-FRAME ROUTER PROFILING"
date
echo "============================================================"


# ------------------------------------------------------------
# DeepfakeBench repository
# ------------------------------------------------------------

if [ ! -f "$DB/preprocessing/preprocess.py" ]; then

    echo "Cloning DeepfakeBench..."

    git clone \
      https://github.com/SCLBD/DeepfakeBench.git \
      "$DB"
fi


# ------------------------------------------------------------
# Landmark predictor used by DeepfakeBench
# ------------------------------------------------------------

PRED="$DB/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

if [ ! -f "$PRED" ]; then

    echo
    echo "Downloading official DeepfakeBench 81-landmark predictor..."

    mkdir -p "$(dirname "$PRED")"

    curl -L \
      --retry 5 \
      --retry-delay 5 \
      -o "$PRED" \
      "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat"
fi


echo
echo "Predictor:"
ls -lh "$PRED"


# ============================================================
# STAGE 1 - FF++ OFFICIAL X2DFD/DEEPFAKEBENCH CROP
# ============================================================

echo
echo "============================================================"
echo "STAGE 1/6 - FF++ X2DFD 32-FRAME FACE CROP"
echo "============================================================"

source "$CONDA"
conda activate FFPPCrop

cd "$DB/preprocessing"

python "$WORK/crop_ffpp_x2dfd.py"


# ============================================================
# STAGE 2 - BÙ FREQUENCY GANGEN
# ============================================================

echo
echo "============================================================"
echo "STAGE 2/6 - GANGEN FREQUENCY"
echo "============================================================"

source "$CONDA"
conda activate DFFreq

cd "$DFF"

python "$WORK/run_frequency.py" \
    --out "$GAN" \
    --name GANGEN


# ============================================================
# STAGE 3 - FF++ BLENDING
# ============================================================

echo
echo "============================================================"
echo "STAGE 3/6 - FF++ BLENDING"
echo "============================================================"

source "$CONDA"
conda activate X2DFD

cd "$X2"

python "$WORK/run_x2.py" \
    --out "$FF" \
    --expert blending


# ============================================================
# STAGE 4 - FF++ DIFFUSION
# ============================================================

echo
echo "============================================================"
echo "STAGE 4/6 - FF++ DIFFUSION"
echo "============================================================"

source "$CONDA"
conda activate X2DFD

cd "$X2"

python "$WORK/run_x2.py" \
    --out "$FF" \
    --expert diffusion


# ============================================================
# STAGE 5 - FF++ FREQUENCY
# ============================================================

echo
echo "============================================================"
echo "STAGE 5/6 - FF++ FREQUENCY"
echo "============================================================"

source "$CONDA"
conda activate DFFreq

cd "$DFF"

python "$WORK/run_frequency.py" \
    --out "$FF" \
    --name FFPP_X2DFD32


# ============================================================
# STAGE 6 - FF++ TEXTURE
# ============================================================

echo
echo "============================================================"
echo "STAGE 6/6 - FF++ TEXTURE"
echo "============================================================"

source "$CONDA"
conda activate TextureExpert

cd "$GRAMWORK"

python "$WORK/run_texture.py" \
    --out "$FF" \
    --work "$GRAMWORK"


# ============================================================
# VERIFY
# ============================================================

echo
echo "============================================================"
echo "FINAL VERIFY"
echo "============================================================"

python "$WORK/verify.py"

echo
echo "============================================================"
echo "PIPELINE FINISHED"
date
echo "============================================================"
